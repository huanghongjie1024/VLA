"""Classify the original failures using the recorded replay geometry."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
VIDEO_SEED = 9106
LABELS = {
    "lost_pass": "中途达到插入判据，结束时未保持",
    "shallow": "出现近轴下探，未达到判定深度",
    "contact_only": "接触孔块，未出现近轴下探",
    "no_contact": "未接触孔块，也未出现近轴下探",
}


def result_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    out = {row["seed"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate seeds: {path}")
    return out


def inspect(seed, branch, original, replay):
    if replay.get("error") or replay["ckpt_step"] != original["ckpt_step"]:
        raise ValueError(f"Invalid replay {branch} {seed}: {replay.get('error')}")
    with np.load(ROOT / branch / f"seed_{seed}.npz", allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    if not np.array_equal(trace["tick"], np.arange(1, original["n_steps"] + 1)):
        raise ValueError(f"Incomplete replay trace: {branch} {seed}")
    radial, depth, tip = (trace[n] for n in ("center_error_mm", "benchmark_depth_mm",
                                            "physical_tip_depth_mm"))
    aligned = radial < 7.5
    passed = trace["success"].astype(bool)
    if not np.array_equal(passed, aligned & (depth > 20)):
        raise ValueError(f"Trace does not match official geometry: {branch} {seed}")
    if bool(passed[-1]) != replay["success"]:
        raise ValueError(f"Trace does not match official replay verdict: {branch} {seed}")
    touched = trace["peg_hole_contact"].astype(bool)
    # A proxy for entry, not a claim that all four corners physically fit.
    near_entry = aligned & (tip > 2.0) & (trace["peg_tilt_deg"] <= 25.0)
    released = ~trace["weld"].astype(bool)

    def first(mask):
        return int(trace["tick"][np.flatnonzero(mask)[0]]) if mask.any() else None

    first_pass, first_release = first(passed), first(released)
    if original["success"] != replay["success"]:
        label = "verdict_changed"
    elif passed[-1]:
        label = "pass"
    elif passed.any():
        label = "lost_pass"
    elif near_entry.any():
        label = "shallow"
    elif touched.any():
        label = "contact_only"
    else:
        label = "no_contact"

    return dict(
        seed=seed, label=label, final_error=float(radial[-1]), final_depth=float(depth[-1]),
        max_aligned_depth=float(depth[aligned].max()) if aligned.any() else None,
        min_error=float(radial.min()), first_contact=first(touched), first_entry=first(near_entry),
        first_pass=first_pass, first_release=first_release, pass_ticks=int(passed.sum()),
        early_release=first_release is not None and (first_pass is None or first_release < first_pass),
        contact_ticks=int(touched.sum()), final_aligned=bool(aligned[-1]),
        peak_delta=round(replay["peak_wrench_N"] - original["peak_wrench_N"], 2),
    )


def times_table(row):
    times = []
    for key in ("first_contact", "first_entry", "first_pass", "first_release"):
        tick = row[key]
        times.append(f"{tick / 20:.2f}" if tick is not None else "—")
    return " | ".join(times)


def main():
    branches = {}
    for branch in ("current", "history"):
        original = result_rows(PROJECT / f"runs/paired/{branch}/eval/results.jsonl")
        replay = result_rows(ROOT / branch / "results.jsonl")
        seeds = {VIDEO_SEED}
        if branch == "history":
            seeds |= {seed for seed, row in original.items() if not row["success"]}
        if replay.keys() != seeds:
            raise ValueError(f"Unexpected replay seed set for {branch}: {replay.keys() ^ seeds}")
        branches[branch] = {seed: inspect(seed, branch, original[seed], replay[seed]) for seed in sorted(seeds)}

    failures = [row for seed, row in branches["history"].items() if seed != VIDEO_SEED]
    if len(failures) != 39:
        raise ValueError("Expected 39 original history failures")
    counts = Counter(row["label"] for row in failures)
    changed_peaks = sum(row["peak_delta"] != 0 for row in failures)
    early = sum(row["early_release"] for row in failures)
    final_aligned = sum(row["final_aligned"] for row in failures)
    lines = [
        "# HiddenPhys 历史力组失败阶段与同种子回放", "",
        "使用原 5,000 步 checkpoint、相同种子及原评估控制循环做推理回放。诊断数据只用于观察。", "",
        "| 39 局原始失败的回放分类 | 局数 |", "|---|---:|",
    ]
    lines += [f"| {label} | {counts[key]} |" for key, label in LABELS.items()]
    if counts["verdict_changed"]:
        lines += [f"| 回放成败与原结果不同 | {counts['verdict_changed']} |"]
    lines += [
        "",
        f"- {early} 局在首次达标前解除抓持焊接，包含从未达标的回合。这一动作可与表中阶段重叠。",
        f"- {final_aligned} 局结束时销中心仍在官方横向阈值内。",
        f"- {changed_peaks} 局的回放峰值力与原日志不同，以 0.01 N 的精度比较。",
        "",
        "官方成功判据为销中心到孔轴的平面偏差小于 7.5 mm，且基准深度超过 20 mm，最后一个控制步判定成败。"
        "近轴下探要求同一步偏差小于 7.5 mm、估算销端越过物理孔顶超过 2 mm、倾角不超过 25°。"
        "物理孔顶比官方深度参考面高 20 mm，视频显示官方深度。", "",
        "## 并排回放（种子 9106）", "",
        "打开 seed_9106_current_vs_history.mp4。左半为当前力，右半为历史力，每半左为外部相机、右为腕部相机。"
        "视频以 4 倍速播放，画面时间为仿真控制时间。", "",
        "| 模型 | 首次接触孔块 (s) | 首次近轴下探 (s) | 首次达标 (s) | 首次松夹 (s) | 最终偏差 (mm) | 最终基准深度 (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch, rows in branches.items():
        row = rows[VIDEO_SEED]
        lines.append(f"| {branch} | {times_table(row)} | {row['final_error']:.2f} | {row['final_depth']:.2f} |")
    lines += ["", "## 历史力组逐局", "",
              "| 种子 | 阶段 | 首次接触 (s) | 首次下探 (s) | 首次达标 (s) | 首次松夹 (s) | 最终偏差 (mm) | 最终基准深度 (mm) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in failures:
        label = LABELS.get(row["label"], "回放结果不一致")
        lines.append(f"| {row['seed']} | {label} | {times_table(row)} | {row['final_error']:.2f} | {row['final_depth']:.2f} |")
    (ROOT / "失败阶段分析.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(counts=counts, changed_peaks=changed_peaks, early_release=early,
                          final_aligned=final_aligned, examples={b:r[VIDEO_SEED] for b,r in branches.items()}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
