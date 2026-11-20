"""Summarize complete, paired HiddenPhys evaluation runs."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def load(path, seeds):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    relevant = [row for row in rows if row["seed"] in seeds]
    results = {row["seed"]: row for row in relevant}
    missing = sorted(seeds - results.keys())
    errors = {seed: row.get("error") for seed, row in results.items()
              if row.get("error") or not isinstance(row.get("success"), bool)
              or not isinstance(row.get("peak_wrench_N"), (int, float))}
    if missing or errors or len(relevant) != len(results):
        raise ValueError(f"{path}: missing={missing}, invalid={errors}, duplicates={len(relevant) - len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    start, end = map(int, args.seeds.split("-"))
    seeds = set(range(start, end + 1))
    runs = [load(path, seeds) for path in args.results]
    for path, run in zip(args.results, runs, strict=True):
        force = [row["peak_wrench_N"] for row in run.values()]
        successes = sum(row["success"] for row in run.values())
        print(f"{path.parent.parent.name}: {successes}/{len(run)} "
              f"({100 * successes / len(run):.1f}%), "
              f"peak force mean={mean(force):.2f} N median={median(force):.2f} N")
    if len(runs) == 2:
        current, history = runs
        wins = sum(history[s]["success"] and not current[s]["success"] for s in seeds)
        losses = sum(current[s]["success"] and not history[s]["success"] for s in seeds)
        print(f"History vs current: {wins} history-only successes, {losses} current-only successes, "
              f"{100 * (wins - losses) / len(seeds):+.1f} pp")


if __name__ == "__main__":
    main()
