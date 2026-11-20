"""Train LeRobot ACT on archived HiddenPhys demonstrations and serve action chunks."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from itertools import accumulate
import json
from multiprocessing.connection import Listener
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from forceharness.data.demos import (
    action_matrix, load_demo, pack, prep_jpeg, proprio, scene_vector, unpack, wrench_obs,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMERAS = ("agentview", "robot0_eye_in_hand")
CAMERA_KEYS = {camera: f"observation.images.{camera}" for camera in CAMERAS}
HISTORY = 8
IMAGE_SIZE = 128
ACTION_DIM = 13
AUTH = b"fh-dp"
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
cv2.setNumThreads(0)


def fold_wrench(rows, mode):
    """Rows are trailing control ticks, from oldest through the current one."""
    current = rows[-1]
    force = rows[:, 11:17] if mode == "history" else np.broadcast_to(current[11:17], (HISTORY, 6))
    return np.concatenate((current[:11], force.reshape(-1), current[17:]))


def image_batch(images, device):
    return {
        CAMERA_KEYS[camera]: (images[camera].float().div(255).sub(IMAGE_MEAN).div(IMAGE_STD)).to(
            device, non_blocking=True
        )
        for camera in CAMERAS
    }


class PegDemos(Dataset):
    def __init__(self, directory, chunk, mode, limit):
        self.mode = mode
        self.chunk_offsets = np.arange(chunk)
        paths = sorted(directory.glob("episode_*.npz"))
        if not paths or (limit and len(paths) < limit):
            raise ValueError(f"Need {limit or 'at least one'} demonstrations, found {len(paths)} in {directory}")
        self.episodes = []
        self.scene_keys = None
        for path in (paths[:limit] if limit else paths):
            demo = load_demo(str(path))
            meta = demo["meta"]
            if meta["task"] != "peg_in_hole_hard":
                raise ValueError(f"Unexpected task in {path}")
            actions, osc_dim = action_matrix(demo)
            if osc_dim != 7 or len(demo["t"]) != len(actions) + 1:
                raise ValueError(f"Unexpected action or time alignment in {path}")
            scene, self.scene_keys = scene_vector(meta.get("scene0"), self.scene_keys)
            low = np.column_stack((
                proprio(demo["eef_pos"], demo["eef_quat"], demo["gripper_qpos"]),
                wrench_obs(demo, int(meta["seed"])),
                np.broadcast_to(scene, (len(demo["t"]), len(scene))),
            ))
            self.episodes.append((low[:-1], actions, demo["jpeg"], int(meta["seed"])))
        self.ends = list(accumulate(len(actions) for _, actions, _, _ in self.episodes))
        low = np.concatenate([entry[0] for entry in self.episodes])
        act = np.concatenate([entry[1] for entry in self.episodes])
        if not (np.isfinite(low).all() and np.isfinite(act).all()):
            raise ValueError("Nonfinite demonstration values")
        low_stats = lambda values: fold_wrench(np.repeat(values[None], HISTORY, axis=0), "current")
        self.state_mean = low_stats(low.mean(axis=0))
        self.state_std = low_stats(low.std(axis=0).clip(1e-4))
        self.act_mean = act.mean(axis=0)
        self.act_std = act.std(axis=0).clip(1e-4)
        self.act_bounds = {"lo": act.min(axis=0).tolist(), "hi": act.max(axis=0).tolist()}

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode = bisect_right(self.ends, index)
        tick = index - (self.ends[episode - 1] if episode else 0)
        low, actions, jpeg, _ = self.episodes[episode]
        time_range = np.arange(tick - HISTORY + 1, tick + 1).clip(0)
        state = fold_wrench(low[time_range], self.mode)
        action_idx = tick + self.chunk_offsets
        images = {}
        for camera in CAMERAS:
            payload, offsets = jpeg[camera]
            images[camera] = prep_jpeg(payload[offsets[tick]:offsets[tick + 1]], IMAGE_SIZE, IMAGE_SIZE)
        return {
            "state": (state - self.state_mean) / self.state_std,
            "images": images,
            "action": (actions[action_idx.clip(max=len(actions) - 1)] - self.act_mean) / self.act_std,
            "action_is_pad": action_idx >= len(actions),
        }


def make_policy(device, state_dim, chunk, execute, initialization=None):
    config = ACTConfig(
        device=device.type, push_to_hub=False, chunk_size=chunk, n_action_steps=execute,
        dim_model=256, n_heads=8, dim_feedforward=1024, n_encoder_layers=3,
        n_vae_encoder_layers=3, pretrained_backbone_weights=initialization,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, IMAGE_SIZE, IMAGE_SIZE))
               for key in CAMERA_KEYS.values()},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
    )
    return ACTPolicy(config).to(device)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dataset = PegDemos(args.demos, args.chunk, args.mode, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "checkpoint.pt").exists():
        raise FileExistsError(args.out / "checkpoint.pt")
    metadata = dict(
        mode=args.mode, scene=True, scene_keys=dataset.scene_keys, force=True,
        osc_dim=7, To=HISTORY, Ta=args.execute, Tp=args.chunk, cams=list(CAMERAS),
        max_T=max(len(ep[1]) for ep in dataset.episodes), act_norm=dataset.act_bounds,
        train_seeds=[ep[3] for ep in dataset.episodes], steps=args.steps,
        batch=args.batch, seed=args.seed, lr=args.lr, imagenet=args.imagenet,
    )
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{args.mode}: {len(dataset.episodes)} episodes, {len(dataset)} frames, {device}", flush=True)
    dl = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    if not len(dl):
        raise ValueError("Batch exceeds available frames")
    init = "ResNet18_Weights.IMAGENET1K_V1" if args.imagenet else None
    model = make_policy(device, len(dataset.state_mean), args.chunk, args.execute, init)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("model.backbone")]},
        {"params": model.model.backbone.parameters(), "lr": args.lr * (0.3 if init else 1)},
    ], lr=args.lr, weight_decay=1e-4, fused=device.type == "cuda")
    step = 0
    while step < args.steps:
        for data in dl:
            batch = image_batch(data["images"], device)
            batch["observation.state"] = data["state"].to(device, non_blocking=True)
            batch["action"] = data["action"].to(device, non_blocking=True)
            batch["action_is_pad"] = data["action_is_pad"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(f"{args.mode} step={step}/{args.steps} loss={loss.item():.5f}", flush=True)
            if step == args.steps:
                break
    dest = args.out / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(), "metadata": metadata,
        **{name: torch.from_numpy(getattr(dataset, name))
           for name in ("state_mean", "state_std", "act_mean", "act_std")},
    }, dest.with_suffix(".tmp"))
    dest.with_suffix(".tmp").rename(dest)
    model.save_pretrained(args.out / "lerobot_policy")
    print(f"Saved {dest}", flush=True)


def serve(args):
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = ck["metadata"]
    model = make_policy(device, len(ck["state_mean"]), meta["Tp"], meta["Ta"])
    model.load_state_dict(ck["model"])
    model.eval()
    info = dict(meta, ckpt=str(args.checkpoint), ckpt_step=meta["steps"])
    stats = {k: ck[k].to(device) for k in ("state_mean", "state_std", "act_mean", "act_std")}
    lock = threading.Lock()

    def session(conn):
        with conn:
            try:
                while True:
                    message = conn.recv()
                    if message["cmd"] == "config":
                        conn.send(json.dumps(info))
                        continue
                    if message["cmd"] != "act" or message["task"] != "peg_in_hole_hard":
                        raise ValueError("Unexpected request from evaluator")
                    obs = unpack(message["low"])
                    value = fold_wrench(obs, meta["mode"])
                    state = (torch.from_numpy(value).to(device) - stats["state_mean"]) / stats["state_std"]
                    images = {c: torch.from_numpy(prep_jpeg(
                        message["jpeg"][c][-1], IMAGE_SIZE, IMAGE_SIZE
                    ))[None] for c in CAMERAS}
                    batch = image_batch(images, device)
                    batch["observation.state"] = state[None]
                    with lock, torch.inference_mode():
                        action = model.predict_action_chunk(batch)[0]
                        action = action * stats["act_std"] + stats["act_mean"]
                    conn.send(pack(action.cpu().numpy()))
            except (EOFError, ConnectionResetError, BrokenPipeError):
                pass

    with Listener(("127.0.0.1", args.port), authkey=AUTH) as listener:
        print(f"READY {args.port} {args.checkpoint}", flush=True)
        while True:
            threading.Thread(target=session, args=(listener.accept(),), daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="command", required=True)
    t = modes.add_parser("train")
    t.add_argument("--demos", type=Path, default=Path("demos"))
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--mode", choices=("current", "history"), required=True)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--limit", type=int, default=185)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--chunk", type=int, default=24)
    t.add_argument("--execute", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--imagenet", action="store_true")
    t.set_defaults(func=train)
    s = modes.add_parser("serve")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--port", type=int, required=True)
    s.set_defaults(func=serve)
    for sub in (t, s):
        sub.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
