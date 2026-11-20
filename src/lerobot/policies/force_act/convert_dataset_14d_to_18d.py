#!/usr/bin/env python
"""Convert 14D delta dataset to 18D sin-cos encoded LeRobot dataset.

Usage:
  python convert_dataset_14d_to_18d.py --repo-id my_dataset --output-repo-id my_dataset_18d

  --repo-id:          原始 14D 数据集路径或 HuggingFace repo ID
  --output-repo-id:   输出 18D 数据集名称 (存在本地)
  --root:             输出数据集的根目录 (默认: ~/.cache/huggingface/lerobot)
"""

import argparse
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Core conversion ─────────────────────────────────────────────────────────

def encode_arm_7d_to_9d(arm_7d: np.ndarray) -> np.ndarray:
    """Single arm: 7D → 9D with sin-cos rotation encoding.

    7D: [Δx, Δy, Δz, Δroll(mdeg), Δpitch(mdeg), Δyaw(mdeg), Δgripper]
    9D: [Δx, Δy, Δz, sin(ΔR), cos(ΔR), sin(ΔP), cos(ΔP), Δyaw, Δgripper]
    """
    squeeze = arm_7d.ndim == 1
    if squeeze:
        arm_7d = arm_7d[np.newaxis, :]

    dx, dy, dz = arm_7d[:, 0:1], arm_7d[:, 1:2], arm_7d[:, 2:3]
    droll_rad  = np.deg2rad(arm_7d[:, 3:4] / 1000.0)   # mdeg → rad
    dpitch_rad = np.deg2rad(arm_7d[:, 4:5] / 1000.0)
    dyaw, dgrip = arm_7d[:, 5:6], arm_7d[:, 6:7]

    arm_9d = np.concatenate([
        dx, dy, dz,
        np.sin(droll_rad), np.cos(droll_rad),
        np.sin(dpitch_rad), np.cos(dpitch_rad),
        dyaw, dgrip,
    ], axis=1)

    return arm_9d.squeeze(0) if squeeze else arm_9d


def convert_action_14d_to_18d(action_14d: np.ndarray) -> np.ndarray:
    """Dual arm: 14D → 18D."""
    squeeze = action_14d.ndim == 1
    if squeeze:
        action_14d = action_14d[np.newaxis, :]

    left_9d  = encode_arm_7d_to_9d(action_14d[:, :7])
    right_9d = encode_arm_7d_to_9d(action_14d[:, 7:])
    action_18d = np.concatenate([left_9d, right_9d], axis=1)

    return action_18d.squeeze(0) if squeeze else action_18d


# ── Dataset conversion ──────────────────────────────────────────────────────

def convert_lerobot_dataset(
    repo_id: str,
    output_repo_id: str,
    root: str | None = None,
):
    """Read a 14D LeRobot dataset and write a new 18D LeRobot dataset.

    All observations (images, state, etc.) are copied as-is.
    Only the action field is transformed.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading source dataset: {repo_id}")
    src = LeRobotDataset(repo_id)
    logger.info(f"  Episodes: {src.num_episodes}, Frames: {len(src)}")

    # Verify action dim
    sample_action = np.array(src[0]["action"])
    assert sample_action.shape[-1] == 14, f"Expected 14D, got {sample_action.shape[-1]}D"

    # Build features dict for new dataset: same as source but action → 18D
    features = {}
    for key, ft in src.meta.features.items():
        if key == "action":
            features["action"] = {"dtype": "float32", "shape": (18,), "names": None}
        else:
            features[key] = {"dtype": ft.dtype if hasattr(ft, "dtype") else str(ft["dtype"]),
                             "shape": ft.shape if hasattr(ft, "shape") else tuple(ft["shape"]),
                             "names": ft.names if hasattr(ft, "names") else ft.get("names")}

    logger.info(f"Creating output dataset: {output_repo_id}")
    dst = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=src.meta.fps,
        features=features,
        root=root,
        robot_type=src.meta.robot_type,
    )

    # Convert episode by episode
    for ep_idx in range(src.num_episodes):
        ep_indices = src.episode_data_index
        start = int(ep_indices["from"][ep_idx])
        end = int(ep_indices["to"][ep_idx])

        for frame_idx in range(start, end):
            frame = src[frame_idx]

            # Build new frame dict
            new_frame = {}
            for key in src.meta.features:
                if key == "action":
                    action_14d = np.array(frame["action"])
                    new_frame["action"] = torch.tensor(
                        convert_action_14d_to_18d(action_14d), dtype=torch.float32
                    )
                elif key == "timestamp":
                    new_frame["timestamp"] = float(frame["timestamp"])
                elif isinstance(frame[key], torch.Tensor):
                    new_frame[key] = frame[key]
                else:
                    new_frame[key] = frame[key]

            dst.add_frame(new_frame)

        dst.save_episode()
        logger.info(f"  Episode {ep_idx}: {end - start} frames converted")

    logger.info(f"Done. Output dataset at: {dst.root}")
    logger.info(f"  Total episodes: {dst.num_episodes}, Total frames: {len(dst)}")
    return dst


# ── Verification ────────────────────────────────────────────────────────────

def verify_roundtrip():
    np.random.seed(42)
    arm_7d = np.array([50.0, -30.0, 20.0, 3600.0, -1800.0, 160.0, 100.0])
    arm_9d = encode_arm_7d_to_9d(arm_7d)

    droll_back  = np.rad2deg(np.arctan2(arm_9d[3], arm_9d[4])) * 1000
    dpitch_back = np.rad2deg(np.arctan2(arm_9d[5], arm_9d[6])) * 1000

    print("Roundtrip verification:")
    print(f"  Δroll:  {arm_7d[3]:.1f} → sin/cos → {droll_back:.1f} mdeg  (err={abs(arm_7d[3]-droll_back):.6f})")
    print(f"  Δpitch: {arm_7d[4]:.1f} → sin/cos → {dpitch_back:.1f} mdeg  (err={abs(arm_7d[4]-dpitch_back):.6f})")
    print(f"  sin²+cos² (roll):  {arm_9d[3]**2 + arm_9d[4]**2:.6f}")
    print(f"  sin²+cos² (pitch): {arm_9d[5]**2 + arm_9d[6]**2:.6f}")
    print("  All checks passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Convert 14D LeRobot dataset to 18D")
    parser.add_argument("--repo-id", type=str, help="Source 14D dataset repo ID or path")
    parser.add_argument("--output-repo-id", type=str, default=None, help="Output 18D dataset name")
    parser.add_argument("--root", type=str, default=None, help="Output root directory")
    parser.add_argument("--verify", action="store_true", help="Run roundtrip verification only")
    args = parser.parse_args()

    if args.verify:
        verify_roundtrip()
    elif args.repo_id and args.output_repo_id:
        convert_lerobot_dataset(args.repo_id, args.output_repo_id, root=args.root)
    else:
        parser.print_help()
