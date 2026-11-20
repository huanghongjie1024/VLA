#!/usr/bin/env python
"""Environment adapter for Force-ACT policy.

Converts 18D model output to 14D hardware delta commands and dispatches
to the robot via HAL client.

Model output (18D):
  Left arm [0-8]:   Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper
  Right arm [9-17]: Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper

Hardware input (14D):
  Left arm:  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
  Right arm: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]

Units:
  Δx/Δy/Δz: μm
  Δroll/Δpitch/Δyaw: mdeg (0.001 degree)
  Δgripper: native units
"""

import torch
import numpy as np


# ── Action decode: 18D → 14D ───────────────────────────────────────────────

def decode_arm_9d_to_7d(arm_9d: np.ndarray) -> np.ndarray:
    """Convert single arm action from 9D sin-cos to 7D with arctan2.

    Args:
        arm_9d: shape (9,) or (T, 9)
            [Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper]

    Returns:
        arm_7d: shape (7,) or (T, 7)
            [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
            Δroll/Δpitch in mdeg.
    """
    squeeze = False
    if arm_9d.ndim == 1:
        arm_9d = arm_9d[np.newaxis, :]
        squeeze = True

    dx     = arm_9d[:, 0:1]
    dy     = arm_9d[:, 1:2]
    dz     = arm_9d[:, 2:3]

    # arctan2: sin/cos → radians → mdeg
    droll_rad  = np.arctan2(arm_9d[:, 3], arm_9d[:, 4])
    dpitch_rad = np.arctan2(arm_9d[:, 5], arm_9d[:, 6])
    droll_mdeg  = (np.rad2deg(droll_rad) * 1000.0).reshape(-1, 1)
    dpitch_mdeg = (np.rad2deg(dpitch_rad) * 1000.0).reshape(-1, 1)

    dyaw  = arm_9d[:, 7:8]
    dgrip = arm_9d[:, 8:9]

    arm_7d = np.concatenate([
        dx, dy, dz,
        droll_mdeg, dpitch_mdeg,
        dyaw, dgrip,
    ], axis=1)

    if squeeze:
        arm_7d = arm_7d.squeeze(0)
    return arm_7d


def decode_action_18d_to_14d(action_18d: np.ndarray) -> np.ndarray:
    """Convert full dual-arm action from 18D to 14D.

    Args:
        action_18d: shape (18,) or (T, 18)
            [left_9d, right_9d]

    Returns:
        action_14d: shape (14,) or (T, 14)
            [left_7d, right_7d]
    """
    squeeze = False
    if action_18d.ndim == 1:
        action_18d = action_18d[np.newaxis, :]
        squeeze = True

    left_9d  = action_18d[:, :9]
    right_9d = action_18d[:, 9:]

    left_7d  = decode_arm_9d_to_7d(left_9d)
    right_7d = decode_arm_9d_to_7d(right_9d)

    action_14d = np.concatenate([left_7d, right_7d], axis=1)

    if squeeze:
        action_14d = action_14d.squeeze(0)
    return action_14d


# ── Safety clamp ────────────────────────────────────────────────────────────

def clamp_action_14d(
    action_14d: np.ndarray,
    max_dx: float = 500.0,     # μm per step
    max_dy: float = 500.0,
    max_dz: float = 500.0,
    max_droll: float = 5000.0,  # mdeg per step (~5°)
    max_dpitch: float = 5000.0,
    max_dyaw: float = 200.0,    # mdeg per step (~0.2°, conservative for ±8° range)
    max_dgripper: float = 1000.0,
) -> np.ndarray:
    """Clamp per-step deltas to safe limits.

    Prevents the policy from commanding dangerous large movements.
    Each arm is clamped independently.
    """
    clamped = action_14d.copy()
    limits = np.array([
        max_dx, max_dy, max_dz,
        max_droll, max_dpitch,
        max_dyaw, max_dgripper,
    ] * 2)  # same limits for both arms

    clamped = np.clip(clamped, -limits, limits)
    return clamped


# ── HAL dispatch ────────────────────────────────────────────────────────────

class RobotAdapter:
    """Dispatches 18D policy output to the robot hardware.

    Usage:
        adapter = RobotAdapter(hal_client)
        # In inference loop:
        action_18d = model.predict(obs)  # from policy
        adapter.step(action_18d)

    Args:
        hal_client: Your HAL client instance. Must implement:
            - move_relative(side: str, axis: str, delta: float) -> None
            where side ∈ {"left", "right"},
                  axis ∈ {"X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"},
                  delta is in μm (linear) or mdeg (rotation) or native (gripper).
    """

    AXIS_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]

    def __init__(self, hal_client, safety_clamp: bool = True):
        self.hal_client = hal_client
        self.safety_clamp = safety_clamp

    def step(self, action_18d: np.ndarray) -> None:
        """Convert 18D action to 14D and send to robot.

        Args:
            action_18d: shape (18,) model output.
        """
        if isinstance(action_18d, torch.Tensor):
            action_18d = action_18d.cpu().numpy()

        # Decode sin-cos → angles
        action_14d = decode_action_18d_to_14d(action_18d).squeeze()

        # Safety clamp
        if self.safety_clamp:
            action_14d = clamp_action_14d(action_14d)

        # Split arms
        left_7d  = action_14d[:7]
        right_7d = action_14d[7:]

        # Dispatch to HAL
        self._send_arm("left", left_7d)
        self._send_arm("right", right_7d)

    def _send_arm(self, side: str, arm_7d: np.ndarray) -> None:
        """Send 7D delta command for one arm to HAL."""
        for i, axis_name in enumerate(self.AXIS_NAMES):
            delta = float(arm_7d[i])
            if abs(delta) > 1e-6:  # skip zero moves
                self.hal_client.move_relative(side, axis_name, delta)


# ── Standalone verification ─────────────────────────────────────────────────

def verify_roundtrip():
    """Verify 18D → 14D → check values."""
    np.random.seed(42)

    # Simulate a model output (18D)
    left_9d = np.array([
        50.0, -30.0, 20.0,            # Δx, Δy, Δz (μm)
        np.sin(np.deg2rad(3.6)),       # sin(Δroll)  where Δroll = 3.6°
        np.cos(np.deg2rad(3.6)),       # cos(Δroll)
        np.sin(np.deg2rad(-1.8)),      # sin(Δpitch) where Δpitch = -1.8°
        np.cos(np.deg2rad(-1.8)),      # cos(Δpitch)
        160.0,                          # Δyaw (mdeg)
        100.0,                          # Δgripper
    ])
    right_9d = np.array([
        -40.0, 25.0, -15.0,
        np.sin(np.deg2rad(-2.0)),
        np.cos(np.deg2rad(-2.0)),
        np.sin(np.deg2rad(1.0)),
        np.cos(np.deg2rad(1.0)),
        -80.0,
        -50.0,
    ])
    action_18d = np.concatenate([left_9d, right_9d])

    action_14d = decode_action_18d_to_14d(action_18d)
    clamped = clamp_action_14d(action_14d)

    print("Decode verification (18D → 14D):")
    print(f"  Left:  Δroll={action_14d[3]:.1f} mdeg (expected 3600), Δpitch={action_14d[4]:.1f} (expected -1800)")
    print(f"  Right: Δroll={action_14d[10]:.1f} mdeg (expected -2000), Δpitch={action_14d[11]:.1f} (expected 1000)")
    print(f"  Clamped left yaw: {clamped[5]:.1f} (limit ±200 mdeg)")

    # Verify sin²+cos² = 1 after decode
    err_r = abs(action_14d[3] - 3600.0)
    err_p = abs(action_14d[4] - (-1800.0))
    print(f"  Decode error: roll={err_r:.4f} mdeg, pitch={err_p:.4f} mdeg")
    assert err_r < 0.01 and err_p < 0.01, "Decode error too large!"
    print("  All checks passed!")


if __name__ == "__main__":
    verify_roundtrip()
