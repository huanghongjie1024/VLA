"""Observe geometry and optionally record views during an unmodified policy rollout."""
from pathlib import Path
import os

import cv2
import numpy as np


FIELDS = ("tick", "center_error_mm", "benchmark_depth_mm", "physical_tip_depth_mm",
          "peg_tilt_deg", "force_N", "weld", "success", "peg_hole_contact")


class ReplayTrace:
    def __init__(self, seed, mode):
        self.seed, self.mode = seed, mode
        self.output = Path(os.environ["REPLAY_TRACE_DIR"])
        self.rows = []
        self.video = None
        video_seed = int(os.environ.get("REPLAY_VIDEO_SEED", "-1"))
        if seed == video_seed:
            filename = str(self.output / f"{mode}_{seed}.avi")
            self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MJPG"), 40, (512, 340))
            if not self.video.isOpened():
                raise RuntimeError(f"Could not open {filename}")

    def observe(self, sim, wrench, tick, welded):
        env, data = sim.env, sim.d
        peg = data.xpos[env.peg_body_id]
        top = env.hole_top()
        radial = float(np.linalg.norm(peg[:2] - top[:2]) * 1000)
        depth = float((top[2] - peg[2] + env.peg_half_length) * 1000)
        rotation = data.xmat[env.peg_body_id].reshape(3, 3)
        vertical = abs(float(rotation[2, 2]))
        true_depth = float((env.hole_top_true()[2] - peg[2] + env.peg_half_length * vertical) * 1000)
        tilt = float(np.degrees(np.arccos(np.clip(vertical, 0, 1))))
        success = bool(sim.success())
        geom_body = sim.m.geom_bodyid
        contact = any(
            {int(geom_body[c.geom1]), int(geom_body[c.geom2])} == {env.peg_body_id, env.hole_body_id}
            for c in data.contact[:data.ncon]
        )
        self.rows.append((tick, radial, depth, true_depth, tilt, wrench.fmag, welded, success, contact))
        if self.video is not None and (tick % 2 == 0 or tick == 3465):
            views = [cv2.resize(np.ascontiguousarray(sim.obs[key][::-1, :, ::-1]), (256, 256),
                                interpolation=cv2.INTER_NEAREST)
                     for key in ("agentview_image", "robot0_eye_in_hand_image")]
            frame = np.full((340, 512, 3), (28, 26, 22), np.uint8)
            frame[34:290] = np.concatenate(views, axis=1)
            color = (150, 255, 150) if success else (232, 232, 232)
            cv2.putText(frame, f"{self.mode.upper()} | Seed {self.seed} | {tick/20:.1f}s sim",
                        (9, 23), cv2.FONT_HERSHEY_SIMPLEX, .59, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Offset {radial:.1f} mm   Depth {depth:.1f} mm   Force {wrench.fmag:.1f} N",
                        (9, 310), cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1, cv2.LINE_AA)
            status = f"{'PASS' if success else '---'} | {'welded' if welded else 'released'} | {'peg-block contact' if contact else 'no peg-block contact'}"
            cv2.putText(frame, status, (9, 330), cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1, cv2.LINE_AA)
            self.video.write(frame)

    def close(self):
        if self.video is not None:
            self.video.release()
        if self.rows:
            np.savez_compressed(self.output / f"seed_{self.seed}.npz",
                                **dict(zip(FIELDS, np.asarray(self.rows).T, strict=True)))
