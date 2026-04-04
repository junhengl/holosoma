#!/usr/bin/env python3
"""
Crop a Themis retargeting .npz and replay it with the MuJoCo passive viewer.

The retargeter now uses themis_28dof_nominal.urdf and themis_28dof_nominal.xml,
which share the same joint convention, so no remapping is needed here.

Steps:
  1. Crop qpos (and human_joints if present) to [frame_start, frame_end).
  2. Save the result to <output_npz>.
  3. Optionally replay using the MuJoCo kinematic viewer.

Usage:
  python process_themis_retarget.py \\
      --input-npz  ../demo_results/themis/robot_only/lafan/walk1_subject1.npz \\
      --output-npz ../demo_results/themis/robot_only/lafan/walk1_subject1_processed.npz \\
      --frame-start 90 --frame-end 350 \\
      --visualize
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro


_DEFAULT_NOMINAL_XML = (
    Path(__file__).resolve().parent.parent
    / "models/themis/themis_28dof_nominal.xml"
)


@dataclass
class Config:
    input_npz: str = (
        "../demo_results/themis/robot_only/lafan/walk1_subject1.npz"
    )
    """Input .npz from the Themis retargeter."""

    output_npz: str = (
        "../demo_results/themis/robot_only/lafan/walk1_subject1_processed.npz"
    )
    """Output .npz with cropped frames."""

    frame_start: int = 90
    """First frame to keep (inclusive)."""

    frame_end: int = 350
    """Last frame to keep (exclusive)."""

    visualize: bool = False
    """Replay the processed trajectory in the MuJoCo passive viewer."""

    nominal_xml: str = str(_DEFAULT_NOMINAL_XML)
    """Path to themis_28dof_nominal.xml used for visualization."""


def main(cfg: Config) -> None:
    input_path  = Path(cfg.input_npz)
    output_path = Path(cfg.output_npz)

    # ── load ─────────────────────────────────────────────────────────────────
    data = np.load(input_path, allow_pickle=True)
    qpos: np.ndarray = data["qpos"]   # (T, 7+DOF[+7])
    fps: int = int(data["fps"]) if "fps" in data else 30

    T     = qpos.shape[0]
    start = max(0, cfg.frame_start)
    end   = min(T, cfg.frame_end)
    print(f"Total frames: {T}  →  cropping [{start}, {end})  ({end - start} frames)")

    # ── crop ─────────────────────────────────────────────────────────────────
    qpos_out = qpos[start:end].copy()

    # ── save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {"qpos": qpos_out, "fps": fps}
    if "human_joints" in data:
        save_kwargs["human_joints"] = data["human_joints"][start:end]
    if "cost" in data:
        save_kwargs["cost"] = data["cost"]

    np.savez(output_path, **save_kwargs)
    print(f"Saved processed motion to: {output_path}")
    print(f"  qpos shape : {qpos_out.shape}")
    print(f"  fps        : {fps}")

    # ── visualize ─────────────────────────────────────────────────────────────
    if cfg.visualize:
        print(f"\nLoading MuJoCo model: {cfg.nominal_xml}")
        m = mujoco.MjModel.from_xml_path(cfg.nominal_xml)
        d = mujoco.MjData(m)

        dt       = 1.0 / fps
        n_frames = qpos_out.shape[0]
        n_qpos   = qpos_out.shape[1]
        print(f"Replaying {n_frames} frames at {fps} fps — close the viewer window to exit.")

        with mujoco.viewer.launch_passive(m, d) as viewer:
            i = 0
            while viewer.is_running():
                t0 = time.perf_counter()
                d.qpos[:n_qpos] = qpos_out[i]
                mujoco.mj_forward(m, d)
                viewer.sync()
                i = (i + 1) % n_frames
                elapsed = time.perf_counter() - t0
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
