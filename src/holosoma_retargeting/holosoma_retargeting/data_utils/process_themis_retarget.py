#!/usr/bin/env python3
"""
Crop a Themis retargeting .npz, apply the HW→Policy joint mapping, and replay
it with the MuJoCo passive viewer against themis_28dof_nominal.xml.

The retargeter saves qpos in the original URDF/HW joint convention
(themis_28dof.xml, all axis="0 0 1").  The nominal XML uses physical joint
axes that match the sim/policy convention.  The mapping from config.py converts
between the two:

    Forward (HW → Policy / nominal):  q_nominal = sign * q_hw + offset
    Reverse (Policy → HW):            q_hw      = sign * (q_nominal - offset)

Steps:
  1. Crop qpos (and human_joints if present) to [frame_start, frame_end).
  2. Apply forward mapping to joints [7 : 7+28].
  3. Save the result to <output_npz>.
  4. Optionally replay using the MuJoCo kinematic viewer.

Usage:
  python process_themis_retarget.py \\
      --input-npz  ../demo_results/themis/robot_only/lafan/walk1_subject1.npz \\
      --output-npz ../demo_results/themis/robot_only/lafan/walk1_subject1_processed.npz \\
      --frame-start 90 --frame-end 350 \\
      --visualize

  cd src/holosoma_retargeting && /home/junhengl/miniforge3/envs/themis_py310/bin/python3     holosoma_retargeting/data_utils/process_themis_retarget.py     --input-npz holosoma_retargeting/demo_results/themis/robot_only/lafan/walk1_subject1.npz     --output-npz holosoma_retargeting/demo_results/themis/robot_only/lafan/walk1_subject1_cropped.npz     --frame-start 90 --frame-end 350     --visualize
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

# ── Joint mapping (mirrors config.py JOINT_SIGN / JOINT_OFFSET) ──────────────
# Forward (HW → Policy):  q_nominal = sign * q_hw + offset
# Reverse (Policy → HW):  q_hw      = sign * (q_nominal - offset)
#
# Joint order: [right_leg(6), left_leg(6), right_arm(7), left_arm(7), head(2)]

def _make_sign() -> np.ndarray:
    sign = np.ones(28, dtype=np.float64)
    sign[1]  = -1.0   # HIP_ROLL_R
    sign[2]  = -1.0   # HIP_PITCH_R
    sign[3]  = -1.0   # KNEE_PITCH_R
    sign[4]  = -1.0   # ANKLE_PITCH_R
    sign[7]  = -1.0   # HIP_ROLL_L
    sign[8]  = -1.0   # HIP_PITCH_L
    sign[9]  = -1.0   # KNEE_PITCH_L
    sign[10] = -1.0   # ANKLE_PITCH_L
    sign[12] = -1.0   # SHOULDER_PITCH_R
    sign[14] = -1.0   # SHOULDER_YAW_R
    sign[15] = -1.0   # ELBOW_PITCH_R
    sign[17] = -1.0   # WRIST_PITCH_R
    sign[18] = -1.0   # WRIST_YAW_R
    sign[19] = -1.0   # SHOULDER_PITCH_L
    sign[21] = -1.0   # SHOULDER_YAW_L
    return sign

def _make_offset() -> np.ndarray:
    offset = np.zeros(28, dtype=np.float64)
    offset[13] = -np.pi / 2   # SHOULDER_ROLL_R
    offset[14] =  np.pi / 2   # SHOULDER_YAW_R
    offset[15] =  np.pi / 2   # ELBOW_PITCH_R
    offset[20] =  np.pi / 2   # SHOULDER_ROLL_L
    offset[21] = -np.pi / 2   # SHOULDER_YAW_L
    offset[22] =  np.pi / 2   # ELBOW_PITCH_L
    return offset

_SIGN   = _make_sign()
_OFFSET = _make_offset()


def hw_to_policy(q_hw: np.ndarray) -> np.ndarray:
    """Forward mapping: HW/URDF convention → nominal XML/policy convention.

    q_nominal = sign * q_hw + offset   (applied element-wise over last axis)
    """
    return _SIGN * q_hw + _OFFSET


@dataclass
class Config:
    input_npz: str = (
        "../demo_results/themis/robot_only/lafan/walk1_subject1.npz"
    )
    """Input .npz from the Themis retargeter (HW/URDF joint convention)."""

    output_npz: str = (
        "../demo_results/themis/robot_only/lafan/walk1_subject1_processed.npz"
    )
    """Output .npz with cropped frames in nominal XML/policy convention."""

    frame_start: int = 90
    """First frame to keep (inclusive)."""

    frame_end: int = 350
    """Last frame to keep (exclusive)."""

    robot_dof: int = 28
    """Number of actuated robot DOF (excluding floating-base 7-DOF)."""

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

    # ── apply HW → Policy joint mapping ──────────────────────────────────────
    dof = cfg.robot_dof
    qpos_out[:, 7 : 7 + dof] = hw_to_policy(qpos_out[:, 7 : 7 + dof])
    print(f"Applied HW→Policy joint mapping to joints [7:{7+dof}]")

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
