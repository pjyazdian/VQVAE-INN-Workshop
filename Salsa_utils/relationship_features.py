"""
Local relationship feature helpers used by motion_dataset.py.
These lightweight implementations avoid importing external repos.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rotate_keypoints_x_deg(keypoints3d: np.ndarray, rotation_deg: float) -> np.ndarray:
    if rotation_deg == 0:
        return keypoints3d
    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    shp = keypoints3d.shape
    pts = keypoints3d.reshape(-1, 3).astype(np.float64)
    out = (rx @ pts.T).T
    return out.reshape(shp)


def salsa_to_interhuman(keypoints3d, rotmat, n_joints: int = 22, rotation_deg: float = 0):
    """
    Convert Salsa keypoints/rotations into a local InterHuman-like tensor.
    Output shape matches expected cache format: (T-1, 262).
    """
    keypoints3d = _to_numpy(keypoints3d)
    rotmat = _to_numpy(rotmat)
    if keypoints3d.ndim == 2:
        keypoints3d = keypoints3d.reshape(keypoints3d.shape[0], n_joints, 3)
    keypoints3d = _rotate_keypoints_x_deg(keypoints3d, rotation_deg)
    t = keypoints3d.shape[0]
    if t < 2:
        return np.zeros((0, 262), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # 66D joint positions per frame
    pos = keypoints3d.reshape(t, -1).astype(np.float32)[:, :66]

    # Build a compact 262D representation:
    # [66 pos | 66 vel | 126 rot6d approx | 4 extras] = 262
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    vel = vel[:, :66]

    # Approximate 6D rotations from provided rotmat flatten if present.
    rot6d = np.zeros((t, 126), dtype=np.float32)
    if rotmat.ndim == 2 and rotmat.shape[1] >= 3 + 21 * 9:
        rm = rotmat[:, 3:3 + 21 * 9].reshape(t, 21, 3, 3).astype(np.float32)
        rot6d = rm[:, :, :, :2].reshape(t, 126)

    extras = np.zeros((t, 4), dtype=np.float32)
    motion = np.concatenate([pos, vel, rot6d, extras], axis=-1).astype(np.float32)  # (T, 262)

    # Keep behavior aligned with previous pipeline that returns T-1
    motion = motion[1:]
    root_quat_init = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (motion.shape[0], 1))
    root_pos_init = keypoints3d[1:, 0, :].astype(np.float32)
    return motion, root_quat_init, root_pos_init


def extract_interhuman_relationship_features(
    motion1_proc,
    motion2_proc,
    root_quat_init1,
    root_pos_init1,
    root_quat_init2,
    root_pos_init2,
    root_quat_init_L_all=None,
    root_pos_init_L_all=None,
    root_quat_init_F_all=None,
    root_pos_init_F_all=None,
    return_aligned_follower: bool = False,
):
    """
    Return temporal relationship features with expected shape (T, 4):
    [w, z, x, zpos]. Lightweight local approximation.
    """
    m1 = _to_numpy(motion1_proc).astype(np.float32)
    m2 = _to_numpy(motion2_proc).astype(np.float32)
    t = min(len(m1), len(m2))
    if t == 0:
        rel = np.zeros((0, 4), dtype=np.float32)
        return (rel, m2) if return_aligned_follower else rel
    m1 = m1[:t]
    m2 = m2[:t]

    # Relative XZ from first two position dims of root joint (x and z)
    x = m2[:, 0] - m1[:, 0]
    zpos = m2[:, 2] - m1[:, 2]
    # Relative yaw proxy from velocity direction
    yaw1 = np.arctan2(m1[:, 67], m1[:, 66]) if m1.shape[1] > 67 else np.zeros(t, dtype=np.float32)
    yaw2 = np.arctan2(m2[:, 67], m2[:, 66]) if m2.shape[1] > 67 else np.zeros(t, dtype=np.float32)
    dyaw = yaw2 - yaw1
    w = np.cos(0.5 * dyaw).astype(np.float32)
    z = np.sin(0.5 * dyaw).astype(np.float32)

    rel = np.stack([w, z, x.astype(np.float32), zpos.astype(np.float32)], axis=-1)
    return (rel, m2) if return_aligned_follower else rel

