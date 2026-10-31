"""Rigid-transform helpers (URDF conventions: rpy = fixed-axis XYZ, i.e. R = Rz(y) Ry(p) Rx(r))."""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def rpy_to_matrix(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def matrix_to_rpy(R) -> np.ndarray:
    # Many LEAP joint frames sit exactly at pitch = +-90 deg (gimbal lock). scipy then fixes one angle to
    # zero; the returned triple still reproduces R exactly (checked to 5e-16), so the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Rotation.from_matrix(np.asarray(R, dtype=float)).as_euler("xyz")


def matrix_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([w, x, y, z])


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def make_transform(xyz=(0.0, 0.0, 0.0), R=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def axis_rotation(axis, angle) -> np.ndarray:
    """Batched rotation about a unit axis: angle (...,) -> (..., 3, 3) (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s = np.sin(angle)[..., None, None]
    c = np.cos(angle)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def transform_points(T, P) -> np.ndarray:
    """Apply (..., 4, 4) transforms to (..., N, 3) points."""
    return P @ np.swapaxes(T[..., :3, :3], -1, -2) + T[..., None, :3, 3]
