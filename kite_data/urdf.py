"""URDF export. Kinematics (joint origins, axes, limits, locked joints as fixed) are exact.

Visuals reference the base LEAP meshes. A stretched link's visual is scaled uniformly along its long
axis (same end points as the band stretch used for contacts and collisions); palm mount shifts are not
reflected in the palm visual. Collision geometry lives in the MJCF (hand.xml), not here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config
from .hand import Hand
from .leap import link_type
from .transforms import matrix_to_rpy


def _fmt(v) -> str:
    return " ".join(f"{x:.9g}" for x in np.ravel(v))


def _visual(hand: Hand, link: str, mesh_dir: str) -> tuple[np.ndarray, np.ndarray, str]:
    bl = hand.base.links[link]
    origin = bl.visual_origin.copy()
    scale = np.ones(3)
    t = link_type(link)
    s = hand.geometry.scale(link)
    if t in config.LONG_LINKS and s != 1.0:
        k = int(np.argmax(np.abs(origin[1, :3])))  # mesh axis that maps onto the link's y axis
        if abs(origin[1, k]) > 0.999:
            scale[k] = s
            origin[1, 3] *= s
    return origin, scale, f"{mesh_dir}/{bl.mesh_file.name}"


def to_urdf(hand: Hand, out_dir: Path) -> str:
    mesh_dir = os.path.relpath(config.LEAP_DIR, out_dir)
    lines = [f'<?xml version="1.0"?>', f'<robot name="leap_{hand.id}">']
    for link in hand.links:
        bl = hand.base.links[link]
        origin, scale, mesh = _visual(hand, link, mesh_dir)
        I = bl.inertia
        lines += [
            f'  <link name="{link}">',
            "    <visual>",
            f'      <origin xyz="{_fmt(origin[:3, 3])}" rpy="{_fmt(matrix_to_rpy(origin[:3, :3]))}"/>',
            f'      <geometry><mesh filename="{mesh}" scale="{_fmt(scale)}"/></geometry>',
            "    </visual>",
            "    <inertial>",
            f'      <origin xyz="{_fmt(bl.inertial_origin[:3, 3])}" rpy="{_fmt(matrix_to_rpy(bl.inertial_origin[:3, :3]))}"/>',
            f'      <mass value="{bl.mass:.9g}"/>',
            f'      <inertia ixx="{I[0]:.9g}" ixy="{I[1]:.9g}" ixz="{I[2]:.9g}" iyy="{I[3]:.9g}" iyz="{I[4]:.9g}" izz="{I[5]:.9g}"/>',
            "    </inertial>",
            "  </link>",
        ]
    for j in hand.joints:
        kind = "fixed" if j.locked else "revolute"
        lines += [
            f'  <joint name="{j.name}" type="{kind}">',
            f'    <parent link="{j.parent}"/>',
            f'    <child link="{j.child}"/>',
            f'    <origin xyz="{_fmt(j.origin[:3, 3])}" rpy="{_fmt(matrix_to_rpy(j.origin[:3, :3]))}"/>',
        ]
        if not j.locked:
            lines += [
                f'    <axis xyz="{_fmt(j.axis)}"/>',
                f'    <limit lower="{j.lower:.9g}" upper="{j.upper:.9g}" effort="{j.effort:.9g}" velocity="{j.velocity:.9g}"/>',
            ]
        lines.append("  </joint>")
    lines += ["</robot>", ""]
    return "\n".join(lines)
