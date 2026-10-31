"""Geometry variation: per-link length scales and finger-mount shifts, and the deformations they imply.

Both deformations are smooth, orientation-preserving maps (Jacobian determinant > 0), so meshes never
fold. Points map exactly; normals map by the inverse transpose of the Jacobian; surface area scales by
det(J) * |J^-T n|.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from . import config
from .leap import BaseModel, link_type
from .topology import Topology


@dataclass(frozen=True)
class Geometry:
    scales: dict = field(default_factory=dict)  # long link name -> length scale s
    shifts: dict = field(default_factory=dict)  # finger -> mount shift (m) along config.MOUNT_AXIS
    draw: int = 0

    def scale(self, link: str) -> float:
        return float(self.scales.get(link, 1.0))

    def shift(self, finger: str) -> float:
        return float(self.shifts.get(finger, 0.0))

    def to_dict(self) -> dict:
        return {
            "draw": self.draw,
            "length_scales": {k: round(v, 6) for k, v in sorted(self.scales.items())},
            "mount_shifts_m": {k: round(v, 6) for k, v in sorted(self.shifts.items())},
        }


def long_links_of(topology: Topology) -> list[str]:
    links = []
    for f in topology.fingers:
        links += [l for l in config.FINGER_LINKS[f] if link_type(l) in config.LONG_LINKS]
    return links


def sample_geometry(topology: Topology, draw: int, seed: int = 0, attempt: int = 0) -> Geometry:
    """Draw 0 is the nominal LEAP geometry; draws >= 1 are random and deterministic in
    (seed, topology, draw, attempt). `attempt` > 0 redraws a geometry that failed the validity gate."""
    if draw == 0:
        return Geometry(draw=0)
    rng = np.random.default_rng([seed, zlib.crc32(topology.id.encode()), draw, attempt])
    lo, hi = config.LENGTH_SCALE_RANGE
    scales = {l: float(rng.uniform(lo, hi)) for l in long_links_of(topology)}
    shifts = {
        f: float(rng.uniform(-config.MOUNT_SHIFT_MAX, config.MOUNT_SHIFT_MAX))
        for f in config.MOUNT_FINGERS
        if f in topology.fingers
    }
    return Geometry(scales=scales, shifts=shifts, draw=draw)


# ---------------------------------------------------------------- deformations


class Identity:
    def map_points(self, P):
        return np.asarray(P, dtype=float)

    def jacobian(self, P):
        return np.broadcast_to(np.eye(3), (len(P), 3, 3))

    def map_normals(self, P, N):
        return np.asarray(N, dtype=float)

    def area_factor(self, P, N):
        return np.ones(len(P))


class _Deformation(Identity):
    def map_normals(self, P, N):
        Jinv_T = np.linalg.inv(self.jacobian(P)).transpose(0, 2, 1)
        out = np.einsum("nij,nj->ni", Jinv_T, N)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def area_factor(self, P, N):
        J = self.jacobian(P)
        Jinv_T = np.linalg.inv(J).transpose(0, 2, 1)
        return np.linalg.det(J) * np.linalg.norm(np.einsum("nij,nj->ni", Jinv_T, N), axis=1)


class AxisStretch(_Deformation):
    """Lengthen (or shorten) a link by `delta` along its distal axis (link-frame y, signed).

    With u = sign * y: points with u < u0 stay, points with u > u1 move by delta, and the band
    [u0, u1] is stretched linearly in between.
    """

    def __init__(self, sign: int, length: float, delta: float, band=config.STRETCH_BAND):
        self.sign, self.length, self.delta = sign, length, delta
        self.u0, self.u1 = band[0] * length, band[1] * length
        width = self.u1 - self.u0
        self.k = (width + delta) / width
        if self.k <= 0:
            raise ValueError("stretch band too narrow for this delta")

    def _u_new(self, u):
        return np.where(u < self.u0, u, np.where(u > self.u1, u + self.delta, self.u0 + (u - self.u0) * self.k))

    def map_points(self, P):
        P = np.array(P, dtype=float)
        P[:, 1] = self.sign * self._u_new(self.sign * P[:, 1])
        return P

    def jacobian(self, P):
        u = self.sign * np.asarray(P)[:, 1]
        J = np.repeat(np.eye(3)[None], len(u), axis=0)
        J[:, 1, 1] = np.where((u >= self.u0) & (u <= self.u1), self.k, 1.0)
        return J

    def map_interval(self, lo, hi):
        """Image of the link-frame y-interval [lo, hi] (the map is monotone)."""
        a, b = self.map_points(np.array([[0.0, lo, 0.0], [0.0, hi, 0.0]]))[:, 1]
        return min(a, b), max(a, b)


class BoxShift(_Deformation):
    """Translate everything inside an axis-aligned box by `delta * axis`, fading linearly to zero over
    `falloff` outside it (Euclidean distance to the box)."""

    def __init__(self, center, half, axis, delta, falloff):
        self.c, self.h = np.asarray(center, float), np.asarray(half, float)
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.delta, self.falloff = float(delta), float(falloff)

    def _excess(self, P):
        p = np.asarray(P, float) - self.c
        e = np.maximum(np.abs(p) - self.h, 0.0)
        return p, e, np.linalg.norm(e, axis=1)

    def weight(self, P):
        _, _, d = self._excess(P)
        return np.clip(1.0 - d / self.falloff, 0.0, 1.0)

    def map_points(self, P):
        return np.asarray(P, float) + self.delta * self.weight(P)[:, None] * self.a

    def jacobian(self, P):
        p, e, d = self._excess(P)
        grad = np.zeros_like(p)
        band = (d > 0) & (d < self.falloff)
        grad[band] = -(np.sign(p[band]) * e[band] / d[band, None]) / self.falloff
        return np.eye(3)[None] + self.delta * self.a[None, :, None] * grad[:, None, :]


class Compose(_Deformation):
    """Sum of displacements with disjoint supports (e.g. several BoxShifts on the palm)."""

    def __init__(self, parts):
        self.parts = parts

    def map_points(self, P):
        P = np.asarray(P, float)
        return P + sum(part.map_points(P) - P for part in self.parts)

    def jacobian(self, P):
        J = np.repeat(np.eye(3)[None], len(P), axis=0)
        for part in self.parts:
            J = J + (part.jacobian(P) - np.eye(3)[None])
        return J


def link_deformation(base: BaseModel, link: str, geometry: Geometry):
    """The deformation applied to a link's geometry (in its own frame)."""
    t = link_type(link)
    if t in config.LONG_LINKS and geometry.scale(link) != 1.0:
        L = base.long_link_lengths[link]
        return AxisStretch(config.LONG_LINKS[t], L, (geometry.scale(link) - 1.0) * L)
    if link == config.PALM and geometry.shifts:
        boxes = {g.name: g for g in base.collision[config.PALM]}
        parts = []
        for f, delta in geometry.shifts.items():
            if delta == 0.0:
                continue
            box = boxes[config.MOUNT_BOX[f]]
            assert np.allclose(box.rot, np.eye(3)), "mount boxes must be axis-aligned"
            parts.append(
                BoxShift(box.pos, box.half_size + config.MOUNT_CORE_GROW, config.MOUNT_AXIS, delta, config.MOUNT_FALLOFF)
            )
        return Compose(parts) if parts else Identity()
    return Identity()
