"""Self-collision for hand instances.

Fast checks use Menagerie's hand-tuned boxes (and fingertip pad meshes) in MuJoCo: they agree with
exact mesh checks on ~95% of configurations at ~0.3 ms per check. Exact mesh checks (FCL) are only used
for the validity gate at the zero pose.
"""

from __future__ import annotations

import dataclasses
import functools
from xml.sax.saxutils import quoteattr

import numpy as np

from . import config
from .geometry import AxisStretch
from .hand import Hand
from .leap import CollisionGeom, link_mesh
from .transforms import matrix_to_quat_wxyz


def _fmt(v) -> str:
    return " ".join(f"{x:.9g}" for x in np.ravel(v))


def hand_collision_geoms(hand: Hand) -> dict[str, list[CollisionGeom]]:
    out = {}
    for link in hand.links:
        D = hand.deformation[link]
        geoms = []
        for g in hand.base.collision.get(link, []):
            if g.kind == "mesh":
                geoms.append(dataclasses.replace(g, vertices=D.map_points(g.vertices)))
            elif isinstance(D, AxisStretch):
                if not np.allclose(g.rot, np.eye(3)):
                    raise ValueError(f"{g.name}: stretched boxes must be axis-aligned")
                a, b = D.map_interval(g.pos[1] - g.half_size[1], g.pos[1] + g.half_size[1])
                pos, half = g.pos.copy(), g.half_size.copy()
                pos[1], half[1] = (a + b) / 2, (b - a) / 2
                geoms.append(dataclasses.replace(g, pos=pos, half_size=half))
            else:  # palm mount shifts move whole boxes; identity elsewhere
                geoms.append(dataclasses.replace(g, pos=D.map_points(g.pos[None])[0]))
        out[link] = geoms
    return out


def excluded_pairs(hand: Hand) -> list[tuple[str, str]]:
    present = set(hand.links)
    return [(a, b) for a, b in hand.base.exclude if a in present and b in present]


def to_mjcf(hand: Hand) -> str:
    """Collision-only MJCF (boxes + fingertip pads) with the hand's exact kinematics; palm fixed to world."""
    geoms = hand_collision_geoms(hand)
    assets, children = [], {l: [] for l in hand.links}
    for j in hand.joints:
        children[j.parent].append(j)

    def body_xml(link: str, indent: str) -> list[str]:
        lines = []
        bl = hand.base.links[link]
        ixx, ixy, ixz, iyy, iyz, izz = bl.inertia
        R = bl.inertial_origin[:3, :3]
        I = R @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ R.T  # into the body frame
        lines.append(f'{indent}<inertial pos="{_fmt(bl.inertial_origin[:3, 3])}" mass="{bl.mass:.9g}" '
                     f'fullinertia="{_fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])}"/>')
        for i, g in enumerate(geoms[link]):
            name = quoteattr(f"{link}_c{i}")
            if g.kind == "box":
                lines.append(f'{indent}<geom name={name} type="box" size="{_fmt(g.half_size)}" pos="{_fmt(g.pos)}" '
                             f'quat="{_fmt(matrix_to_quat_wxyz(g.rot))}"/>')
            else:
                assets.append(f'    <mesh name="{link}_pad" vertex="{_fmt(g.vertices)}"/>')
                lines.append(f'{indent}<geom name={name} type="mesh" mesh="{link}_pad"/>')
        for j in children[link]:
            lines.append(f'{indent}<body name="{j.child}" pos="{_fmt(j.origin[:3, 3])}" '
                         f'quat="{_fmt(matrix_to_quat_wxyz(j.origin[:3, :3]))}">')
            if not j.locked:
                lines.append(f'{indent}  <joint name="{j.name}" axis="{_fmt(j.axis)}" range="{j.lower:.9g} {j.upper:.9g}"/>')
            lines += body_xml(j.child, indent + "  ")
            lines.append(f"{indent}</body>")
        return lines

    body = body_xml(config.PALM, "      ")
    excl = [f'    <exclude body1="{a}" body2="{b}"/>' for a, b in excluded_pairs(hand)]
    return "\n".join([
        f'<mujoco model="leap_{hand.id}">',
        '  <compiler angle="radian" autolimits="true" balanceinertia="true"/>',
        '  <default>',
        '    <geom contype="1" conaffinity="1" rgba="0.6 0.6 0.6 1"/>',
        '  </default>',
        '  <asset>', *assets, '  </asset>',
        '  <worldbody>',
        f'    <body name="{config.PALM}">', *body, '    </body>',
        '  </worldbody>',
        '  <contact>', *excl, '  </contact>',
        '</mujoco>',
        "",
    ])


class Collider:
    """Fast self-collision checks on the box model."""

    def __init__(self, hand: Hand, xml: str | None = None):
        import mujoco

        self._mj = mujoco
        self.hand = hand
        self.model = mujoco.MjModel.from_xml_string(xml or to_mjcf(hand))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j.name)]
                              for j in hand.dof_joints], dtype=int)
        self.geom_link = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.model.geom_bodyid]

    def colliding_pairs(self, q) -> set[tuple[str, str]]:
        mj, m, d = self._mj, self.model, self.data
        d.qpos[:] = 0.0
        d.qpos[self.qadr] = q
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        out = set()
        for c in d.contact[: d.ncon]:
            if c.dist < -config.PENETRATION_TOL:
                out.add(tuple(sorted((self.geom_link[c.geom1], self.geom_link[c.geom2]))))
        return out

    def collides(self, q) -> bool:
        return bool(self.colliding_pairs(q))

    def free_mask(self, Q) -> np.ndarray:
        return np.array([not self.collides(q) for q in np.atleast_2d(Q)])


@functools.lru_cache(maxsize=None)
def _simplified_mesh(link: str, faces: int):
    return link_mesh(link).simplify_quadric_decimation(face_count=faces)


def deformed_link_mesh(hand: Hand, link: str, faces: int | None = None):
    """The link's visual mesh with the hand's geometry applied; optionally simplified first."""
    mesh = (link_mesh(link) if faces is None else _simplified_mesh(link, faces)).copy()
    mesh.vertices = hand.deformation[link].map_points(mesh.vertices)
    return mesh


def exact_colliding_pairs(hand: Hand, q, pairs=None, faces: int | None = None) -> set[tuple[str, str]]:
    """Mesh-mesh intersections (FCL) between non-adjacent links at configuration q, optionally only
    among `pairs` and on meshes simplified to `faces` triangles."""
    import trimesh

    wanted = None if pairs is None else {tuple(sorted(p)) for p in pairs}
    links = hand.links if wanted is None else sorted({l for p in wanted for l in p})
    T = hand.fk(q)
    cm = trimesh.collision.CollisionManager()
    for link in links:
        cm.add_object(link, deformed_link_mesh(hand, link, faces), transform=T[link])
    _, names = cm.in_collision_internal(return_names=True)
    adjacent = {tuple(sorted((j.parent, j.child))) for j in hand.joints}
    found = {tuple(sorted(p)) for p in names} - adjacent
    return found if wanted is None else found & wanted


def validity_gate(hand: Hand, n_samples: int = config.GATE_SAMPLES, seed: int = 0, collider: Collider | None = None) -> dict:
    """Reject geometry draws that collide at the zero pose or leave too little collision-free space.

    The box model covers every pair except the ones Menagerie excludes (palm vs. finger bases,
    neighbouring finger bases); those are checked on simplified exact meshes, which agreed with
    full-resolution all-pairs checks on 28/28 hands including extreme geometries, ~16x faster.
    """
    collider = collider or Collider(hand)
    q0 = np.zeros(hand.n_dof)
    box_q0 = sorted(collider.colliding_pairs(q0))
    exact_q0 = sorted(exact_colliding_pairs(hand, q0, pairs=excluded_pairs(hand), faces=config.GATE_MESH_FACES))
    rng = np.random.default_rng(seed)
    Q = rng.uniform(hand.lower, hand.upper, size=(n_samples, hand.n_dof))
    free_rate = float(collider.free_mask(Q).mean())
    ok = not box_q0 and not exact_q0 and free_rate >= config.GATE_MIN_FREE_RATE
    return {"ok": ok, "box_pairs_at_zero": [list(p) for p in box_q0], "exact_pairs_at_zero": [list(p) for p in exact_q0],
            "free_rate": round(free_rate, 4), "n_samples": n_samples}
