"""A hand instance = topology + geometry applied to the base LEAP model: kinematics, FK and its descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import __version__, config
from .geometry import Geometry, link_deformation
from .leap import BaseModel, link_type, load_base_model
from .topology import Topology
from .transforms import axis_rotation, make_transform, matrix_to_quat_wxyz, matrix_to_rpy


@dataclass
class HandJoint:
    name: str  # LEAP motor id from the base URDF
    finger: str
    k: int  # position in the finger, 0 = proximal
    parent: str  # parent link
    child: str  # child link
    origin: np.ndarray  # 4x4 parent-link frame -> joint frame (lock angle folded in when locked)
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float
    locked: bool
    dof: int = -1  # index among movable joints, -1 if locked
    parent_dof: int = -1  # nearest movable ancestor, -1 = wrist


class Hand:
    def __init__(self, topology: Topology, geometry: Geometry, base: BaseModel | None = None):
        self.base = base or load_base_model()
        self.topology, self.geometry = topology, geometry
        self.id = f"{topology.id}_g{geometry.draw}"
        axis = np.asarray(config.MOUNT_AXIS, float)
        self.links = [config.PALM]
        self.joints: list[HandJoint] = []
        for f in topology.fingers:
            last_dof = -1
            for k, (jn, ln) in enumerate(zip(config.FINGER_JOINTS[f], config.FINGER_LINKS[f])):
                bj = self.base.joints[jn]
                assert bj.child == ln
                origin = bj.origin.copy()
                if k == 0:
                    origin[:3, 3] += geometry.shift(f) * axis
                ptype = link_type(bj.parent)
                if ptype in config.LONG_LINKS:  # parent link lengthened -> child joint moves along its axis
                    L = self.base.long_link_lengths[bj.parent]
                    origin[1, 3] += config.LONG_LINKS[ptype] * (geometry.scale(bj.parent) - 1.0) * L
                locked = topology.is_locked(f, k)
                if locked:
                    origin = origin @ make_transform(R=axis_rotation(bj.axis, config.LOCK_ANGLE))
                j = HandJoint(jn, f, k, bj.parent, ln, origin, bj.axis.copy(), bj.lower, bj.upper,
                              bj.effort, bj.velocity, locked, parent_dof=last_dof)
                if not locked:
                    j.dof = sum(not x.locked for x in self.joints)
                    last_dof = j.dof
                self.joints.append(j)
                self.links.append(ln)
        self.dof_joints = [j for j in self.joints if not j.locked]
        self.n_dof = len(self.dof_joints)
        assert self.n_dof == topology.n_dof
        self.link_parent_joint = {j.child: j for j in self.joints}
        self.region = {l: ("palm" if l == config.PALM else self.link_parent_joint[l].finger) for l in self.links}
        self.deformation = {l: link_deformation(self.base, l, geometry) for l in self.links}

    # ------------------------------------------------------------ kinematics

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.dof_joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.dof_joints])

    def link_dof_ancestors(self, link: str) -> list[int]:
        """Movable joints upstream of a link (the joints whose motion moves it)."""
        out = []
        while link != config.PALM:
            j = self.link_parent_joint[link]
            if not j.locked:
                out.append(j.dof)
            link = j.parent
        return sorted(out)

    def fk(self, q, wrist=None) -> dict[str, np.ndarray]:
        """Link frames for joint angles q (..., n_dof); wrist (..., 4, 4) is the palm's pose (identity if None)."""
        q = np.asarray(q, float)
        batch = q.shape[:-1]
        T = {config.PALM: np.broadcast_to(np.eye(4) if wrist is None else np.asarray(wrist, float), batch + (4, 4))}
        for j in self.joints:
            M = T[j.parent] @ j.origin
            if not j.locked:
                Rq = np.zeros(batch + (4, 4))
                Rq[..., :3, :3] = axis_rotation(j.axis, q[..., j.dof])
                Rq[..., 3, 3] = 1.0
                M = M @ Rq
            T[j.child] = M
        return T

    def dof_link_poses(self, q, wrist=None) -> np.ndarray:
        """World pose of each movable joint's frame, (..., n_dof, 4, 4): the overview's p_j(q)."""
        T = self.fk(q, wrist)
        return np.stack([T[j.child] for j in self.dof_joints], axis=-3)

    # ------------------------------------------------------------ graph (GET attention biases)

    def graph(self, include_wrist: bool = True, sibling_edges: bool = False) -> dict[str, np.ndarray]:
        """Shortest-path, ancestor and descendant distance matrices over [wrist] + movable joints.

        anc[i, j] = d > 0 if node j is an ancestor of node i at distance d (else 0); desc = anc.T.
        sibling_edges (GET-Zero's convention, used without a wrist node) links the finger roots.
        """
        parents = [j.parent_dof for j in self.dof_joints]
        if include_wrist:
            parents = [-1] + [p + 1 for p in parents]  # finger roots (-1) hang off the wrist, node 0
        n = len(parents)
        adj = np.zeros((n, n), bool)
        for i, p in enumerate(parents):
            if p >= 0:
                adj[i, p] = adj[p, i] = True
        if sibling_edges:
            roots = [i for i, p in enumerate(parents) if p == -1]
            for a in roots:
                for b in roots:
                    if a != b:
                        adj[a, b] = True
        spd = np.full((n, n), -1, int)
        for s in range(n):
            spd[s, s], frontier = 0, [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in np.flatnonzero(adj[u]):
                        if spd[s, v] < 0:
                            spd[s, v] = spd[s, u] + 1
                            nxt.append(v)
                frontier = nxt
        anc = np.zeros((n, n), int)
        for i in range(n):
            d, p = 1, parents[i]
            while p >= 0:
                anc[i, p] = d
                d, p = d + 1, parents[p]
        return {"spd": spd, "anc": anc, "desc": anc.T.copy()}

    # ------------------------------------------------------------ descriptor

    def descriptor(self) -> dict:
        depth = {}
        for j in self.dof_joints:
            depth[j.dof] = 1 if j.parent_dof < 0 else depth[j.parent_dof] + 1
        has_child = {j.parent_dof for j in self.dof_joints}
        joints = []
        for j in self.joints:
            joints.append({
                "name": j.name, "finger": j.finger, "k": j.k,
                "parent_link": j.parent, "child_link": j.child,
                "origin_xyz": np.round(j.origin[:3, 3], 9).tolist(),
                "origin_rpy": np.round(matrix_to_rpy(j.origin[:3, :3]), 9).tolist(),
                "origin_quat_wxyz": np.round(matrix_to_quat_wxyz(j.origin[:3, :3]), 9).tolist(),
                "axis": j.axis.tolist(), "lower": j.lower, "upper": j.upper,
                "locked": j.locked, "dof": j.dof, "parent_dof": j.parent_dof,
                "depth": depth.get(j.dof, -1), "leaf": (not j.locked) and (j.dof not in has_child),
            })
        return {
            "id": self.id,
            "topology": self.topology.id,
            "fingers": list(self.topology.fingers),
            "n_dof": self.n_dof,
            "geometry": self.geometry.to_dict(),
            "dof_names": [j.name for j in self.dof_joints],
            "links": [{"name": l, "type": link_type(l), "region": self.region[l],
                       "parent_joint": None if l == config.PALM else self.link_parent_joint[l].name}
                      for l in self.links],
            "joints": joints,
            "generator": {"kite_data": __version__, "get_zero_sha": config.GET_ZERO_SHA,
                          "menagerie_sha": config.MENAGERIE_SHA, "lock_angle": config.LOCK_ANGLE},
        }
