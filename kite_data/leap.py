"""The base LEAP Hand: kinematics and meshes from GET-Zero's URDF, collision geometry from MuJoCo Menagerie.

Both sources use the same link frames (checked to 0.001 deg / 0.11 mm), so Menagerie's collision
geoms can be attached to the URDF links unchanged.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .transforms import make_transform, quat_wxyz_to_matrix, rpy_to_matrix


def _floats(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def link_type(link: str) -> str:
    """'mcp_joint_2' -> 'mcp_joint'. Thumb links have unique names and keep them."""
    for suffix in ("_2", "_3"):
        if link.endswith(suffix) and not link.startswith("thumb"):
            return link[: -len(suffix)]
    return link


@dataclass
class BaseJoint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4, parent-link frame -> joint (= child-link) frame at q = 0
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass
class BaseLink:
    name: str
    mesh_file: Path
    visual_origin: np.ndarray  # 4x4, link frame <- mesh frame
    mass: float
    inertial_origin: np.ndarray
    inertia: np.ndarray  # ixx ixy ixz iyy iyz izz


@dataclass
class CollisionGeom:
    """A Menagerie collision geom expressed in our link frame."""

    link: str
    kind: str  # "box" or "mesh"
    pos: np.ndarray
    rot: np.ndarray  # 3x3
    half_size: np.ndarray | None = None  # box
    vertices: np.ndarray | None = None  # mesh (convex hull is what MuJoCo collides with)
    name: str = ""


@dataclass
class BaseModel:
    links: dict[str, BaseLink]
    joints: dict[str, BaseJoint]
    collision: dict[str, list[CollisionGeom]] = field(default_factory=dict)
    exclude: list[tuple[str, str]] = field(default_factory=list)

    def child_joint(self, link: str) -> BaseJoint | None:
        for j in self.joints.values():
            if j.parent == link:
                return j
        return None

    def parent_joint(self, link: str) -> BaseJoint | None:
        for j in self.joints.values():
            if j.child == link:
                return j
        return None

    @functools.cached_property
    def long_link_lengths(self) -> dict[str, float]:
        """Nominal length L of every long link along its distal axis: the child-joint offset, or
        the mesh's farthest extent for fingertips."""
        out = {}
        for name in self.links:
            t = link_type(name)
            if t not in config.LONG_LINKS:
                continue
            sign = config.LONG_LINKS[t]
            child = self.child_joint(name)
            if child is not None:
                out[name] = float(sign * child.origin[1, 3])
            else:
                out[name] = float(np.max(sign * link_mesh(name).vertices[:, 1]))
            assert out[name] > 0.0, f"{name}: long axis sign is wrong"
        return out


@functools.lru_cache(maxsize=None)
def load_base_model(urdf: Path = config.BASE_URDF, menagerie_xml: Path = config.MENAGERIE_XML) -> BaseModel:
    if not Path(urdf).exists():
        raise FileNotFoundError(f"{urdf} is missing; run scripts/fetch_assets.sh first")
    root = ET.parse(urdf).getroot()
    links = {}
    for l in root.findall("link"):
        v = l.find("visual")
        vo = v.find("origin")
        inert = l.find("inertial")
        io = inert.find("origin")
        I = inert.find("inertia").attrib
        links[l.attrib["name"]] = BaseLink(
            name=l.attrib["name"],
            mesh_file=config.LEAP_DIR / Path(v.find("geometry/mesh").attrib["filename"]).name,
            visual_origin=make_transform(_floats(vo.get("xyz")), rpy_to_matrix(_floats(vo.get("rpy")))),
            mass=float(inert.find("mass").attrib["value"]),
            inertial_origin=make_transform(_floats(io.get("xyz")), rpy_to_matrix(_floats(io.get("rpy")))),
            inertia=np.array([float(I[k]) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]),
        )
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        joints[j.attrib["name"]] = BaseJoint(
            name=j.attrib["name"],
            parent=j.find("parent").attrib["link"],
            child=j.find("child").attrib["link"],
            origin=make_transform(_floats(o.get("xyz")), rpy_to_matrix(_floats(o.get("rpy")))),
            axis=_floats(j.find("axis").attrib["xyz"]),
            lower=float(lim.attrib["lower"]),
            upper=float(lim.attrib["upper"]),
            effort=float(lim.attrib.get("effort", 0.95)),
            velocity=float(lim.attrib.get("velocity", 8.48)),
        )
    model = BaseModel(links=links, joints=joints)
    model.collision, model.exclude = _load_menagerie_collision(Path(menagerie_xml))
    return model


def _load_menagerie_collision(xml: Path):
    import mujoco

    if not xml.exists():
        raise FileNotFoundError(f"{xml} is missing; run scripts/fetch_assets.sh first")
    m = mujoco.MjModel.from_xml_path(str(xml))
    body_to_link = {b: l for l, b in config.MENAGERIE_BODY.items()}
    geoms: dict[str, list[CollisionGeom]] = {}
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        link = body_to_link[body]
        pos = m.geom_pos[g].copy()
        rot = quat_wxyz_to_matrix(m.geom_quat[g])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            geoms.setdefault(link, []).append(
                CollisionGeom(link, "box", pos, rot, half_size=m.geom_size[g].copy(), name=name)
            )
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            start, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[start : start + n] @ rot.T + pos  # into the body (= link) frame
            geoms.setdefault(link, []).append(
                CollisionGeom(link, "mesh", np.zeros(3), np.eye(3), vertices=verts, name=name)
            )
        else:
            raise ValueError(f"unexpected collision geom type on {body}")
    exclude = []
    for i in range(m.nexclude):
        sig = int(m.exclude_signature[i])
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, sig >> 16)
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, sig & 0xFFFF)
        exclude.append((body_to_link[b1], body_to_link[b2]))
    return geoms, exclude


@functools.lru_cache(maxsize=None)
def _load_mesh_file(path: Path):
    import trimesh

    return trimesh.load(path, force="mesh", process=False)


@functools.lru_cache(maxsize=None)
def link_mesh(link: str):
    """The link's visual mesh in its own link frame (undeformed)."""
    base = load_base_model()
    bl = base.links[link]
    mesh = _load_mesh_file(bl.mesh_file).copy()
    mesh.apply_transform(bl.visual_origin)
    return mesh
