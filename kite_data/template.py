"""Surface contact template (KITE App. A): 4096 surface points per hand with outward normals, stored in
their link frames, plus 32 farthest-point-sampled indices (KITE's decoder surface tokens, App. B).

LEAP's CAD meshes contain a lot of internal geometry (motor internals, screw holes): 18-36% of
area-sampled points are fully enclosed. Candidate points are therefore filtered by visibility: a point
is kept only if enough rays from its outward hemisphere escape its own part.
"""

from __future__ import annotations

import functools
import hashlib
import warnings
import zlib
from dataclasses import dataclass

import numpy as np

from . import config
from .hand import Hand
from .leap import link_mesh, link_type


@dataclass
class Pool:
    points: np.ndarray  # (N, 3) link frame
    normals: np.ndarray  # (N, 3)
    escape: np.ndarray  # (N,) rays out of VISIBILITY_RAYS that escaped the part
    area_per_point: float

    @property
    def visible(self) -> np.ndarray:
        return self.escape >= config.VISIBILITY_MIN_ESCAPE


def _sphere_directions(n: int) -> np.ndarray:
    """Fibonacci sphere (deterministic, near-uniform)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _pool_key(link: str) -> str:
    bl = link_mesh(link)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(bl.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(bl.faces).tobytes())
    h.update(repr((config.CANDIDATES_PER_MESH, config.VISIBILITY_RAYS, 1)).encode())
    return f"{link}_{h.hexdigest()[:12]}"


@functools.lru_cache(maxsize=None)
def candidate_pool(link: str) -> Pool:
    """Area-uniform candidates on a base link mesh with their visibility, cached in data/cache."""
    path = config.CACHE_DIR / "template_pools" / f"{_pool_key(link)}.npz"
    if path.exists():
        z = np.load(path)
        return Pool(z["points"], z["normals"], z["escape"], float(z["area_per_point"]))
    import trimesh

    mesh = link_mesh(link)
    if "embree" not in type(mesh.ray).__module__:
        warnings.warn("embreex not installed: the visibility pass will be slow (pip install embreex)")
    n = config.CANDIDATES_PER_MESH
    seed = zlib.crc32(link.encode())
    points, fidx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[fidx]
    dirs = _sphere_directions(config.VISIBILITY_RAYS)
    flip = np.where(normals @ dirs.T < 0, -1.0, 1.0)  # (N, K): every ray in the outward hemisphere
    ray_dirs = (flip[:, :, None] * dirs[None]).reshape(-1, 3)
    origins = np.repeat(points + 1e-5 * normals, len(dirs), axis=0)
    hit = mesh.ray.intersects_any(origins, ray_dirs).reshape(n, len(dirs))
    pool = Pool(points.astype(np.float64), normals.astype(np.float64), (~hit).sum(axis=1).astype(np.int16), mesh.area / n)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=pool.points, normals=pool.normals, escape=pool.escape, area_per_point=pool.area_per_point)
    return pool


def farthest_point_sampling(P: np.ndarray, k: int) -> np.ndarray:
    """Deterministic FPS, starting from the point farthest from the centroid."""
    idx = [int(np.argmax(np.linalg.norm(P - P.mean(axis=0), axis=1)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return np.array(idx)


def build_template(hand: Hand, seed: int = 0) -> dict[str, np.ndarray]:
    pts, nrm, wts, lnk = [], [], [], []
    for li, link in enumerate(hand.links):
        pool = candidate_pool(link_type(link))
        vis = pool.visible
        P, N = pool.points[vis], pool.normals[vis]
        D = hand.deformation[link]
        wts.append(pool.area_per_point * D.area_factor(P, N))
        pts.append(D.map_points(P))
        nrm.append(D.map_normals(P, N))
        lnk.append(np.full(len(P), li))
    P, N, W, L = np.concatenate(pts), np.concatenate(nrm), np.concatenate(wts), np.concatenate(lnk)
    rng = np.random.default_rng([seed, zlib.crc32(hand.id.encode())])
    pick = np.sort(rng.choice(len(P), size=config.TEMPLATE_SIZE, replace=False, p=W / W.sum()))
    P, N, L = P[pick], N[pick], L[pick]
    region_of_link = np.array([config.REGIONS.index(hand.region[l]) for l in hand.links])
    T0 = hand.fk(np.zeros(hand.n_dof))
    P0 = np.empty_like(P)
    for li, link in enumerate(hand.links):
        m = L == li
        P0[m] = P[m] @ T0[link][:3, :3].T + T0[link][:3, 3]
    return {
        "link": L.astype(np.int16),
        "pos": P.astype(np.float32),
        "normal": N.astype(np.float32),
        "region": region_of_link[L].astype(np.int8),
        "fps": farthest_point_sampling(P0, config.FPS_SIZE).astype(np.int16),
        "link_names": np.array(hand.links),
        "region_names": np.array(config.REGIONS),
    }
