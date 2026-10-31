"""KITE's synthetic training examples (KITE §IV-B, App. A) for one hand.

Each example: a contact pose q_contact = (wrist pose, finger joints) that does not self-collide;
K ~ U{2..6} contacts, each a template point on a uniformly chosen non-empty region (palm or a
finger), drawn independently with replacement; their world positions and outward normals at
q_contact; a perturbed start pose q_init; the mask of joints upstream of a selected point; and the
label q_goal = mask * q_contact + (1 - mask) * q_init, with the wrist always taking q_contact's pose.

The wrist is a floating base: translation uniform in a cube, rotation uniform over SO(3) (KITE's
recipe; no arm is involved in training). z = psi(contacts) is not stored: it is computed from the
contacts once a psi exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .collision import Collider
from .hand import Hand

K_SLOTS = 6  # contacts are padded to this many slots; unused slots have template index -1


@dataclass(frozen=True)
class SamplerConfig:
    wrist_box: float = 0.40  # wrist translation ~ U[-b, b]^3 (m)
    k_min: int = 2
    k_max: int = K_SLOTS
    joint_noise_frac: float = 0.05  # q_init joint noise sigma = frac * joint range, then clamped
    wrist_pos_noise: float = 0.005  # m, per axis
    wrist_rot_noise_deg: float = 5.0  # per-axis sigma of the rotation vector (convention to confirm with KITE's authors)
    wide_fraction: float = 0.0  # 'wide' regime: share of examples with larger wrist noise
    wide_pos_noise: float = 0.05
    wide_rot_noise_deg: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


REGIMES = {"kite": SamplerConfig(), "wide": SamplerConfig(wide_fraction=0.2)}


def _quat_wxyz(R) -> np.ndarray:
    xyzw = Rotation.from_matrix(R).as_quat()
    q = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)  # canonical sign: w >= 0


def _matrix(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], axis=1)).as_matrix()


def _wrist_transforms(pos, R) -> np.ndarray:
    W = np.zeros((len(pos), 4, 4))
    W[:, :3, :3], W[:, :3, 3], W[:, 3, 3] = R, pos, 1.0
    return W


class HandSampler:
    def __init__(self, hand: Hand, template: dict, collider: Collider | None = None):
        self.hand = hand
        self.collider = collider or Collider(hand)
        self.tpl_link = np.asarray(template["link"], dtype=int)
        self.tpl_pos = np.asarray(template["pos"], dtype=float)
        self.tpl_normal = np.asarray(template["normal"], dtype=float)
        region = np.asarray(template["region"], dtype=int)
        self.tpl_region = region
        self.regions = np.unique(region)  # non-empty regions
        counts = np.array([np.sum(region == r) for r in self.regions])
        self.region_points = np.full((len(self.regions), counts.max()), -1, dtype=int)
        for i, r in enumerate(self.regions):
            self.region_points[i, : counts[i]] = np.flatnonzero(region == r)
        self.region_counts = counts
        self.link_ancestors = np.zeros((len(hand.links), hand.n_dof), dtype=bool)
        for li, link in enumerate(hand.links):
            self.link_ancestors[li, hand.link_dof_ancestors(link)] = True
        self.lower, self.upper = hand.lower, hand.upper
        self.accepted = self.tried = 0

    def contact_joints(self, n: int, rng) -> np.ndarray:
        """n joint configurations uniform within limits, rejecting self-collisions."""
        out = []
        while len(out) < n:
            need = n - len(out)
            batch = self.lower + rng.random((max(2 * need, 16), self.hand.n_dof)) * (self.upper - self.lower)
            for q in batch:
                self.tried += 1
                if not self.collider.collides(q):
                    out.append(q)
                    self.accepted += 1
                    if len(out) == n:
                        break
        return np.array(out)

    def contact_points(self, q, wrist_pos, wrist_R, tidx) -> tuple[np.ndarray, np.ndarray]:
        """World positions and normals of template points tidx (n, slots) at configuration (q, wrist)."""
        T = self.hand.fk(q, _wrist_transforms(wrist_pos, wrist_R))
        Tl = np.stack([T[l] for l in self.hand.links], axis=1)  # (n, n_links, 4, 4)
        valid = tidx >= 0
        safe = np.where(valid, tidx, 0)
        Ts = Tl[np.arange(len(q))[:, None], self.tpl_link[safe]]  # (n, slots, 4, 4)
        x = np.einsum("nsij,nsj->nsi", Ts[..., :3, :3], self.tpl_pos[safe]) + Ts[..., :3, 3]
        nrm = np.einsum("nsij,nsj->nsi", Ts[..., :3, :3], self.tpl_normal[safe])
        return np.where(valid[..., None], x, 0.0), np.where(valid[..., None], nrm, 0.0)

    def sample(self, n: int, rng, cfg: SamplerConfig = REGIMES["kite"]) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(rng)
        nd = self.hand.n_dof
        # contact pose
        qc = self.contact_joints(n, rng)
        pos_c = rng.uniform(-cfg.wrist_box, cfg.wrist_box, size=(n, 3))
        quat = rng.normal(size=(n, 4))
        quat_c = _quat_wxyz(_matrix(quat / np.linalg.norm(quat, axis=1, keepdims=True)))
        R_c = _matrix(quat_c)
        # contacts: region uniformly among non-empty regions, then a point uniformly within it
        k = rng.integers(cfg.k_min, cfg.k_max + 1, size=n)
        reg = rng.integers(0, len(self.regions), size=(n, K_SLOTS))
        within = np.floor(rng.random((n, K_SLOTS)) * self.region_counts[reg]).astype(int)
        tidx = self.region_points[reg, within]
        tidx[np.arange(K_SLOTS)[None, :] >= k[:, None]] = -1
        x, nrm = self.contact_points(qc, pos_c, R_c, tidx)
        # mask: joints upstream of any selected point (wrist always 1)
        valid = tidx >= 0
        mask = np.any(self.link_ancestors[self.tpl_link[np.where(valid, tidx, 0)]] & valid[..., None], axis=1)
        # start pose
        wide = rng.random(n) < cfg.wide_fraction
        joint_noise = rng.normal(size=(n, nd)) * cfg.joint_noise_frac * (self.upper - self.lower)
        q_init = np.clip(qc + joint_noise, self.lower, self.upper)
        sig_p = np.where(wide, cfg.wide_pos_noise, cfg.wrist_pos_noise)[:, None]
        sig_r = np.deg2rad(np.where(wide, cfg.wide_rot_noise_deg, cfg.wrist_rot_noise_deg))[:, None]
        pos_i = pos_c + rng.normal(size=(n, 3)) * sig_p
        R_i = R_c @ Rotation.from_rotvec(rng.normal(size=(n, 3)) * sig_r).as_matrix()
        # label
        q_goal = np.where(mask, qc, q_init)
        f32 = np.float32
        return {
            "k": k.astype(np.int8),
            "template_idx": tidx.astype(np.int16),
            "contact_pos": x.astype(f32),
            "contact_normal": nrm.astype(f32),
            "joints_contact": qc.astype(f32),
            "joints_init": q_init.astype(f32),
            "joints_goal": q_goal.astype(f32),
            "wrist_pos_contact": pos_c.astype(f32),
            "wrist_quat_contact": quat_c.astype(f32),
            "wrist_pos_init": pos_i.astype(f32),
            "wrist_quat_init": _quat_wxyz(R_i).astype(f32),
            "wrist_pos_goal": pos_c.astype(f32),
            "wrist_quat_goal": quat_c.astype(f32),
            "mask": mask,
            "wide": wide,
        }


def verify(sampler: HandSampler, ex: dict, atol: float = 2e-5, check_collisions: int = 0) -> dict:
    """Re-derive the invariants from stored arrays. Returns max errors and pass/fail flags."""
    hand = sampler.hand
    tidx = ex["template_idx"].astype(int)
    valid = tidx >= 0
    k_ok = bool(np.all(valid.sum(axis=1) == ex["k"]) and np.all(valid[:, :2]) and np.all(ex["k"] >= 2))
    R_goal = _matrix(ex["wrist_quat_goal"])
    x, nrm = sampler.contact_points(ex["joints_goal"].astype(float), ex["wrist_pos_goal"].astype(float), R_goal, tidx)
    pos_err = float(np.abs(x - ex["contact_pos"])[valid].max())
    nrm_err = float(np.abs(nrm - ex["contact_normal"])[valid].max())
    lo, hi = sampler.lower - 1e-6, sampler.upper + 1e-6
    limits_ok = all(bool(np.all((ex[k] >= lo) & (ex[k] <= hi))) for k in ("joints_contact", "joints_init", "joints_goal"))
    expect_mask = np.any(sampler.link_ancestors[sampler.tpl_link[np.where(valid, tidx, 0)]] & valid[..., None], axis=1)
    mask_ok = bool(np.array_equal(expect_mask, ex["mask"]))
    goal_ok = bool(np.array_equal(ex["joints_goal"], np.where(ex["mask"], ex["joints_contact"], ex["joints_init"])))
    collisions = 0
    if check_collisions:
        for q in ex["joints_contact"][:check_collisions]:
            collisions += sampler.collider.collides(q.astype(float))
    ok = pos_err < atol and nrm_err < atol and k_ok and limits_ok and mask_ok and goal_ok and collisions == 0
    return {"ok": ok, "contact_pos_err": pos_err, "contact_normal_err": nrm_err, "k_ok": k_ok, "limits_ok": limits_ok,
            "mask_ok": mask_ok, "goal_ok": goal_ok, "collisions": collisions}
