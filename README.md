# GET Your KITE: training data

Synthetic kinematic training data for a single morphology-conditioned decoder on LEAP Hand variants. The project is described in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) and the data plan in [DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md).

**Status:** the hand generator, splits and example sampler are done. z = ψ(contacts) is not stored: it can be computed from the stored contacts once a ψ exists (KITE's, or our own).

## Quickstart

```bash
./scripts/fetch_assets.sh            # pinned LEAP assets from GET-Zero and MuJoCo Menagerie -> third_party/
pip install -e ".[fast,dev]"         # fast = embreex (100x faster visibility rays)
pytest                               # ~30 s
python -m kite_data.build_hands      # all 243 topologies x 7 geometries -> data/hands/ (~4 min on 4 cores)
python -m kite_data.splits           # hand-level splits -> data/splits.json
python -m kite_data.generate         # 2.4M examples for every split -> data/examples/kite/ (~4 min, 850 MB)
python -m kite_data.qa hands I4-M4-R4-T4_g0 I3-M0-R2-T2_g1 --pose random   # data/qa/hands.png
python -m kite_data.qa examples data/examples/kite/val/I2-M4-R3-T3_g1.npz  # data/qa/examples.png
```

## The hand family

- **Topologies (243, 189 with a thumb).** Each finger (index, middle, ring, thumb) is absent or keeps 2, 3 or 4 joints, and every hand has at least 2 fingers. A finger with fewer joints keeps all its links, and its joints are locked from the fingertip inward at 0 rad. IDs read `I4-M3-R0-T2`.
- **Geometry.** Draw 0 is the nominal LEAP Hand. Draws 1 to 6 are random:
  - Each of the five long links (`mcp_joint`, `dip` and `fingertip` per finger; `thumb_dip` and `thumb_fingertip`) gets a length scale in [0.8, 1.2]. Only the middle of the link stretches, so motors and fingertip pads keep their shape.
  - The index, middle and ring mounts shift by up to ±4 mm along the palm, and their motor housings move with them.
  - A hand is redrawn if it collides at the zero pose or leaves less than 5% of random poses collision-free.
- **Hand IDs** are `<topology>_g<draw>`, e.g. `I4-M4-R4-T4_g0`.

![Eight hands from the family in random collision-free poses, drawn from their contact templates](docs/hands_random_poses.png)

*Eight random hands in random collision-free poses, drawn from their 4,096-point contact templates: palm grey, index blue, middle green, ring orange, thumb red; x = the 32 farthest-point samples.*

## Outputs (`data/hands/<hand_id>/`)

| File | Contents |
| --- | --- |
| `hand.json` | Descriptor: joints in canonical order (index, middle, ring, thumb; proximal to distal; locked joints flagged), origins, axes, limits, movable-parent indices, depth, leaf flags, links and their regions, geometry parameters, gate statistics and generator version. |
| `hand.xml` | Collision-only MJCF: MuJoCo Menagerie's hand-tuned boxes and fingertip pads, deformed with the geometry; palm fixed to the world. |
| `hand.urdf` | URDF with exact kinematics. Visuals reference the base meshes, with stretched links scaled approximately. |
| `template.npz` | 4,096 surface points (KITE's contact template): `link` (index into `link_names`), `pos` and `normal` (link frame), `region` (index into `region_names`), `fps` (32 farthest-point indices). |

`data/hands/index.csv` has one row per hand (DoF, fingers, gate result, collision-free rate, template points per region). `data/hands/summary.json` has build totals.

## Using a hand in code

```python
import numpy as np
from kite_data.build_hands import make_hand
from kite_data.collision import Collider

hand = make_hand("I3-M0-R2-T4", draw=2, seed=0, attempt=0)   # attempt: see hand.json["build"]
q = np.zeros(hand.n_dof)
frames = hand.fk(q)                    # link name -> 4x4 pose (palm = wrist frame)
p = hand.dof_link_poses(q)             # (n_dof, 4, 4): each movable joint's frame, the overview's p_j(q)
g = hand.graph()                       # shortest-path / ancestor / descendant matrices, wrist = node 0
free = not Collider(hand).collides(q)  # fast self-collision check (~0.3 ms)
```

## Splits (`data/splits.json`)

Splits are by hand, fixed before any example is sampled:

| Split | Hands | What it holds out |
| --- | --- | --- |
| `train` | 1,015 | 145 layouts, draws 0–6 |
| `val` | 56 | 8 other layouts, draws 0–6, for model selection only |
| `test_finger_subset` | 36 | Hands whose fingers are exactly {thumb, ring} or {thumb, index, ring}, nominal geometry |
| `test_combined` | 216 | The same layouts with random geometry (draws 1–6) |
| `test_joint_count` | 378 | Every other hand with a 3-joint index finger, draws 0–6 |
| `test_geometry` | 40 | 10 training layouts with geometries training never sees (draws 7–10) |

Thumb and ring still appear together in larger training hands; only the exact finger sets are held out.

## Training examples (`data/examples/kite/<split>/<hand_id>.npz`)

Each file holds one hand's examples, generated with KITE's recipe (KITE §IV-B, App. A):

1. A contact pose: finger joints uniform within limits, rejected if the hand self-collides. The wrist (KITE's floating base) is uniform in [−0.4, 0.4]³ m with a uniformly random orientation; no arm is involved in training.
2. K ~ U{2, …, 6} contacts. Each picks a non-empty region (palm or a finger) uniformly, then a template point uniformly within it, with replacement.
3. A start pose: joints perturbed by N(0, (5% of range)²) and clamped; wrist by 5 mm and 5° (per axis).
4. The label: joints upstream of a contact take the contact pose; all other joints stay at the start pose. The wrist takes the contact pose.

Default sizes are 1,000 examples per training hand (about 1M in total, as in the overview), 1,000 per validation hand and 2,000 per test hand; override with `--per-hand train=4000`. A `wide` regime (`--regime wide`) gives 20% of examples 5 cm / 30° wrist noise, for robustness ablations.

| Key | Shape | Contents |
| --- | --- | --- |
| `k` | (n,) | number of contacts, 2–6 |
| `template_idx` | (n, 6) | template points that make the contacts; −1 for unused slots |
| `contact_pos`, `contact_normal` | (n, 6, 3) | contact targets in the world frame (zero in unused slots) |
| `joints_contact`, `joints_init`, `joints_goal` | (n, n_dof) | finger joints in the hand's canonical order |
| `wrist_pos_*`, `wrist_quat_*` | (n, 3), (n, 4) | wrist (palm) pose for contact / init / goal; quaternions are w, x, y, z with w ≥ 0 |
| `mask` | (n, n_dof) | joints that move (upstream of a contact); the wrist always moves |
| `wide` | (n,) | whether the example used the wide noise regime |

![Four training examples: start pose in grey, label pose in colour, contact targets and normals in black](docs/example_targets.png)

*Four examples for one validation hand: the start pose (grey), the label (coloured; only joints upstream of a contact move) and the K contact targets with their outward normals (black).*

Before writing, every file is checked by re-deriving the contacts from the stored label (they agree to within 1e-7 m). The checks also confirm joint limits, the mask, the label and that contact poses don't collide. `kite_data.dataset.load_split("train")` loads a split into arrays padded to 16 joints (the 1M training examples load in ~5 s and take ~0.9 GB). The example view in `kite_data.qa` draws the start pose, the label and the contacts.

## Third-party assets

These are not committed. `scripts/fetch_assets.sh` fetches them at pinned commits:
- **GET-Zero** (`real-stanford/get_zero` @ `279652d`): the base LEAP URDF and meshes. Its repository has no license file, so nothing from it is copied into this repo.
- **MuJoCo Menagerie** (`google-deepmind/mujoco_menagerie` @ `b065505`, `leap_hand/`, MIT): collision boxes and fingertip pad meshes.

The papers are listed in [reference/](reference/README.md).
