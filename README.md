# GET Your KITE: training data

Synthetic kinematic training data for a single morphology-conditioned decoder on LEAP Hand variants. The project is described in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) and the data plan in [DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md).

**Status:** the hand generator (Stage A) is done. The sampler that turns hands into (contacts, q_init, q_goal) examples comes next.

## Quickstart

```bash
./scripts/fetch_assets.sh            # pinned LEAP assets from GET-Zero and MuJoCo Menagerie -> third_party/
pip install -e ".[fast,dev]"         # fast = embreex (100x faster visibility rays)
pytest                               # ~20 s
python -m kite_data.build_hands      # all 243 topologies x 7 geometries -> data/hands/ (~4 min on 4 cores)
python -m kite_data.qa I4-M4-R4-T4_g0 I3-M0-R2-T2_g1 --pose random   # renders data/qa/sheet.png
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

## Third-party assets

These are not committed. `scripts/fetch_assets.sh` fetches them at pinned commits:
- **GET-Zero** (`real-stanford/get_zero` @ `279652d`): the base LEAP URDF and meshes. Its repository has no license file, so nothing from it is copied into this repo.
- **MuJoCo Menagerie** (`google-deepmind/mujoco_menagerie` @ `b065505`, `leap_hand/`, MIT): collision boxes and fingertip pad meshes.

The papers are listed in [reference/](reference/README.md).
