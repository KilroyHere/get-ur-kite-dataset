# Training Data Generation Plan

Status: 2026-10-31. Stages A–C (hand generator, splits, example sampler) are implemented in `kite_data/`; see [README.md](README.md). The z cache waits for a ψ.

Decisions so far:
- LEAP hands only.
- Fewer joints means locking joints from the fingertip inward (not GET-Zero's link removal).
- Lengths vary only on the five long links.
- Thumbless hands are included.
- Wrist poses follow KITE's floating cube, not an arm workspace.
- The finger-subset test holds out exact finger sets.
- About 1M training examples by default, as in the overview.

This plan covers the synthetic kinematic data for the GET decoder in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (section "Training data"). It follows KITE's own recipe wherever KITE specifies one. Our contacts should look like what KITE's ψ was trained on, and the decoder should expect intents one small step ahead of the current pose, which is how KITE's policy produces them.

Sources: the KITE and GET-Zero papers (links and a fetch script in [reference/](reference/README.md)), GET-Zero's code ([real-stanford/get_zero @ 279652d](https://github.com/real-stanford/get_zero/tree/279652da160823521e881952f67870b0d98674ce)), MuJoCo Menagerie's LEAP model ([mujoco_menagerie/leap_hand @ b065505](https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/leap_hand)), and checks run on those assets (section 2).

## 1. What the papers and code settle

| Question | Answer | Source |
| --- | --- | --- |
| Is KITE's code or a pretrained ψ available? | No. The project page says "Code soon". The paper describes ψ and the decoder in enough detail to re-implement them. | [kite-manip.github.io](https://kite-manip.github.io/), KITE App. B |
| What does ψ take as input? | An unordered set of 2–6 contacts. Each contact is a world-frame position plus the robot's outward surface normal there. Contacts carry no finger or link identity. A permutation-invariant attention encoder maps the set to z ∈ R^256. | KITE Eq. 1–2, App. B |
| How is ψ trained? | Jointly with the decoder bank, on synthetic data from every embodiment including the targets, then frozen. A new embodiment is added by fine-tuning only the decoder bank. KITE's embodiments are Robotiq 2F-85, Barrett, Allegro, Wuji, XHand and Robotiq+Stick, plus MANO for human demos. LEAP is not among them. | KITE §IV-B, §IV-D, §V-A |
| How are contacts stored? | A surface template of 4,096 points per embodiment. Each point is a position and an outward normal in its parent link's frame, and FK maps them to the world. | KITE App. A |
| How are contacts sampled? | K ~ U{2, …, 6}. For each contact, pick a non-empty region (one finger or the palm) uniformly, then a template point uniformly within it. Contacts are independent and drawn with replacement. | KITE App. A |
| How are configurations sampled? | Joints are uniform within limits. The floating base (our wrist) has translation uniform in [−0.40, 0.40]³ m and rotation uniform on SO(3). Self-colliding configurations are discarded. There is no arm during training; IK is used only at deployment. | KITE §IV-B, App. A, App. B-A |
| How is q_init perturbed? | Gaussian noise with σ = 5% of each joint's range, 0.005 m on base translation and 5° on base rotation. Joints are then clamped to their limits. | KITE §IV-B, App. A |
| What are the mask and label? | m_j = 1 for every joint upstream of a selected point, and q_goal = m ⊙ q_contact + (1 − m) ⊙ q_init. | KITE Eq. 3, App. A |
| How much data? | 10^6 training and 1,000 validation examples per embodiment, trained for 500 epochs at batch size 128 per embodiment. Adding one embodiment took 50 epochs, about 10 h on one RTX 5090. | KITE App. B, §V-C |
| What does the decoder output? | A residual. Translation and joint residuals are added to q_init. The rotation residual uses the 6D representation (the first two columns of the rotation matrix) and is composed with the initial rotation. | KITE App. B |
| Does z say which finger should touch? | No. The decoder infers the assignment from q_init, so its success depends on the start pose. Oracle-intent success is 85% from the base start pose. It is 80% with a 5 cm / 30° offset, 78% with 20 cm / 60°, 66% with a 180° flip and 57% with a 30 cm shift. | KITE §II-B, §V-D, Tab. III |
| What does GET-Zero's generator vary? | It removes intermediate links rather than truncating fingers or locking joints. Each main finger has 5 options: 4 joints; 3 (drops the `dip` link); 2 (`dip` + `fingertip` mounted on the palm); 1 (`fingertip` on the palm); or absent. The thumb always has 4 or 3 joints, and every hand has one. This gives 236 topologies. Lengths change only through discrete +1.5 cm printed parts (IDs 237–633). It has no continuous scaling, no shortening, no attachment shifts and no thumbless hands. | GET-Zero §IV-A, `rl/scripts/gen_leap_assets.py` |
| How does GET-Zero split hands? | 44 training graphs (the hands with working RL experts). Tests are 10 new graphs, 10 new geometries (training graphs with +1.5 cm parts), and 10 with both. | GET-Zero §V, `distill/cfg/dataset/LeapHandRot.yaml` |
| What graph features does GET-Zero use? | Per-head learned attention biases from shortest-path, parent and child distances. Finger roots on the palm count as adjacent (shortest-path distance 1). The self-model target is each joint's 3D position. Token geometry is joint origin xyz + rpy. Joint limits are left out, and so are axes, because every LEAP axis is (0, 0, −1). | GET-Zero §III, `distill/utils/embodiment_util.py`, `distill/cfg/tokenization/Default.yaml` |

**Where this changes PROJECT_OVERVIEW.md.** The overview is left unchanged; README.md lists every way the data departs from it.

- **ψ and π are not released.** The "frozen KITE parts" do not exist for us yet, so the oracle-intent and closed-loop evaluation modes are blocked on the release (fallbacks in section 3).
- **KITE details are off.** KITE has three authors (Wang, Jin, Fang) and six simulated robot embodiments, not five.
- **The wrist needs no arm workspace.** KITE samples it as a floating base anywhere in a 0.8 m cube, at any orientation. That answers the workspace open question: the arm only matters at deployment.
- **The training budget may be low.** KITE uses 10^6 examples per embodiment. We follow the overview's ~1M total by default, which is about 1,000 per hand. Its 2–6 GPU-hour training estimate is also far below KITE's, so run a scaling test before fixing the budget.
- **The variant table needs new code.** GET-Zero's link removal shortens a finger as it loses joints: the index finger reaches 140 mm with 4 joints, 105 mm with 3 and 88 mm with 2. A joint-count test on those hands would also be a length test. We therefore lock joints instead (decided), and wrote our own generator with continuous lengths and mount shifts, which GET-Zero lacks.

## 2. Measured facts about the LEAP assets

These were checked on 2026-10-31 with MuJoCo 3.17, trimesh and python-fcl, on GET-Zero's URDFs and STLs. The scripts are not committed.

- **FK conventions are unambiguous.** A short numpy URDF FK (fixed-axis XYZ rpy) matches MuJoCo's URDF import to 1e-16 m on hands 001, 013, 139 and 600.
- **Topology counts.** Take the overview's family: 2–4 fingers, each present finger with 2–4 joints.
  - *Built with GET-Zero's link removal*, where the thumb can only have 3 or 4 joints, it has 126 topologies with a thumb.
  - *Built with joint locking*, which is what we use and lets the thumb have 2–4 joints too, it has 243, of which 189 have a thumb.
- **Self-collision is common.** Under uniform joint sampling with an exact, non-convex mesh check (parent–child pairs excluded), about 20% of configurations are collision-free for the full hand. The rate is 45% for hand 013 and 58% for hand 139. Most rejections come from neighbouring fingers swinging into each other, the thumb hitting the index finger, and the thumb hitting the palm.
- **MuJoCo's default collision is unusable on these meshes.** MuJoCo wraps each STL in its convex hull, and the palm's hull swallows every finger root. As a result, 0% of configurations pass, including q = 0.
- **Exact mesh checks are slow.** They take 8–13 ms each. Decimating the meshes makes them only about 4× faster and disagrees with the exact check on 8–12% of configurations.
- **Menagerie's box primitives work.** The LEAP model in MuJoCo Menagerie (MIT-licensed) has hand-tuned box primitives in the same link frames as GET-Zero's URDF; they differ by at most 0.11 mm, which is rounding. On 1,500 identical full-hand configurations the boxes agree with the exact check 95.3% of the time (1.5% missed collisions, 3.1% false ones). Each check takes 0.34 ms, about 38× faster. Menagerie's thumb joint limits differ from GET-Zero's, so we keep GET-Zero's limits (LEAP's official ones) and use Menagerie only for the boxes.
- **Surface quality is the hidden problem.** On the LEAP link meshes, 18–36% of area-sampled surface points are fully enclosed: none of 32 rays in the normal's hemisphere escapes. These are motor internals, screw holes and other internal CAD geometry, and another 9–19% sit in deep recesses. A contact sampled there is physically meaningless. Most of these meshes are not watertight, but their winding is consistent and their volumes are positive, so face normals do point outward.
- **Licenses.** GET-Zero's repository has no license file, so we fetch its code and generated URDFs at a pinned commit instead of copying them in. Menagerie's LEAP model is MIT-licensed.

## 3. Questions still open

**For the KITE authors** (Qianxu Wang and Kuan Fang, Cornell):

1. When will the code be released, and can we get early access to ψ and the decoder bank? Ideally also the task policies, demonstrations and MuJoCo scenes. This decides whether "frozen KITE ψ and π" is possible at all.
2. How were the 4,096 template points sampled (area-weighted? which mesh?), were internal or occluded surfaces removed, and how are finger and palm regions defined?
3. How is self-collision checked: which checker, which collision geometry, and which pairs are excluded?
4. Where does the world origin sit in their scenes relative to the table and the arm base? The training cube is [−0.4, 0.4]³ m around it, so if we reuse ψ, our scenes must use the same frame.
5. What exactly is the 5° rotation noise, and how are the "per-coordinate scale factors" computed? An isotropic tangent-space vector with σ = 5° per axis gives a mean angle of about 8°; a random axis with angle ~ N(0, 5°) gives about 4°.
6. Did they try larger q_init noise, given Table III?

**For the team:**

1. ~~LEAP only or other hand designs too?~~ Decided: LEAP only.
2. ~~How to make fingers with fewer joints?~~ Decided: lock joints from the fingertip inward at 0 rad. GET-Zero's link-removal hands become an extra test set.
3. ~~Which links change length?~~ Decided: the five long links. The motor-to-motor brackets stay fixed.
4. ~~Thumbless hands?~~ Decided: included (54 of the 243 topologies).
5. If ψ isn't available in time, there are two options. (a) Re-implement ψ and train it jointly with our decoder. That gives us our own intent space but no KITE policy, so closed-loop evaluation needs scripted intent sequences. (b) Use the raw-contact fallback. The dataset supports both. (KITE's code and data are expected.)
6. Which upper bound to compare against? Options: a re-implemented KITE decoder trained on each test hand (as the overview says), our GET decoder trained with the test hands included, or none. Test hands get their own training data either way.
7. What length range? GET-Zero's +1.5 cm parts correspond to scale factors of about 1.2–1.4, beyond 0.8–1.2. Recommendation: keep 0.8–1.2 for training and use those parts as an extrapolation test.
8. ~~Which split definitions?~~ Decided: exact finger sets {thumb, ring} and {thumb, index, ring} are held out (section 4.3). The literal "never together" reading would remove all 144 layouts that contain both, including every 4-finger hand.
9. What GPU budget? The default is 1,000 examples per training hand (about 1M, as in the overview); raise it with `--per-hand` if a scaling test shows the decoder needs more.
10. Which noise regime? KITE's is the default. A wide-noise mixture is implemented (`--regime wide`) for an ablation.
11. Should the contact loss be tied to the labelled points, as in the overview, or be assignment-free, scoring each target against its nearest template point? Both need the template, which we store either way.
12. ~~Wrist poses from an arm's workspace or KITE's cube?~~ Decided: KITE's cube. This deviates from the overview's "w* ∈ arm workspace". It keeps contacts in the range KITE's ψ was trained on, keeps the data arm-agnostic, and needs no arm choice yet. Changing it later only means regenerating.

## 4. Plan

### 4.1 Principles

- **Match KITE wherever it is specific:** sampling, noise, mask, K, regions and the 4,096-point template.
- **Store raw values; derive the rest in code.** Store contacts, template indices and configurations. Compute z, graph matrices and normalized features from them with versioned code, so a change of convention never forces regeneration.
- **Split by hand, before sampling.** Nothing from a held-out hand reaches training, and geometry variants of one topology never straddle splits.
- **Make everything deterministic.** Every output follows from (config, seed) and records the generator's git SHA.

### 4.2 Stage A: hand assets (implemented)

1. **Fetch.** `scripts/fetch_assets.sh` clones GET-Zero (279652d) and Menagerie (b065505) at pinned commits into `third_party/`, which is git-ignored. From GET-Zero we use only the base LEAP URDF, its meshes and joint limits; its variant code is not used. Its generated hands are fetched too, for the extra test set (importer still to do).
2. **Topologies.** Each finger is absent or has 2, 3 or 4 joints; a hand has at least 2 fingers. That gives 243 topologies, 189 of them with a thumb. A finger with fewer joints keeps every link: its joints are locked from the fingertip inward at 0 rad, so its length doesn't change. IDs read `I4-M3-R0-T2` (joints per index, middle, ring, thumb).
3. **Geometry.** Each hand instance is one topology plus one geometry draw. Draw 0 is the nominal LEAP; draws 1–6 are random and deterministic in (seed, topology, draw).
   - *Link lengths:* s ~ U[0.8, 1.2] independently for each of the five long links (`mcp_joint`, `dip`, `fingertip`, `thumb_dip`, `thumb_fingertip`). The stretch runs along the link-frame y axis, the axis GET-Zero extends, and only the middle half of the link [0.25 L, 0.75 L] is stretched, so motor housings and fingertip pads keep their shape. The child joint moves by Δ = (s − 1)·L. The brackets `pip`, `pip_4` and `thumb_pip` stay fixed.
   - *Mounts:* index, middle and ring roots shift by δ ~ U[−4, 4] mm along the palm's finger-spacing axis (spacing is 45.5 mm). The palm's motor housing for that finger (Menagerie's box, grown by 2 mm) moves with it, fading out over 7 mm, so the palm never folds. The thumb mount stays fixed: its housing is fused into the palm mesh with no clear boundary.
   - *Validity gate*, otherwise redraw:
     - no box collision at q = 0;
     - no exact-mesh intersection at q = 0 among the pairs the box model excludes (palm vs. finger bases, neighbouring bases), checked on simplified meshes;
     - at least 5% collision-free under uniform joint sampling.

     The simplified check matched full-resolution all-pairs checks on 28 of 28 hands, including the extremes of every range.
4. **Collision model.** Menagerie's boxes and fingertip-pad meshes are attached to each link (frames match to 0.001°/0.11 mm) and deformed with the geometry. Menagerie's exclusion pairs are kept. The model is written as a collision-only MJCF.
5. **Surface template** (KITE's T^e).
   - 20,000 area-uniform candidates per base mesh, each tested with 32 hemisphere rays against its own part.
   - Points where fewer than 8 rays escape are dropped. This visibility pass is cached in `data/cache/`.
   - Per hand, 4,096 points are drawn in proportion to their deformed surface area.
   - Each point stores its link index, link-frame position, outward normal and region (palm, index, middle, ring, thumb).
   - The template also keeps 32 farthest-point-sampled indices at the zero pose, for KITE-style surface tokens.
6. **Hand descriptor** (`hand.json`).
   - *Joint order:* index, middle, ring, thumb; proximal to distal; locked joints included and flagged.
   - *Per joint:* parent link, child link, origin xyz/rpy/quaternion, axis, limits, movable-parent index (−1 = wrist), depth, leaf flag and LEAP motor ID.
   - *Graph matrices:* shortest-path, ancestor and descendant, computed by `Hand.graph()` with the wrist as node 0. GET-Zero's sibling-edge convention is a flag.
7. **Outputs per hand:** `hand.json`, `hand.xml` (collision MJCF), `hand.urdf` and `template.npz`. The URDF's kinematics are exact; its visuals approximate the stretch.
8. **QA:**
   - Tests: FK matches MuJoCo to 1e-8, locking keeps finger shape, deformations never fold, the template is deterministic, the gate passes.
   - Contact-sheet renders (`python -m kite_data.qa`).
   - Per-hand gate statistics in `index.csv`.

### 4.3 Stage B: splits (implemented: `kite_data/splits.py`, `data/splits.json`)

Each topology goes into exactly one bucket, assigned in the order below. Counts are for all 243 topologies.

| Bucket | Definition | Topologies | Geometry |
| --- | --- | --- | --- |
| Finger subset | Exact finger sets {thumb, ring} and {thumb, index, ring} | 36 | Nominal only |
| Combined | The same topologies as the finger-subset bucket | (36) | New random draws |
| Joint count | Every remaining hand with a 3-joint index finger | 54 | Nominal plus new draws |
| Validation | 8 random topologies from the rest, used for model selection only | 8 | Nominal plus draws |
| Train | Everything else | 145 | Nominal plus 6 draws: 1,015 hand instances |
| Geometry | 10 training topologies with new draws (interpolation), plus GET-Zero's +1.5 cm parts where the topology is in our family (extrapolation) | (10) | New draws |
| GET-Zero hands | GET-Zero's link-removal hands: a different way of building variants | external | As built by GET-Zero |

Training still contains 3-joint middle and ring fingers, and the thumb and ring finger together inside larger hands. Each test therefore isolates exactly one new factor.

### 4.4 Stage C: sampling examples (implemented: `kite_data/sample.py`, `kite_data/generate.py`)

For each hand instance:

1. **q_contact.** Draw finger joints uniformly within limits and reject self-collisions using the box model. Draw the wrist translation from U([−0.4, 0.4]³) m and its rotation uniformly on SO(3).
2. **Contacts.** Draw K ~ U{2, …, 6}. For each contact, choose a non-empty region uniformly, then a template point uniformly within that region, with replacement.
3. **World-frame intent.** c = FK of the selected points and normals at q_contact, expressed in the world frame.
4. **Mask.** m_j = 1 if joint j is the joint of, or an ancestor of, a link carrying a selected point. Palm points select only the wrist, and the wrist's mask is always 1.
5. **q_init.** Joints get N(0, (0.05·range)²) noise and are clamped to their limits. Translation gets N(0, 0.005² I). Rotation becomes R·Exp(ω) with ω ~ N(0, (5°)² I), pending the convention question to the KITE authors.
6. **Label.** q_goal = m ⊙ q_contact + (1 − m) ⊙ q_init. The wrist takes q_contact's pose.
7. **Invariants**, checked on every batch:
   - FK(q_goal) reproduces c at the selected points to within 1e-6;
   - all joints are within their limits;
   - the mask matches the ancestor relation;
   - q_contact is collision-free;
   - the same seed produces byte-identical output.

There are two regimes. `kite` is the default, used for the main results. `wide` is an ablation in which about 20% of examples get σ_t = 5 cm and σ_R = 30°. Labels are defined the same way in both.

Each example stores the following; the README has the full key table.
- K, and template indices padded to 6.
- Contacts (x, n) padded to 6.
- q_contact, q_init and q_goal: wrist as position plus quaternion; joints in canonical order.
- The mask, and whether the wide regime was used.

Files are per hand, so joints aren't padded; the loader pads to 16. z is not stored: compute it from the contacts once a ψ exists.

### 4.5 Stage D: volume, storage and compute

- **Volume.**
  - *Train:* 1,015 hands × 1,000 examples ≈ 1M (the overview's figure). Raise the per-hand count if the GPU budget allows; run a scaling test early (1k, 4k and 16k per hand on a few hands).
  - *Validation:* 1,000 examples per hand.
  - *Test:* 2,000 examples per hand.
  - *Upper bound:* if one is trained, separate training data for every test hand, with the same per-hand budget as training and seeds disjoint from its test data.
- **Storage (measured).** There is one `.npz` per hand per split, and hand descriptors are stored once in `data/hands/`. All splits together are 2.41M examples and take 850 MB. The 1M training examples load in about 5 s and use about 0.9 GB of memory.
- **Compute (measured, 4 cores).** Building all 1,741 hands takes about 4 minutes, and generating every split about another 4. Box collision checks cost about 0.3 ms each, at 19% acceptance for the full hand and up to 100% for small hands. That is well under the overview's "few CPU hours".
- **Streaming (optional, not implemented).** Keep a per-hand pool of about 50k collision-free finger configurations, and draw wrist poses, contacts and noise on the fly in the dataloader. That gives effectively unlimited training data, while the static files remain the reference for validation and test.

### 4.6 Repository layout

```
kite_data/    config.py  leap.py (base model)  topology.py  geometry.py  hand.py  collision.py
              template.py  urdf.py  build_hands.py  qa.py         (done: Stage A)
              splits.py  sample.py  generate.py  dataset.py       (done: Stages B-C)
scripts/      fetch_assets.sh
tests/        test_topology.py  test_geometry.py  test_hand.py  test_collision_and_template.py
              test_splits.py  test_sample.py
third_party/  pinned GET-Zero and Menagerie assets (git-ignored)
data/         generated outputs and caches (git-ignored)
reference/    paper links and fetch script
```

### 4.7 Milestones

1. ~~**Assets and FK:** fetch script, topology selection, hand descriptor and the FK parity test.~~ Done.
2. ~~**Geometry and collision:** the geometry variation, box transplant and collision validation.~~ Done.
3. ~~**Templates:** surface templates with visibility filtering, plus renders.~~ Done.
4. ~~**v0 dataset:** the sampler and its invariants on about 10 hands, with stats and renders for the team to review.~~ Done.
5. **v1 dataset:** ~~splits, full generation~~ done. Still to do: a QA report, and importing GET-Zero's hands as the extra test set.
6. **z:** add it once a ψ exists, whether KITE's or our own. Computing it at load time may be enough, with no stored cache.
7. **Check against KITE's code:** compare template, collision and noise conventions when the code arrives, then regenerate if needed.

The README's "What's left" section keeps the current list of remaining work.

### 4.8 Risks

- **ψ never ships.** The dataset is unaffected; only the z cache changes.
- **Template quality.** This is the most LEAP-specific risk. The mitigation is visibility filtering plus visual QA.
- **Box collisions miss about 1.5% of collisions.** Re-check a 5% audit sample of accepted configurations with exact meshes and report the miss rate.
- **The thumb mount never shifts.** Its motor housing can't be separated from the palm mesh. Index, middle and ring mounts shift, and their housings move with them.
- **KITE's code may use different conventions.** This could affect template sampling, collision pairs, the rotation-noise convention or the world origin. The data is cheap to regenerate (about 8 minutes), so the fix is to align the config and rerun.
- **The decoder may need KITE-style surface tokens to generalize across geometry.** The template and the 32-point FPS indices are already stored.
