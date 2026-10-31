# Training Data Generation Plan

Status: proposal, 2026-10-31.

This plan covers the synthetic kinematic data for the GET decoder in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (section "Training data"). It follows KITE's own recipe wherever KITE specifies one, so that the data stays usable with a pretrained KITE intent encoder ψ if one becomes available, and so that the KITE reference decoders can be trained on exactly the same data.

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

**Where this changes PROJECT_OVERVIEW.md** (not edited yet):

- **ψ and π are not released.** The "frozen KITE parts" do not exist for us yet, so the oracle-intent and closed-loop evaluation modes are blocked on the release (fallbacks in section 3).
- **KITE details are off.** KITE has three authors (Wang, Jin, Fang) and six simulated robot embodiments, not five.
- **The wrist needs no arm workspace.** KITE samples it as a floating base anywhere in a 0.8 m cube, at any orientation. That answers the workspace open question: the arm only matters at deployment.
- **The volume and compute estimates are low.** KITE uses 10^6 examples per embodiment. The overview's 0.5–1M total is far less per hand, and its 2–6 GPU-hour training estimate is far below KITE's.
- **The variant table needs new code.** "Joints per finger 2–4" has to mean GET-Zero's link removal, unless we write a different generator. Continuous 0.8–1.2× lengths and attachment shifts do not exist in GET-Zero, so we have to build them.

## 2. Measured facts about the LEAP assets

These were checked on 2026-10-31 with MuJoCo 3.17, trimesh and python-fcl, on GET-Zero's URDFs and STLs. The scripts are not committed.

- **FK conventions are unambiguous.** A short numpy URDF FK (fixed-axis XYZ rpy) matches MuJoCo's URDF import to 1e-16 m on hands 001, 013, 139 and 600.
- **Topology counts.** The overview's family, where every present finger has 2–4 joints, contains 126 topologies with a thumb and 54 without. Of the thumb-present ones, 32 have a 3-joint index finger.
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

1. If ψ isn't available in time, there are two options. (a) Re-implement ψ and train it jointly with our decoder. That gives us our own intent space but no KITE policy, so closed-loop evaluation needs scripted intent sequences. (b) Use the raw-contact fallback. The dataset below supports both.
2. Should we include thumbless hands (54 more topologies)? Should GET-Zero's 1-joint fingers serve as an extra-hard test?
3. What length range? GET-Zero's +1.5 cm parts correspond to scale factors of about 1.2–1.4, beyond 0.8–1.2. Recommendation: keep 0.8–1.2 for training and use those parts as an extrapolation test.
4. Which split definitions (section 4.3)? Recommendation: exact-subset holdout.
5. What data volume and GPU budget (section 4.5)?
6. Which noise regime? KITE's is the default and is required for the baseline. A wide-noise mixture could be added as an ablation.
7. Should the contact loss be tied to the labelled points, as in the overview, or be assignment-free, scoring each target against its nearest template point? Both need the template, which we store either way.

## 4. Plan

### 4.1 Principles

- **Match KITE wherever it is specific:** sampling, noise, mask, K, regions and the 4,096-point template.
- **Store raw values; derive the rest in code.** Store contacts, template indices and configurations. Compute z, graph matrices and normalized features from them with versioned code, so a change of convention never forces regeneration.
- **Split by hand, before sampling.** Nothing from a held-out hand reaches training, and geometry variants of one topology never straddle splits.
- **Make everything deterministic.** Every output follows from (config, seed) and records the generator's git SHA.

### 4.2 Stage A: hand assets

1. **Fetch.** `scripts/fetch_assets.sh` clones GET-Zero (279652d) and Menagerie (b065505) at pinned commits. We use GET-Zero's 236 graph URDFs for topologies and its meshes and joint limits. If thumbless hands are wanted, they come from deleting the thumb chain.
2. **Select topologies.** Use the 126 thumb-present topologies (180 if thumbless hands are included). Give each a readable ID such as `T4-I3-M0-R2` (joints per thumb, index, middle, ring) and keep its GET-Zero ID alongside.
3. **Vary geometry.** Each hand instance is one topology plus one geometry draw.
   - *Link lengths:* draw s ~ U[0.8, 1.2] independently for each of the five links GET-Zero lengthens (`mcp_joint`, `dip`, `fingertip`, `thumb_dip`, `thumb_fingertip`). Stretch along the link-frame y axis, the axis GET-Zero extends. Confine the stretch to a spacer band chosen once per base mesh, using a piecewise-linear map, so motor housings and fingertip pads keep their shape. The child joint's y offset moves by Δ = (s − 1)·L, and collision boxes translate or stretch with the band.
   - *Attachment:* shift each finger-root joint origin by δ ~ U[−4, 4] mm along the palm's finger-spacing axis; the current spacing is 45.5 mm. Keep these shifts small, because each finger's base motor is part of the palm mesh and does not move with the finger.
   - *Instances per topology:* one nominal instance (s = 1, δ = 0), plus G random draws (G = 6 for training topologies).
   - *Validity gate:* no collision at q = 0 or at the canonical pose, and at least 5% collision-free under uniform sampling. Otherwise redraw.
4. **Build the collision model.** Transplant Menagerie's boxes onto each link. Exclude Menagerie's listed pairs plus all parent–child pairs, and deform the boxes along with the geometry. Validate every hand against the exact mesh check on 2,000 configurations; the gate is at least 93% agreement (95.3% was measured on hand 001).
5. **Build the surface template** (KITE's T^e).
   - Oversample area-weighted points on each link's deformed visual mesh by about 4×.
   - Drop points where fewer than 8 of 32 hemisphere rays escape the link's own mesh. Optionally also drop palm points covered by the finger roots at the canonical pose.
   - Subsample to exactly 4,096 points per hand.
   - Store each point's link index, local position, local outward normal and region (palm, index, middle, ring or thumb).
   - Also store 32 farthest-point-sampled indices, in case the decoder adopts KITE-style surface tokens.
6. **Write the hand descriptor** (`hand.npz` plus `hand.json`).
   - *Joint order:* index, middle, ring, thumb; proximal to distal within each finger; at most 16 joints.
   - *Per joint:* parent joint (−1 means the wrist), finger, depth, origin xyz, origin quaternion (rpy kept for parity with GET-Zero), axis, limits, leaf flag and LEAP motor ID.
   - *Per link:* boxes, mesh and template slice.
   - *Graph matrices:* shortest-path, ancestor and descendant, with the wrist as node 0. These are derived in code, and the adjacency convention (tree edges only, or GET-Zero's sibling edges) is a flag.
7. **Run asset QA.**
   - numpy FK matches MuJoCo FK to within 1e-9 m.
   - The canonical pose is collision-free.
   - Collision agreement meets the gate.
   - Template visibility statistics look sane.
   - A contact-sheet render of every hand is produced.

### 4.3 Stage B: splits (fixed before sampling)

Each topology goes into exactly one bucket, assigned in the order below. Counts are for the 126 thumb-present topologies.

| Bucket | Definition | Topologies | Geometry |
| --- | --- | --- | --- |
| Finger subset | Exact finger sets {thumb, ring} and {thumb, index, ring} | 24 | Nominal only |
| Combined | The same topologies as the finger-subset bucket | (24) | New random draws |
| Joint count | Every remaining hand with a 3-joint index finger | 26 | Nominal plus new draws |
| Validation | 8 random topologies from the rest, used for model selection only | 8 | Nominal plus draws |
| Train | Everything else | 68 | Nominal plus 6 draws: 476 hand instances |
| Geometry | 10 training topologies with new draws (interpolation), plus GET-Zero's +1.5 cm parts where the topology is in our family (extrapolation) | (10) | New draws |

Training still contains 3-joint middle and ring fingers, and the thumb and ring finger together inside larger hands. Each test therefore isolates exactly one new factor.

### 4.4 Stage C: sampling examples (KITE's recipe)

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

There are two regimes. `kite` is the default, used for the baseline and the main results. `wide` is an ablation in which about 20% of examples get σ_t = 5 cm and σ_R = 30°. Labels are defined the same way in both.

Each example stores: hand ID, K, template indices (padded to 6), contacts (x, n) padded to 6, q_contact, q_init and q_goal (wrist as position plus quaternion; joints in canonical order, padded to 16), the mask, the regime and the seed. z lives in a separate, optional cache keyed by ψ version.

### 4.5 Stage D: volume, storage and compute

- **Volume.**
  - *Train:* 476 hands × 8,000 examples ≈ 3.8M. Raise the per-hand count if the GPU budget allows.
  - *Validation:* 1,000 examples per hand.
  - *Test:* 2,000 examples per hand.
  - *KITE baseline:* separate training data for every test hand, with the same per-hand budget as training and seeds disjoint from its test data.
- **Storage.** An example is about 130 numbers, roughly 0.5 KB as float32, so ~4M examples take about 2 GB. Write sharded HDF5 per split (or `.npz` shards of 100k), with hand descriptors stored once.
- **Compute.** With box collisions (0.34 ms per check, 20–60% acceptance), each accepted configuration costs about 1–2 ms, so ~4M examples take roughly 1–2 CPU-hours. Templates take about a minute per base mesh, built once. The total is a few CPU-hours, consistent with the overview.
- **Streaming (optional).** Keep a per-hand pool of about 50k collision-free finger configurations, and draw wrist poses, contacts and noise on the fly in the dataloader. That gives effectively unlimited training data, while the static shards remain the reference for validation and test.

### 4.6 Repository layout

```
configs/      family.yaml  splits.yaml  sampling.yaml
kite_data/    assets/ (fetch, topology, geometry, collision, template)
              hand.py (descriptor, numpy and torch FK)  sample.py  splits.py  writer.py  qa.py
scripts/      fetch_assets.sh  build_hands.py  make_splits.py  generate.py  qa_report.py
tests/        test_fk_parity.py  test_invariants.py  test_mask.py  test_determinism.py
data/         generated outputs (git-ignored)
reference/    papers and notes
```

### 4.7 Milestones

1. **Assets and FK:** fetch script, topology selection, hand descriptor and the FK parity test.
2. **Geometry and collision:** the geometry variation, box transplant and collision validation.
3. **Templates:** surface templates with visibility filtering, plus renders.
4. **v0 dataset:** the sampler and its invariants on about 10 hands, with stats and renders for the team to review.
5. **v1 dataset:** splits, full generation and a QA report.
6. **z cache:** added once a ψ exists, whether KITE's or our own.

### 4.8 Risks

- **ψ never ships.** The dataset is unaffected; only the z cache changes.
- **Template quality.** This is the most LEAP-specific risk. The mitigation is visibility filtering plus visual QA.
- **Box collisions miss about 1.5% of collisions.** Re-check a 5% audit sample of accepted configurations with exact meshes and report the miss rate.
- **Attachment shifts move the finger but not its base motor.** Keep shifts small and state this limitation.
- **The decoder may need KITE-style surface tokens to generalize across geometry.** The template and the 32-point FPS indices are already stored.
