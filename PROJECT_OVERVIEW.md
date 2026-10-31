# GET Your KITE — Project Overview

## Problem

KITE can move task knowledge to a new robot hand, but only after training a new decoder for that hand. We want one decoder that works on hands it has never seen, with no per-hand training.

**Why this is hard.** Robots differ in their number of fingers, joints, link lengths and joint limits. The same task can need completely different motions on different bodies, so a policy learned on one robot usually cannot drive another. Collecting demonstrations and retraining for every new robot is expensive.

**What KITE does.** KITE splits manipulation into two parts joined by a shared message:

- A task policy decides *what* interaction should happen and outputs an intent vector z (where to make contact, and in which direction).

- An action decoder decides *how this body* achieves that intent and outputs the robot's next configuration.

The decoder is trained only from the robot's kinematic model, with no task demonstrations. But each body still needs its own output layer, sized to its number of joints, and must be included in training. A new hand means retraining.

**What GET-Zero does.** GET-Zero controls unseen hand variants with a single set of weights. Each joint is a transformer token, and the hand's kinematic tree shapes the attention. However, it learns a full task policy from RL experts and has no reusable notion of task intent.

**The gap.** KITE transfers *what to do* across bodies; GET-Zero transfers *how to move* across related bodies. **We investigate whether KITE’s frozen task-intent interface can be combined with a morphology-conditioned decoder trained solely on synthetic kinematic data.**

**Problem statement.** Given:

- KITE's intent z, from the frozen KITE policy,

- the current configuration q = (6D wrist pose, finger joint angles),

- the hand's URDF (joints, axes, offsets, limits, and the kinematic tree),

output the next configuration (6D wrist pose, finger joint angles) for a hand never seen in training, with no training on that hand. A classical IK solver turns the wrist pose into arm joint angles, and a position controller executes everything.

## Hypothesis

A single graph-conditioned decoder, trained on kinematic data from a diverse family of hands, can turn KITE's intents into correct actions on hands it has never seen, approaching the accuracy of KITE's per-hand decoders.

What we expect to see:

- **Strongest** on held-out hands with new finger lengths and spacing, within the training range.

- **Moderate drop** on held-out hands with an unseen combination of fingers or an unseen joint count per finger.

- **Closed-loop better than open-loop**, because repeated feedback corrects small errors.

- **Arm-agnostic by construction**: the decoder only outputs a wrist pose, so swapping the arm changes only the IK step.

If geometry transfer works but new finger counts fail, that is still a useful result: it shows where interpolation ends and structural generalization begins.

## Method

We keep all of KITE except its per-hand decoders, which we replace with one Graph Embodiment Transformer (GET) decoder. Only this decoder is trained.

### System setup

- **Frozen KITE parts:** the task policy π and the contact encoder ψ. Freezing both keeps the meaning of z unchanged.

- **Arm:** one 6-DoF arm. A classical IK solver converts the decoder's wrist pose into arm joint angles. The decoder never sees the arm.

- **Controller:** MuJoCo position control (a PD loop per joint) tracks the wrist-driven arm angles and the finger angles. No learning.

### Notation

| **Symbol** | **Meaning** |
| --- | --- |
| e | One hand (embodiment) with N finger joints |
| w | 6D wrist pose |
| θ | Finger joint angles, one per joint (θ ∈ R^N) |
| q = (w, θ) | The hand's full configuration |
| ρ\_e = {ρ\_j} | Per-joint URDF features: type, axis, offset from parent, limits, leaf flag |
| 𝒢\_e | The URDF tree as distance matrices: shortest-path, ancestor, descendant |
| z | KITE's intent vector |
| G\_φ | Our GET decoder, with trainable weights φ |

### Inputs and output

The decoder maps an intent, the current state and the hand description to the next configuration:

q̂<sub>t+1</sub> = (ŵ<sub>t+1</sub>, θ̂<sub>t+1</sub>) = G<sub>φ</sub>(z<sub>t</sub>, q<sub>t</sub>, ρ<sub>e</sub>, 𝒢<sub>e</sub>)

| **Input** | **Meaning** | **In training** | **At inference** |
| --- | --- | --- | --- |
| z\_t | What contact should happen | ψ(c), from FK-generated contacts | π(o\_t, z\_{t−1}), from the scene |
| q\_t = (w\_t, θ\_t) | Current state of the hand | q\_init, a perturbed sample | Read from the robot's sensors |
| ρ\_e, 𝒢\_e | Which hand this is | Parsed from the URDF | Parsed from the URDF |

The target q\_goal appears only in training, as the label. At inference it is exactly what the decoder predicts.

### Inside the decoder

**Tokens.** One wrist token, plus one token per finger joint j = 1, …, N:

h<sub>0</sub><sup>(0)</sup> = MLP<sub>wrist</sub>([z<sub>t</sub>, w<sub>t</sub>])

h<sub>j</sub><sup>(0)</sup> = MLP<sub>in</sub>([z<sub>t</sub>, θ̃<sub>t,j</sub>, ρ<sub>j</sub>, p<sub>j</sub>(q<sub>t</sub>)])

Here θ̃ is the angle normalized to [−1, 1] by its limits, and p\_j(q\_t) is the world pose of joint j's link, computed by forward kinematics (FK) at the current state. KITE's contacts are in world coordinates, so each token needs to know where it is.

**Graph-biased attention**, in each of L layers:

A<sub>ij</sub> = (Q<sub>i</sub><sup>T</sup>K<sub>j</sub>) / √d + b<sub>spd</sub>[D<sub>ij</sub><sup>spd</sup>]

          + b<sub>anc</sub>[D<sub>ij</sub><sup>anc</sup>] + b<sub>desc</sub>[D<sub>ij</sub><sup>desc</sup>]

The b terms are small learned lookup tables, so how joints influence each other depends on how they are connected in the tree.

**Outputs**, each predicted as a change from the current state:

ŵ<sub>t+1</sub> = w<sub>t</sub> ⊕ MLP<sub>wrist</sub><sup>out</sup>(h<sub>0</sub><sup>(L)</sup>)

θ̂<sub>t+1,j</sub> = θ<sub>t,j</sub> + r<sub>j</sub> MLP<sub>joint</sub>(h<sub>j</sub><sup>(L)</sup>)

p̂<sub>j</sub> = MLP<sub>fk</sub>(h<sub>j</sub><sup>(L)</sup>)

Here ⊕ applies a small pose change, r\_j is joint j's range, and p̂\_j is the self-model's guess of where joint j ends up. Because MLP\_joint is shared by every finger token, the number of outputs always equals N. This is what removes per-hand training.

### Training data

All data is generated forward from each hand's URDF, with no IK, demonstrations or RL. For each example:

q\* = (w\*, θ\*),  w\* ∈ arm workspace,

θ\* ∼ 𝒰(joint limits)    [sample a configuration]

c = {(x<sub>i</sub>, n<sub>i</sub>)}<sub>i ∈ S</sub> = FK<sub>𝒯</sub>(q\*)<sub>S</sub>,  |S| = K ∈ {2, …, 6}

[place K surface points in the world]

z = ψ(c)    [encode with the frozen encoder]

q<sub>init</sub> = q\* + ε,  ε ∼ 𝒩(0, σ<sup>2</sup>)    [starting (current) state]

q<sub>goal</sub> = m ⊙ q\* + (1 − m) ⊙ q<sub>init</sub>    [label]

The mask m is KITE's: m\_j = 1 if joint j moves a link carrying a selected point, else 0. Joints that don't affect the contacts are told to stay where they are. The wrist moves every point, so its mask is always 1.

Each training example is therefore:

(z, q<sub>init</sub>, ρ<sub>e</sub>, 𝒢<sub>e</sub>) → q<sub>goal</sub>

We generate roughly 0.5–1M examples across the training hands.

### Loss

min<sub>φ</sub>  𝔼<sub>e, (z, q\_init, q\_goal)</sub> [λ<sub>a</sub> ℒ<sub>action</sub> + λ<sub>c</sub> ℒ<sub>contact</sub>

                                      + λ<sub>fk</sub> ℒ<sub>fk</sub> + λ<sub>lim</sub> ℒ<sub>limit</sub>]

**Action:** match the label (KITE's loss). The wrist and fingers are averaged separately so many finger joints don't drown out the wrist.

ℒ<sub>action</sub> = ‖ŵ ⊖ w<sub>goal</sub>‖<sup>2</sup> + (1/N) ∑<sub>j=1</sub><sup>N</sup> (θ̂<sub>j</sub> − θ<sub>goal,j</sub>)<sup>2</sup>

**Contact:** run differentiable FK on the prediction and check that the selected points land on the intended contacts. This accepts any configuration that makes the contact, not only the labeled one.

ℒ<sub>contact</sub> = (1/K) ∑<sub>i ∈ S</sub> [‖FK<sub>x</sub>(q̂)<sub>i</sub> − x<sub>i</sub>‖<sup>2</sup>

                         + λ<sub>n</sub>(1 − FK<sub>n</sub>(q̂)<sub>i</sub><sup>T</sup> n<sub>i</sub>)]

**Self-model:** predict where each joint ends up.

ℒ<sub>fk</sub> = (1/N) ∑<sub>j=1</sub><sup>N</sup> ‖p̂<sub>j</sub> − p<sub>j</sub>(q<sub>goal</sub>)‖<sup>2</sup>

**Limits:** penalize angles outside [l\_j, u\_j].

ℒ<sub>limit</sub> = ∑<sub>j=1</sub><sup>N</sup> [ReLU(θ̂<sub>j</sub> − u<sub>j</sub>)<sup>2</sup> + ReLU(l<sub>j</sub> − θ̂<sub>j</sub>)<sup>2</sup>]

### At inference

Each control step runs:

z<sub>t</sub> = π(o<sub>t</sub>, z<sub>t−1</sub>)

→ (ŵ<sub>t+1</sub>, θ̂<sub>t+1</sub>) = G<sub>φ</sub>(z<sub>t</sub>, q<sub>t</sub>, ρ<sub>e</sub>, 𝒢<sub>e</sub>)

→ q<sub>t+1</sub><sup>arm</sup> = IK(ŵ<sub>t+1</sub>)

The position controller then tracks the arm angles and finger angles, the robot moves, and the next step reads the new observation o\_{t+1} and state q\_{t+1}.

## Evaluation

We test on hands held out from training entirely, and compare against KITE's own decoder trained on each test hand.

### Hand family

All hands are variants of the LEAP Hand, the family GET-Zero used. The base hand has four fingers (index, middle, ring, thumb) with four joints each, 16 DoF in total. Variants come from removing fingers or joints and changing link geometry:

| **Variation** | **Values** | **Example** |
| --- | --- | --- |
| Fingers present | 2, 3 or 4 of index, middle, ring, thumb | Thumb + index only |
| Joints per finger | 2, 3 or 4 | Index finger with 3 joints |
| Link lengths | 0.8–1.2× per link | Longer middle finger |
| Finger attachment | Small shifts in where fingers mount on the palm | Wider finger spacing |

Most variants keep the thumb, since many contacts need an opposing finger. Every example from a held-out hand stays out of training.

### Test splits

Each split holds out one kind of change, so we can tell what generalizes:

| **Split** | **What is new** | **Tests** | **Difficulty** |
| --- | --- | --- | --- |
| Geometry | Unseen link lengths and finger attachment, on seen finger and joint layouts | Interpolation | Easiest |
| Finger subset | Fingers never seen together, e.g. thumb + ring | Composition | Medium |
| Joint count | A held-out per-finger joint count, e.g. no 3-joint index finger in training | Structural interpolation | Medium |
| Combined | An unseen finger subset with unseen link lengths | Both at once | Hardest |

### Metrics

| **Metric** | **Unit** | **Meaning** |
| --- | --- | --- |
| Contact position error | mm | Distance between where the selected surface points land and the intended contacts |
| Contact normal error | degrees | Angle between achieved and intended contact directions |
| Success rate | % | Share of intents realized within a set tolerance (e.g. 1 cm, 15°) |
| Joint-limit violations | % | Predictions outside joint limits |
| Self-collisions | % | Predictions where the hand or arm collides with itself |
| IK failure rate | % | Wrist poses the arm cannot reach |

### Evaluation modes

- **Open-loop:** set the predicted configuration directly and measure contact error with FK. This isolates decoder accuracy.

- **Oracle intent:** feed ground-truth intents from KITE source demonstrations to the decoder, as KITE does, to isolate the decoder from the policy.

- **Closed-loop:** frozen KITE policy → our decoder → IK → position controller in MuJoCo, on simple contact tasks (reaching and touching, button or key pressing).

### Comparison

- **Reference:** KITE with a decoder trained on each test hand. This is the upper bound we aim to approach.

- **Ours:** one GET decoder, never trained on the test hand.

- **Optional:** pair a held-out hand with a different arm (e.g. 7-DoF) to show that only IK changes.

We only test intents the hand can physically achieve, so an impossible request is not counted as a decoder failure.

## Scope, assumptions and open questions

The claim is zero-shot decoder transfer across variants of the LEAP Hand on a fixed arm, in simulation.

**Assumptions**

- KITE's policy and contact encoder stay frozen, so z keeps its meaning.

- Every hand has a URDF; the decoder uses only its kinematic part (joints, offsets, limits, tree), not mass or friction.

- Kinematics only: we learn where to place contacts, not how hard to press. Forces and dynamics are left to the controller.

- Requested contacts are reachable by the test hand.

**Out of scope for now**

- Learning the arm's joints (handled by IK).

- Prismatic joints, mobile bases, two-handed robots.

- Torque or force prediction, and real-robot deployment.

**Open questions to resolve before building**

- Are KITE's code and a pretrained contact encoder available? If not, feed raw contact points and normals to the decoder instead of z (a standalone decoder study).

- Does KITE's latent space cover new hands well? Its encoder was trained jointly with its original five embodiments.

- Can we reuse GET-Zero's code for generating LEAP Hand variants, and which variations does it already cover?

- Which IK solver, and how do we keep sampled wrist poses inside the arm's workspace?

- What tolerance defines success (e.g. 1 cm and 15°)?

**Rough compute:** a few CPU hours for data generation, about 2–6 GPU hours to train the decoder, and a few more for simulation evaluation, on an 8–16 GB GPU.

## References

- [KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation](https://arxiv.org/abs/2606.22113), Wang and Fang, 2026

- [GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment Generalization](https://arxiv.org/abs/2407.15002), Patel and Song, ICRA 2025
