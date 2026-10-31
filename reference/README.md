# References

The PDFs are not stored in this repo. Run `./fetch_papers.sh` to download them into this folder; PDFs here are git-ignored.

| Paper | Links | Most relevant for the dataset |
| --- | --- | --- |
| Q. Wang, S. S. Jin, K. Fang. *KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation.* arXiv:2606.22113v2, 2026. | [arXiv](https://arxiv.org/abs/2606.22113) · [project page](https://kite-manip.github.io/) (code "soon") | §IV-B: decoder learning and synthetic data. App. A: the sampling recipe (template, regions, K, noise, mask). App. B: encoder/decoder architecture and training. |
| A. Patel, S. Song. *GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment Generalization.* ICRA 2025, arXiv:2407.15002v2. | [arXiv](https://arxiv.org/abs/2407.15002) · [project page](https://get-zero-paper.github.io) · [code](https://github.com/real-stanford/get_zero) | §III: graph attention biases and tokens. §IV-A: procedural LEAP variants. §V: splits. Code: `get_zero/rl/scripts/gen_leap_assets.py`, `get_zero/distill/utils/embodiment_util.py`. |

Related asset: [MuJoCo Menagerie LEAP hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/leap_hand) (MIT). The data plan uses its box collision primitives; see [../DATA_GENERATION_PLAN.md](../DATA_GENERATION_PLAN.md).
