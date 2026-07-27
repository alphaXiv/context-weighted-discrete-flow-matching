# Context-weighted Discrete Flow Matching — QM9 reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/context-weighted-discrete-flow-matching/blob/main/notebooks/qm9_reproduction.py)

This repository reproduces the molecular claims of [Context-weighted Discrete Flow Matching (arXiv:2607.21427)](https://arxiv.org/abs/2607.21427) on canonical QM9 SMILES. We trained compact masked discrete-flow transformers with standard cross-entropy (CE) and scaled cross-entropy (SCE), then compared Euler, neighbor-weighted, and entropy-weighted sampling at matched NFE.

**Assessment: partially reproduced.** Neighbor weighting improved CE sampling from NFE 64 upward, but not at very low NFE: at NFE 128 it produced **483.5 valid / 391.0 novel** molecules versus Euler’s **451.5 / 373.5** (four seeds, 1,024 samples each). That is **1.07× / 1.05×**, smaller than the paper’s up-to-**2.8× / 1.9×** molecular gain. SCE did not transfer to the compact substitute: the best radius produced **419.2 / 345.8** versus CE’s **480.2 / 390.5** at NFE 256, whereas the paper’s masked-source table reports SCE improvements of 134.8→177.2 valid and 114.8→137.8 novel.

The main substitutions are a 14.4M-parameter transformer instead of 92M, batch 512 instead of 2,048, a 110k-molecule training split, and 1,024 samples per seed rather than five 1,024-sample folds from 5,120 generations. Training length (25k steps), sequence length (32), masked source, canonical QM9, and the reported validity/novelty outcomes were retained.

- [Detailed illustrated report](reports/qm9/report.md)
- [Self-contained tutorial notebook](notebooks/qm9_reproduction.py)
- [Factorial results](reports/qm9/data/factorial_results.csv) and [robustness results](reports/qm9/data/robustness_results.csv)
- Molab: https://molab.marimo.io/github/alphaXiv/context-weighted-discrete-flow-matching/blob/main/notebooks/qm9_reproduction.py

## Experiment log

Every formal run used the exact command `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, figures, and runnable implementation | Not run as an experiment (publication surface) | Presentation-only | — |
| [CE seeds 0–3](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/ce-seed-0-final) ([s1](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/ce-seed-1), [s2](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/ce-seed-2), [s3](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/ce-seed-3)) | Standard conditional-matching baseline; all samplers, NFE 8–256 | `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py` | Neighbor improves validity at NFE ≥64; entropy wins at 256 | Kubernetes, 1 GPU/run |
| [SCE r=1 seeds 0–3](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/sce-r1-seed-0) | Closest local radius | `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py` | Best SCE radius, but below CE | Kubernetes, 1 GPU/run |
| [SCE r=2 seeds 0–3](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/sce-r2-seed-0) | Radius ablation | `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py` | Lowest SCE validity/novelty | Kubernetes, 1 GPU/run |
| [SCE r=3 seeds 0–3](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/sce-r3-seed-0) | Radius ablation | `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py` | Between r=1 and r=2; below CE | Kubernetes, 1 GPU/run |
| [Temperature 0.8](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/temp-0-8), [temperature 1.2](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/ce-seed-0-temperature-1-2), [scale 2](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/neighbor-scale-2), [scale 6](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/tree/orx/neighbor-scale-6) | Temperature and neighbor-scale robustness | `python -m pip install --quiet rdkit==2025.3.6 && python run_reproduction.py` | High-NFE crossover persists; scale 2–6 gives similar ordering | Kubernetes, 1 GPU/run |

All experiments ran on Kubernetes with **NVIDIA RTX PRO 6000 Blackwell** GPUs, a **peak of 16 concurrent GPUs**, and **0.51 hours actual elapsed wall time** from the first Kubernetes launch through the last completed robustness run.

## Reproduce

The formal command is shown above. It downloads the public QM9 SDF from DeepChem, canonicalizes with RDKit, trains for 25k steps, and prints all evidence as `ORX_RESULT` JSON records. To regenerate the article figures from the committed extracts:

```bash
uv run --with matplotlib==3.10.3 --with numpy==2.3.1 python analysis/make_figures.py
```

To explore the evidence locally:

```bash
uvx marimo edit notebooks/qm9_reproduction.py
```
