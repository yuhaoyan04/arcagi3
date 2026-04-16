# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

`wm_exp` is a research experiment built on top of LeWorldModel (LeWM). It contains the original LeWM code plus a new spherical world model variant (SWM). The full experimental plan is in [`plan_v2.md`](plan_v2.md); the experiment-specific README is [`README_SWM.md`](README_SWM.md).

Dependencies: `stable-pretraining`, `stable-worldmodel`. Training uses PyTorch Lightning + Hydra. No separate build step.

## Working Rules

1. Prefer minimal changes first.

2. Prefer comprehensive minimal fixes over narrowly local ones.

3. Keep options simple unless the user asks for more.

4. Do not invent details when the repo already has an answer.

5. If something is underspecified, use the existing code as the default source of truth.

6. If assumptions still remain, keep them small and say them explicitly.

7. Update `AGENTS.md` only when a new rule is necessary and keep the rule general.

8. Add visibility when an operation may take noticeable time.

9. Go ahead without asking for small, local fixes.

10. For a larger patch, present a short plan and ask for permission before proceeding.

11. Review your changes against these working rules before finishing; if you find a small issue, fix it.

12. Run `ruff format` after edits.

## Experiment: Spherical World Model (SWM) — V0

**Research question:** Can replacing LeWM's Euclidean representations + SIGReg with spherical representations + a uniformity-oriented anti-collapse loss improve performance, particularly on Two-Room (where LeWM scores 87% vs. 100% for simpler baselines)?

**Core hypothesis:** SIGReg forces embeddings toward an isotropic Gaussian, over-constraining low-intrinsic-dimension environments. A spherical geometry may better preserve the state-space topology.

**Three-stage ladder** (each stage only runs if the previous succeeds):
- **V0 (implemented):** Spherical encoder/predictor + cosine pred loss + configurable spread/uniformity regularizer
- **V1:** Add vMF parameterisation (per-observation concentration κ) for adaptive resolution
- **V2:** Add a learnable ball-cap constraint for OOD detection

## Codebase Structure

| File | Role |
|---|---|
| `jepa.py` | `JEPA` (LeWM baseline) + `SphericalJEPA` (V0, subclasses JEPA) |
| `module.py` | Shared architecture modules + `cosine_pred_loss()`, `spread_loss()`, `uniformity_loss()` |
| `train.py` | LeWM training entry point (`python train.py data=tworoom`) |
| `train_swm.py` | SWM training entry point (`python train_swm.py data=tworoom`) |
| `eval.py` | Shared evaluation entry point (works for both LeWM and SWM) |
| `config/train/lewm.yaml` | LeWM training config |
| `config/train/swm.yaml` | SWM training config (defaults to Two-Room, embed_dim=64) |

## Key Design Decisions

- `SphericalJEPA` overrides only `encode()`, `predict()`, `criterion()` — `rollout()` and `get_cost()` are inherited unchanged, so the CEM planner needs no modification.
- SWM projectors use `BatchNorm1d` before the final L2 normalisation.
- Both `spread_loss` and `uniformity_loss` operate on all B×T tokens together (not split by context/target).
- The default SWM anti-collapse regularizer is `loss.regularizer.type=uniformity`, with shared weight `loss.regularizer.weight`.

## Running Experiments

```bash
# Train SWM on Two-Room (primary benchmark)
python train_swm.py data=tworoom

# Train LeWM baseline (for comparison)
python train.py data=tworoom

# Evaluate (both models use the same eval.py)
python eval.py --config-name=tworoom.yaml policy=<subdir>/swm
python eval.py --config-name=tworoom.yaml policy=<subdir>/lewm
```

Run 3–5 seeds per method. Report mean ± std success rate on Two-Room.
