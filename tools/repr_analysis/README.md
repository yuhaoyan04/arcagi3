# Representation Analysis Guide

This document explains how to use the representation-analysis tools added for
LeWM / SWM experiments.

The goal is not just to ask "did the model avoid collapse?", but to answer:

- Is the latent space geometrically meaningful?
- Does it preserve local state-space structure?
- Does action actually produce meaningful latent branching?
- Is a new loss better than SIGReg, or merely different?

## Files

- [analyze_repr.py](/home/ag/projects/le-wm/tools/repr_analysis/analyze_repr.py)
  Runs quantitative analysis on one checkpoint and one dataset.
- [plot_repr.py](/home/ag/projects/le-wm/tools/repr_analysis/plot_repr.py)
  Plots one PCA / t-SNE projection export.
- [compare_repr.py](/home/ag/projects/le-wm/tools/repr_analysis/compare_repr.py)
  Draws side-by-side comparison plots from two analysis directories.
- [probe.py](/home/ag/projects/le-wm/probe.py)
  Older lightweight probing script. Still useful, but less systematic.

## What To Use When

Use `analyze_repr.py` when:
- you want quantitative diagnostics before changing losses
- you want to inspect one model on one environment
- you want JSON outputs for later plotting or comparison

Use `plot_repr.py` when:
- you already ran `analyze_repr.py`
- you want a quick visualization similar to Figure 9 in the paper

Use `compare_repr.py` when:
- you want to compare `SIGReg` vs `BN+uniformity`
- you want to compare two checkpoints on the same environment
- you want matched color scales across both plots

## 1. Run Analysis

Basic usage:

```bash
python tools/repr_analysis/analyze_repr.py \
  --ckpt /path/to/model_object.pt \
  --dataset tworoom \
  --state-key proprio \
  --save-dir /tmp/repr_analysis_tworoom
```

With t-SNE export:

```bash
python tools/repr_analysis/analyze_repr.py \
  --ckpt /path/to/model_object.pt \
  --dataset pusht \
  --state-key proprio \
  --save-dir /tmp/repr_analysis_pusht \
  --export-tsne
```

Useful flags:

- `--ckpt`: object checkpoint path
- `--dataset`: dataset name, e.g. `tworoom`, `pusht`, `ogb`, `dmc`
- `--state-key`: default is `proprio`
- `--n-sequences`: number of sampled sequences, default `128`
- `--max-points`: cap for topology analysis, default `512`
- `--knn-k`: k for neighborhood overlap, default `10`
- `--export-tsne`: export `tsne_projection.json` if `scikit-learn` is installed
- `--tsne-perplexity`: default `30`
- `--save-dir`: output directory for JSON exports

Notes on rigor:
- topology metrics now use a random flat-point subsample instead of taking the first `N` flattened points
- topology also reports `*_cross_seq` metrics to reduce inflation from trivial within-sequence temporal neighbors
- action-effect correlation is computed over all perturbed samples, not just one mean per trial
- interpolation smoothness is averaged over multiple anchors instead of a single context

## 2. Outputs

`analyze_repr.py` prints four analysis blocks:

- `Embedding`
  - anti-collapse / distribution health
  - norms, per-dim std, inter-sample cosine, effective rank
- `Topology`
  - latent distance vs state distance
  - kNN overlap between latent space and state space
- `Dynamics`
  - whether latent motion tracks state motion
  - prediction error and pred-target cosine
- `Action Effect`
  - whether changing actions creates meaningful latent branching
  - whether action interpolation is smooth

It also prints an `Interpretation` section with environment-aware diagnostic hints.

### Saved files

When `--save-dir` is provided:

- `summary.json`
  All scalar metrics and interpretation hints.
- `pca_projection.json`
  2D PCA projection of latent embeddings.
- `tsne_projection.json`
  2D t-SNE projection, only if exported successfully.
- `local_neighbors.json`
  Anchor-by-anchor comparison of nearest neighbors in latent vs state space.

## 3. Visualize One Run

Plot one exported projection:

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_pusht/tsne_projection.json \
  --output /tmp/repr_analysis_pusht/tsne_state0_state1.png \
  --color-dims 0 1 \
  --title "PushT t-SNE"
```

Examples:

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_tworoom/pca_projection.json \
  --output /tmp/repr_analysis_tworoom/pca_state01.png \
  --color-dims 0 1
```

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_pusht/tsne_projection.json \
  --output /tmp/repr_analysis_pusht/tsne_state0.png \
  --color-dims 0
```

## 4. Compare Two Runs

Compare `SIGReg` vs `BN+uniformity`:

```bash
python tools/repr_analysis/compare_repr.py \
  --left-dir /tmp/repr_sigreg \
  --right-dir /tmp/repr_bn_uniform \
  --projection tsne \
  --output /tmp/repr_compare_tsne.png \
  --color-dims 0 1 \
  --left-label SIGReg \
  --right-label BN+uniformity \
  --title "PushT Representation Comparison"
```

Compare PCA instead:

```bash
python tools/repr_analysis/compare_repr.py \
  --left-dir /tmp/repr_sigreg \
  --right-dir /tmp/repr_bn_uniform \
  --projection pca \
  --output /tmp/repr_compare_pca.png \
  --color-dims 0 1 \
  --left-label SIGReg \
  --right-label BN+uniformity
```

## 5. How To Read The Results

### Embedding

Good signs:
- `effective_rank` is not tiny
- `inter_sample_cosine_mean` is not near collapse
- per-dim std is not concentrated near zero

Bad signs:
- rank is very low
- cosine similarity stays very high
- one or a few dimensions dominate

### Topology

Most important metrics:
- `distance_corr`
- `distance_rank_corr`
- `knn_overlap`
- `distance_corr_cross_seq`
- `distance_rank_corr_cross_seq`
- `knn_overlap_cross_seq`

Good signs:
- nearby states remain nearby in latent space
- local neighborhoods are preserved
- cross-sequence metrics stay close to all-pairs metrics, which suggests the result is not mainly coming from trivial temporal adjacency

Bad signs:
- local geometry is scrambled even though spread loss looks good
- the manifold is fragmented into visually distinct islands

Notes:
- `distance_corr` is Pearson correlation on pairwise distances
- `distance_rank_corr` is Spearman-style rank correlation on pairwise distances
- `*_cross_seq` excludes comparisons inside the same sampled sequence window; these are often the more trustworthy metrics when judging planning geometry

### Dynamics

Most important metrics:
- `latent_state_step_corr`
- `pred_error_mean`
- `pred_target_cosine_mean`
- `pred_target_cosine_distance_mean`

Good signs:
- latent motion magnitude tracks true state motion
- one-step prediction error is low
- cosine similarity is high / cosine distance is low

Bad signs:
- latent geometry looks nice, but dynamics are not action- or state-consistent

Notes:
- `pred_target_cosine_mean` is now the **true cosine similarity**, not an unnormalised dot product.
- This makes it much more appropriate for comparing SWM-like cosine-trained models against LeWM/SIGReg models.
- `pred_error_mean` is still useful, but it is scale-dependent and should not be the only cross-method comparison metric.

### Action Effect

Most important metrics:
- `action_perturb_pred_shift_corr`
- `interpolation_monotonicity`

Good signs:
- small action changes produce smooth, structured latent changes

Bad signs:
- action perturbations barely move the latent prediction
- interpolation is jagged or non-monotonic

Notes:
- `action_perturb_pred_shift_corr` is now computed over all perturbed samples, not just one averaged value per trial
- `interpolation_monotonicity` is averaged over several anchors, so it is less sensitive to one lucky or unlucky context

## 6. How To Read t-SNE Correctly

This is important.

`t-SNE` is useful for:
- visualizing local neighborhoods
- checking whether a smooth state sheet becomes fragmented
- reproducing paper-style latent visualizations

`t-SNE` is *not* reliable for:
- judging global distance
- comparing absolute cluster spacing
- deciding that one method is better from visualization alone

Use it together with:
- `distance_corr`
- `knn_overlap`
- `latent_state_step_corr`

## 7. Environment-Specific Focus

### TwoRoom

Look for:
- smooth latent continuity inside each room
- meaningful connection near the doorway
- no artificial tearing around door-adjacent states

### PushT

Look for:
- a smooth sheet as agent/block x-y positions vary
- no fragmented islands for nearby states
- local neighborhoods that remain physically meaningful

### Cube / OGB

Look for:
- smooth transitions across contact regimes
- no abrupt manifold tears caused by contact changes

### Reacher / DMC

Look for:
- continuity across nearby arm poses
- no latent shortcuts between distant joint configurations

## 8. Dependencies

Main analysis:
- no extra dependency beyond the training environment

Optional:
- `scikit-learn` for `t-SNE`
- `matplotlib` for `plot_repr.py` and `compare_repr.py`

If missing, the scripts fail with explicit messages.

## 9. Suggested Workflow

Recommended workflow before changing losses again:

1. Run `analyze_repr.py` on `SIGReg` and the candidate model.
2. Compare `summary.json` metrics first.
3. Inspect `local_neighbors.json` for anchor-level failures.
4. Plot `PCA` and optionally `t-SNE`.
5. Use `compare_repr.py` for side-by-side visual comparison.
6. Only then decide whether the next change should target:
   - anti-collapse
   - geometry
   - predictor / action conditioning
   - planning objective

This keeps the optimization loop evidence-driven instead of loss-driven.
