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
- [batch_repr_analysis.py](/home/ag/projects/le-wm/tools/repr_analysis/batch_repr_analysis.py)
  Runs the same analysis for multiple checkpoints and writes aggregate tables.
- [compare_repr.py](/home/ag/projects/le-wm/tools/repr_analysis/compare_repr.py)
  Draws side-by-side comparison plots from two analysis directories.
- [run_repr_batch_example.sh](/home/ag/projects/le-wm/tools/repr_analysis/run_repr_batch_example.sh)
  Editable shell example that wraps the batch script for multi-model comparison.
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

Use `batch_repr_analysis.py` when:
- you want one Python entrypoint instead of a long bash script
- you want to compare several models on the same dataset with identical settings
- you want aggregate `CSV` / `Markdown` tables for later visualization
- you want the script to also emit pairwise comparison plots

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

`analyze_repr.py` prints the usual analysis blocks, and for hybrid models it
also prints raw-space counterparts:

- `Embedding (Normalized)`
  - anti-collapse / distribution health
  - norms, per-dim std, inter-sample cosine, effective rank
- `Embedding (Raw)`
  - the same anti-collapse diagnostics on `emb_raw`
- `Topology (Normalized)`
  - latent distance vs state distance
  - kNN overlap between latent space and state space
- `Topology (Raw)`
  - the same geometry diagnostics on `emb_raw`
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
- `metric_guide.json`
  Glossary describing what each metric means and how to use it.
- `pca_projection.json`
  2D PCA projection of latent embeddings.
- `pca_projection_raw.json`
  2D PCA projection of raw latent embeddings.
- `tsne_projection.json`
  2D t-SNE projection, only if exported successfully.
- `tsne_projection_raw.json`
  2D t-SNE projection of raw latent embeddings, only if exported successfully.
- `local_neighbors.json`
  Anchor-by-anchor comparison of nearest neighbors in latent vs state space.
- `local_neighbors_raw.json`
  The same nearest-neighbor report in raw latent space.

## 3. Batch Analyze Multiple Runs

When you want a single Python command to replace a manual bash loop, use
`batch_repr_analysis.py`.

Example:

```bash
python -m tools.repr_analysis.batch_repr_analysis \
  --dataset /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/pusht_expert_train \
  --state-key proprio \
  --save-dir /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/repr_analysis/pusht_batch_compare \
  --plot-projections pca \
  --compare-projections pca \
  --color-dims 0 1 \
  --model "SIGReg=/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/pusht_lewm_20260416/pusht_lewm_20260416_epoch_9_object.ckpt" \
  --model "BN+uniformity=/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/pusht_swm_v0_mlp_bn_uniform_lambda_0p1_t_2_emb_dim_192_20260417/pusht_swm_v0_mlp_bn_uniform_lambda_0p1_t_2_emb_dim_192_20260417_epoch_9_object.ckpt"
```

Batch outputs:

- one subdirectory per model with the usual `summary.json`, projection JSONs, and optional plots
- `metrics_wide.csv`
  one row per model, one column per metric
- `metrics_long.csv`
  one row per `(model, metric)` pair with metric description fields; this is the easiest file to use for later plotting
- `metrics_table.md`
  human-readable per-section tables
- `batch_summary.json`
  machine-readable aggregate result bundle
- `batch_report.txt`
  the full printed logs and per-model metric dumps
- `pairwise_compare/`
  optional side-by-side comparison plots plus `compare_manifest.json`

Pairwise compare rules:

- if you pass exactly two models and set `--compare-projections`, the script compares those two automatically
- use `--compare-all-pairs` to render all pairs when you pass more than two models
- use `--compare-pair "A=B"` to compare only specific labels

Example shell wrapper:

```bash
bash tools/repr_analysis/run_repr_batch_example.sh
```

That example script mirrors your current `STABLEWM_HOME / DATASET / EPOCH / model name`
workflow, but now uses a `MODEL_SPECS` array so you can pass 2, 3, or more models in one run.
Each entry supports `label=model_name` or `label=/full/path/to.ckpt`.

## 4. Visualize One Run

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

## 5. Compare Two Runs

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

## 6. How To Read The Results

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
- `distance_rank_corr` is the `_rank` metric: a Spearman-style rank correlation on pairwise distances
- use `_rank` when you care about whether the latent preserves the ordering of distances, even if one method rescales or normalizes the geometry
- `*_cross_seq` excludes comparisons inside the same sampled sequence window; these are often the more trustworthy metrics when judging planning geometry
- use `_cross_seq` as the stricter number when adjacent frames inside one rollout are trivially easy and may inflate the all-pairs metric

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

## 7. How To Read t-SNE Correctly

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

## 8. Environment-Specific Focus

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

## 9. Dependencies

Main analysis:
- no extra dependency beyond the training environment

Optional:
- `scikit-learn` for `t-SNE`
- `matplotlib` for `plot_repr.py` and `compare_repr.py`

If missing, the scripts fail with explicit messages.

## 10. Suggested Workflow

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
