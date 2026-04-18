"""
analyze_repr.py - Representation analysis for LeWM / SWM checkpoints.

This script is designed to answer one question before changing losses again:
"What is wrong (or right) about the representation?"

It focuses on four analysis themes:
1. embedding      - anti-collapse / distribution health
2. topology       - does latent geometry preserve state-space geometry
3. dynamics       - does latent motion track true state motion
4. action_effect  - does changing action create meaningful latent branching

For hybrid models, embedding/topology are now reported in both normalized and
raw spaces so it is easier to tell whether normalization itself is the
bottleneck.

Typical usage:

  python analyze_repr.py \
      --ckpt /path/to/model_object.pt \
      --dataset tworoom \
      --state-key proprio \
      --save-dir /tmp/repr_analysis_tworoom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

import stable_pretraining as spt
import stable_worldmodel as swm

from utils import get_column_normalizer, get_img_preprocessor

SECTION_ORDER = [
    "meta",
    "embedding",
    "embedding_raw",
    "topology",
    "topology_raw",
    "dynamics",
    "action_effect",
    "visualization",
]
SECTION_TITLES = {
    "meta": "Meta",
    "embedding": "Embedding (Normalized)",
    "embedding_raw": "Embedding (Raw)",
    "topology": "Topology (Normalized)",
    "topology_raw": "Topology (Raw)",
    "dynamics": "Dynamics",
    "action_effect": "Action Effect",
    "visualization": "Visualization",
}

# Metric metadata is used in three places:
# 1. to keep the code self-documenting,
# 2. to help the batch-analysis script write readable tables,
# 3. to make later metric additions explicit instead of burying meaning in names.
METRIC_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "embedding": {
        "n_embeddings": {
            "better": "context",
            "summary": "Number of flattened latent points included in the analysis.",
            "use": "Mainly for sanity checking that two runs are compared on the same sample count.",
        },
        "dim": {
            "better": "context",
            "summary": "Latent embedding dimensionality.",
            "use": "Only a context field; it does not measure representation quality by itself.",
        },
        "norm_mean": {
            "better": "context",
            "summary": "Average L2 norm of latent embeddings.",
            "use": "Check whether a method uses norm as part of the code. This matters before comparing scale-dependent metrics like latent step size or prediction error.",
        },
        "norm_std": {
            "better": "context",
            "summary": "Standard deviation of latent embedding norms.",
            "use": "Near-zero values often mean the method constrains embeddings onto a thin shell or unit sphere.",
        },
        "dim_std_mean": {
            "better": "higher",
            "summary": "Average standard deviation across latent dimensions.",
            "use": "Quick anti-collapse signal. Low values mean many dimensions barely move across samples.",
        },
        "dim_std_min": {
            "better": "higher",
            "summary": "Smallest per-dimension standard deviation.",
            "use": "Detects dead or nearly dead latent directions.",
        },
        "dim_std_max": {
            "better": "context",
            "summary": "Largest per-dimension standard deviation.",
            "use": "Helps detect whether a few dimensions dominate the representation.",
        },
        "inter_sample_cosine_mean": {
            "better": "lower_abs",
            "summary": "Average pairwise cosine similarity between different samples.",
            "use": "Collapse check. Values drifting toward 1 mean samples point in nearly the same direction.",
        },
        "inter_sample_cosine_std": {
            "better": "context",
            "summary": "Spread of pairwise cosine similarities.",
            "use": "Useful together with the mean to tell whether the latent cloud is uniform or anisotropic.",
        },
        "effective_rank": {
            "better": "higher",
            "summary": "Entropy-based effective rank of the centered embedding matrix.",
            "use": "Interpret it as the number of directions that carry meaningful variance, not the raw embedding size. It is a stronger capacity check than just looking at per-dimension std.",
        },
    },
    "topology": {
        "n_points": {
            "better": "context",
            "summary": "Number of flattened points used for topology analysis.",
            "use": "Sanity check that two runs used the same comparison budget.",
        },
        "k": {
            "better": "context",
            "summary": "Neighborhood size used in kNN overlap metrics.",
            "use": "Context field for interpreting `knn_overlap` values.",
        },
        "distance_corr": {
            "better": "higher",
            "summary": "Pearson correlation between latent-space and state-space pairwise distances.",
            "use": "Answers whether global distances are approximately linearly related. Good for spotting geometry collapse or large-scale distortions.",
        },
        "distance_rank_corr": {
            "better": "higher",
            "summary": "Rank correlation between latent-space and state-space pairwise distances.",
            "use": "This is the `_rank` metric. It only asks whether distance ordering is preserved, so it remains informative when one method rescales or normalizes embeddings nonlinearly.",
        },
        "knn_overlap": {
            "better": "higher",
            "summary": "Average overlap between latent and state k-nearest neighbors.",
            "use": "Local-geometry metric. High values mean nearby states stay nearby after encoding.",
        },
        "distance_corr_cross_seq": {
            "better": "higher",
            "summary": "Distance correlation computed only on pairs from different sampled sequences.",
            "use": "This is the `_cross_seq` version. It removes easy within-sequence temporal neighbors that can inflate metrics even when cross-trajectory geometry is wrong.",
        },
        "distance_rank_corr_cross_seq": {
            "better": "higher",
            "summary": "Rank correlation on cross-sequence pairs only.",
            "use": "Often the most trustworthy topology metric when judging planning geometry. It asks whether the model preserves which cross-trajectory states are closer or farther, without being fooled by temporal adjacency.",
        },
        "knn_overlap_cross_seq": {
            "better": "higher",
            "summary": "kNN overlap after excluding same-sequence candidates.",
            "use": "Stricter local-geometry check. Useful when adjacent frames in one rollout are trivially close in both latent and state space.",
        },
    },
    "dynamics": {
        "latent_step_mean": {
            "better": "context",
            "summary": "Average one-step latent displacement.",
            "use": "Only compare directly when embedding norms are comparable. Otherwise treat it as within-method context.",
        },
        "latent_step_std": {
            "better": "context",
            "summary": "Standard deviation of one-step latent displacement.",
            "use": "Tells you whether the latent dynamics vary across easy vs hard transitions.",
        },
        "state_step_mean": {
            "better": "context",
            "summary": "Average one-step state displacement in the proxy state space.",
            "use": "Reference scale for interpreting latent step statistics.",
        },
        "state_step_std": {
            "better": "context",
            "summary": "Standard deviation of one-step state displacement.",
            "use": "Reference spread for transition magnitudes in the sampled batch.",
        },
        "latent_state_step_corr": {
            "better": "higher",
            "summary": "Correlation between latent step size and state step size.",
            "use": "Checks whether larger physical changes also look larger in latent space. Good for action sensitivity and predictor calibration.",
        },
        "pred_error_mean": {
            "better": "lower",
            "summary": "Average Euclidean prediction error between one-step prediction and target embedding.",
            "use": "Useful within one method family, but scale-dependent. Always read it together with `norm_mean` or cosine metrics before comparing across methods.",
        },
        "pred_error_std": {
            "better": "lower",
            "summary": "Standard deviation of Euclidean prediction error.",
            "use": "Helps tell whether the predictor is consistently wrong or fails only on certain transitions.",
        },
        "pred_target_cosine_mean": {
            "better": "higher",
            "summary": "Average cosine similarity between one-step prediction and target embedding.",
            "use": "Scale-free predictor metric. Particularly useful when one method enforces unit-norm embeddings.",
        },
        "pred_target_cosine_std": {
            "better": "lower",
            "summary": "Standard deviation of prediction-target cosine similarity.",
            "use": "Low spread means prediction quality is more consistent across transitions.",
        },
        "pred_target_cosine_distance_mean": {
            "better": "lower",
            "summary": "Average cosine distance between one-step prediction and target embedding.",
            "use": "Equivalent information to cosine similarity but in a lower-is-better form that is sometimes easier to compare in tables.",
        },
    },
    "action_effect": {
        "n_trials": {
            "better": "context",
            "summary": "Number of perturbation trials used in action-effect analysis.",
            "use": "Context field for the robustness of the perturbation statistics.",
        },
        "n_action_pairs": {
            "better": "context",
            "summary": "Number of action-perturbation / prediction-shift pairs evaluated.",
            "use": "Sanity check that the action-effect estimate is based on enough samples.",
        },
        "n_interp_anchors": {
            "better": "context",
            "summary": "Number of anchors used for action interpolation tests.",
            "use": "Context field for how many contexts contribute to interpolation smoothness.",
        },
        "perturb_scale": {
            "better": "context",
            "summary": "Scale multiplier applied to action standard deviation when perturbing actions.",
            "use": "Keep this fixed across runs if you want fair action-effect comparisons.",
        },
        "mean_action_perturb_norm": {
            "better": "context",
            "summary": "Average norm of the sampled action perturbations.",
            "use": "Reference scale for interpreting the resulting latent shift.",
        },
        "mean_pred_shift_norm": {
            "better": "context",
            "summary": "Average norm of the prediction change caused by action perturbations.",
            "use": "Within-method measure of how much action can move the prediction. It is scale-dependent if embeddings use different norms.",
        },
        "action_perturb_pred_shift_corr": {
            "better": "higher",
            "summary": "Correlation between perturbation magnitude and prediction-shift magnitude.",
            "use": "Tests whether larger action changes reliably cause larger latent changes. This is stronger than just asking whether any shift happened at all.",
        },
        "interpolation_endpoint_shift": {
            "better": "context",
            "summary": "Average latent distance from the interpolation start to the interpolation end.",
            "use": "Context field for how much the chosen action interpolation actually moves the prediction.",
        },
        "interpolation_endpoint_shift_std": {
            "better": "lower",
            "summary": "Standard deviation of endpoint shift across interpolation anchors.",
            "use": "High spread means action sensitivity is unstable across contexts.",
        },
        "interpolation_monotonicity": {
            "better": "higher",
            "summary": "Fraction of interpolation steps whose latent distance grows monotonically from the start.",
            "use": "Smoothness check. High values mean action interpolation produces orderly latent motion instead of jagged jumps.",
        },
        "interpolation_monotonicity_std": {
            "better": "lower",
            "summary": "Standard deviation of interpolation monotonicity across anchors.",
            "use": "Low spread means the smoothness pattern is stable across contexts.",
        },
    },
    "visualization": {
        "tsne_exported": {
            "better": "context",
            "summary": "Whether t-SNE export succeeded.",
            "use": "Context only. t-SNE is for visual inspection, not for quantitative selection.",
        },
        "tsne_perplexity": {
            "better": "context",
            "summary": "Perplexity actually used for t-SNE export.",
            "use": "Useful when comparing plots produced from different sample counts.",
        },
        "tsne_error": {
            "better": "context",
            "summary": "Error message captured when t-SNE export failed.",
            "use": "Debug field for the visualization step.",
        },
    },
}

for raw_section, base_section in {"embedding_raw": "embedding", "topology_raw": "topology"}.items():
    METRIC_SPECS[raw_section] = {
        key: dict(spec) for key, spec in METRIC_SPECS[base_section].items()
    }


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load an object checkpoint saved by ModelObjectCallBack."""
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.to(device).eval().requires_grad_(False)


def infer_history_size(model) -> int:
    predictor = getattr(model, "predictor", None)
    if predictor is None or not hasattr(predictor, "pos_embedding"):
        raise ValueError("Unable to infer history_size from model.predictor.pos_embedding")
    return int(predictor.pos_embedding.shape[1])


def load_dataset_samples(
    *,
    dataset_name: str,
    state_key: str,
    n_sequences: int,
    history_size: int,
    frameskip: int,
    img_size: int,
    seed: int,
    device: str,
):
    """Load a random subset of sequences with the same preprocessing as training."""
    num_steps = history_size + 1
    ds = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=["pixels", "action", state_key],
        transform=None,
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size),
        get_column_normalizer(ds, "action", "action"),
        get_column_normalizer(ds, state_key, state_key),
    )

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_sequences].tolist()
    samples = [ds[i] for i in indices]

    batch = {
        "pixels": torch.stack([s["pixels"] for s in samples]).to(device),
        "action": torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0).to(device),
        "state": torch.stack([s[state_key] for s in samples]).to(device),
    }
    return batch


@torch.no_grad()
def encode_sequences(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Encode sequences and compute one-step predictions."""
    info = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    emb = info["emb"]
    act_emb = info["act_emb"]

    ctx_len = infer_history_size(model)
    pred = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    tgt = emb[:, 1:]

    return {
        "emb": emb,
        "pred": pred,
        "tgt": tgt,
        "action": batch["action"],
        "state": batch["state"],
    }


def exclude_diagonal(mat: torch.Tensor) -> torch.Tensor:
    n = mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat[mask]


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((x @ y) / denom)


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank correlation for geometry comparisons.

    Pearson correlation on distances is still useful, but it assumes a roughly
    linear relationship between latent and state distances. The `_rank` metric
    relaxes that assumption: it only checks whether the ordering of distances is
    preserved. This is especially helpful when comparing methods that impose
    different latent scales, such as unit-sphere losses vs unconstrained codes.
    """

    def rankdata(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=v.device)
        return ranks

    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    return pearson_corr(rankdata(x), rankdata(y))


def effective_rank(z: torch.Tensor) -> float:
    """Entropy-based effective rank of the centered embedding matrix.

    The raw embedding dimension answers "how many coordinates exist?".
    Effective rank answers "how many directions actually carry variance?".
    A 192-dim embedding can still behave like a 20-dim code if almost all
    variance is concentrated in a small subspace.
    """

    z = z - z.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(z)
    power = singular_values.square()
    total = power.sum()
    if float(total) < 1e-12:
        return 0.0
    p = power / total
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def flatten_sequence_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1)).cpu()


def flattened_sequence_ids(x: torch.Tensor) -> torch.Tensor:
    """Return the sequence id for each flattened time step."""
    b, t = x.shape[:2]
    return torch.arange(b).repeat_interleave(t)


def sample_flat_points(
    z: torch.Tensor,
    state: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly subsample flattened points for topology analysis.

    Random subsampling avoids a subtle bias from taking the first N flattened
    points, which would over-represent early sequences and time steps.
    """

    flat_z = flatten_sequence_tensor(z)
    flat_s = flatten_sequence_tensor(state)
    seq_ids = flattened_sequence_ids(state)

    n_total = flat_z.shape[0]
    n_keep = min(max_points, n_total)
    if n_keep == n_total:
        return flat_z, flat_s, seq_ids

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)[:n_keep]
    return flat_z[indices], flat_s[indices], seq_ids[indices]


def pca_projection(z: torch.Tensor, out_dim: int = 2) -> torch.Tensor:
    z = z - z.mean(0, keepdim=True)
    q = min(max(out_dim, 2), z.shape[1], z.shape[0] - 1)
    u, s, _ = torch.pca_lowrank(z, q=q)
    return u[:, :out_dim] * s[:out_dim]


def tsne_projection(
    z: torch.Tensor,
    *,
    out_dim: int = 2,
    perplexity: float = 30.0,
    random_state: int = 3072,
    pca_dim: int = 32,
) -> torch.Tensor:
    """Optional t-SNE projection for visual inspection of local neighborhoods.

    This stays separate from the quantitative analysis because t-SNE is useful
    for visualizing local structure, but not reliable for judging global
    geometry or absolute cluster spacing.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "t-SNE requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    z = z.cpu()
    n = z.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for t-SNE projection.")

    max_perplexity = max(1.0, float((n - 1) // 3))
    perplexity = min(perplexity, max_perplexity)
    if pca_dim and z.shape[1] > pca_dim:
        z = pca_projection(z, out_dim=min(pca_dim, z.shape[1]))

    tsne = TSNE(
        n_components=out_dim,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    proj = tsne.fit_transform(z.numpy())
    return torch.from_numpy(proj)


def knn_overlap(
    a_dist: torch.Tensor,
    b_dist: torch.Tensor,
    k: int,
    *,
    invalid_mask: torch.Tensor | None = None,
) -> float:
    """Average neighbor overlap between two distance matrices.

    This is a local-geometry metric. Unlike distance correlations, it does not
    care about exact metric scale; it only asks whether each point keeps roughly
    the same nearest neighbors after encoding.
    """

    n = a_dist.size(0)
    a = a_dist.clone()
    b = b_dist.clone()
    eye = torch.eye(n, dtype=torch.bool, device=a.device)
    a[eye] = float("inf")
    b[eye] = float("inf")
    if invalid_mask is not None:
        a[invalid_mask] = float("inf")
        b[invalid_mask] = float("inf")
    a_idx = torch.topk(a, k=k, largest=False).indices
    b_idx = torch.topk(b, k=k, largest=False).indices

    overlaps = []
    for i in range(n):
        overlaps.append(len(set(a_idx[i].tolist()) & set(b_idx[i].tolist())) / float(k))
    return float(sum(overlaps) / len(overlaps))


def analyze_embedding(z: torch.Tensor) -> Dict[str, float]:
    """Measure distribution health and anti-collapse behavior of the latent space."""
    flat = z.reshape(-1, z.size(-1)).cpu()
    norms = flat.norm(dim=-1)
    cos = F.normalize(flat, dim=-1) @ F.normalize(flat, dim=-1).T
    dim_std = flat.std(dim=0)
    return {
        "n_embeddings": int(flat.shape[0]),
        "dim": int(flat.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "dim_std_mean": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "inter_sample_cosine_mean": float(exclude_diagonal(cos).mean()),
        "inter_sample_cosine_std": float(exclude_diagonal(cos).std()),
        "effective_rank": effective_rank(flat),
    }


def analyze_topology(
    z: torch.Tensor,
    state: torch.Tensor,
    *,
    k: int,
    max_points: int,
    seed: int,
) -> Dict[str, float]:
    """Measure whether latent geometry preserves state-space geometry.

    Two additions here deserve special attention:

    - `distance_rank_corr`:
      a Spearman-style version of distance correlation that is robust to
      monotonic rescaling. This is important when one method constrains latent
      norms and another does not.

    - `*_cross_seq` metrics:
      these exclude pairs from the same sampled sequence window. Within-sequence
      temporal neighbors are often trivially close and can make a broken
      representation look better than it is. Cross-sequence metrics are stricter
      and are usually more informative for planning-style geometry.
    """

    flat_z, flat_s, seq_ids = sample_flat_points(z, state, max_points=max_points, seed=seed)
    n = flat_z.shape[0]

    # Standardize state coordinates so one physical dimension does not dominate
    # the distance metric purely due to units or scale.
    flat_s = (flat_s - flat_s.mean(0, keepdim=True)) / flat_s.std(0, keepdim=True).clamp_min(1e-6)
    z_dist = torch.cdist(flat_z, flat_z)
    s_dist = torch.cdist(flat_s, flat_s)

    offdiag = ~torch.eye(n, dtype=torch.bool)
    cross_seq = offdiag & (seq_ids[:, None] != seq_ids[None, :])

    # Keep the original normalized sections for backward compatibility, then
    # add explicit raw-space counterparts so hybrid models can reveal whether
    # the planning bottleneck lives in normalization or in the raw code itself.
    result = {
        "n_points": int(n),
        "k": int(k),
        "distance_corr": pearson_corr(z_dist[offdiag], s_dist[offdiag]),
        "distance_rank_corr": spearman_corr(z_dist[offdiag], s_dist[offdiag]),
        "knn_overlap": knn_overlap(z_dist, s_dist, k=min(k, n - 1)),
    }

    if bool(cross_seq.any()):
        invalid_same_seq = ~cross_seq & ~torch.eye(n, dtype=torch.bool)
        result["distance_corr_cross_seq"] = pearson_corr(z_dist[cross_seq], s_dist[cross_seq])
        result["distance_rank_corr_cross_seq"] = spearman_corr(z_dist[cross_seq], s_dist[cross_seq])
        result["knn_overlap_cross_seq"] = knn_overlap(
            z_dist,
            s_dist,
            k=min(k, n - 1),
            invalid_mask=invalid_same_seq,
        )

    return result


def analyze_dynamics(z: torch.Tensor, state: torch.Tensor, pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    """Measure whether one-step latent dynamics track real state transitions."""
    z_now = z[:, :-1]
    z_next = z[:, 1:]
    s_now = state[:, :-1]
    s_next = state[:, 1:]

    latent_step = (z_next - z_now).norm(dim=-1).reshape(-1).cpu()
    state_step = (s_next - s_now).norm(dim=-1).reshape(-1).cpu()
    pred_err = (pred - tgt).norm(dim=-1).reshape(-1).cpu()

    pred_flat = pred.reshape(-1, pred.size(-1))
    tgt_flat = tgt.reshape(-1, tgt.size(-1))
    pred_cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).cpu()
    pred_cos_dist = (1.0 - pred_cos).cpu()

    return {
        "latent_step_mean": float(latent_step.mean()),
        "latent_step_std": float(latent_step.std()),
        "state_step_mean": float(state_step.mean()),
        "state_step_std": float(state_step.std()),
        "latent_state_step_corr": pearson_corr(latent_step, state_step),
        "pred_error_mean": float(pred_err.mean()),
        "pred_error_std": float(pred_err.std()),
        "pred_target_cosine_mean": float(pred_cos.mean()),
        "pred_target_cosine_std": float(pred_cos.std()),
        "pred_target_cosine_distance_mean": float(pred_cos_dist.mean()),
    }


@torch.no_grad()
def analyze_action_effect(
    model,
    z: torch.Tensor,
    action: torch.Tensor,
    *,
    n_trials: int,
    interp_steps: int,
    perturb_scale: float,
) -> Dict[str, float]:
    """Measure whether action changes produce structured latent branching."""
    ctx_len = infer_history_size(model)
    ctx_emb = z[:, :ctx_len]
    base_action = action[:, :ctx_len].clone()
    base_pred = model.predict(ctx_emb, model.action_encoder(base_action))[:, -1]

    action_std = action.reshape(-1, action.size(-1)).std(dim=0).clamp_min(1e-6)

    perturb_norms = []
    pred_shift_norms = []
    for _ in range(n_trials):
        delta = torch.randn_like(base_action[:, -1]) * action_std * perturb_scale
        perturbed = base_action.clone()
        perturbed[:, -1] = perturbed[:, -1] + delta
        pred = model.predict(ctx_emb, model.action_encoder(perturbed))[:, -1]

        perturb_norms.append(delta.norm(dim=-1))
        pred_shift_norms.append((pred - base_pred).norm(dim=-1))

    perturb_norms = torch.cat(perturb_norms).cpu()
    pred_shift_norms = torch.cat(pred_shift_norms).cpu()

    n_interp_anchors = min(8, base_action.size(0))
    anchor_idx = torch.linspace(0, base_action.size(0) - 1, steps=n_interp_anchors).long().unique()
    alphas = torch.linspace(0, 1, interp_steps, device=base_action.device)

    endpoint_shifts = []
    monotonicities = []
    for idx in anchor_idx.tolist():
        action_a = base_action[idx : idx + 1].clone()
        action_b = base_action[idx : idx + 1].clone()
        action_b[:, -1] = action_b[:, -1] + action_std.unsqueeze(0) * perturb_scale

        interp_preds: List[torch.Tensor] = []
        single_ctx = ctx_emb[idx : idx + 1]
        for alpha in alphas:
            act = (1 - alpha) * action_a + alpha * action_b
            pred = model.predict(single_ctx, model.action_encoder(act))[:, -1]
            interp_preds.append(pred.squeeze(0))

        interp_preds = torch.stack(interp_preds)
        interp_dist = (interp_preds - interp_preds[0]).norm(dim=-1).cpu()
        endpoint_shifts.append(interp_dist[-1])
        monotonicities.append(((interp_dist[1:] - interp_dist[:-1]) >= 0).float().mean())

    endpoint_shifts = torch.stack(endpoint_shifts)
    monotonicities = torch.stack(monotonicities)

    return {
        "n_trials": int(n_trials),
        "n_action_pairs": int(perturb_norms.numel()),
        "n_interp_anchors": int(anchor_idx.numel()),
        "perturb_scale": float(perturb_scale),
        "mean_action_perturb_norm": float(perturb_norms.mean()),
        "mean_pred_shift_norm": float(pred_shift_norms.mean()),
        "action_perturb_pred_shift_corr": pearson_corr(perturb_norms, pred_shift_norms),
        "interpolation_endpoint_shift": float(endpoint_shifts.mean()),
        "interpolation_endpoint_shift_std": float(endpoint_shifts.std()),
        "interpolation_monotonicity": float(monotonicities.mean()),
        "interpolation_monotonicity_std": float(monotonicities.std()),
    }


def build_local_neighbor_report(
    z: torch.Tensor,
    state: torch.Tensor,
    *,
    k: int,
    n_anchors: int,
) -> List[Dict[str, Any]]:
    """Compare local neighborhoods in latent space and state space.

    This file is meant for qualitative debugging after the scalar metrics point
    to a topology problem. It lets you inspect concrete anchor states and see
    which neighbors are preserved vs scrambled.
    """

    flat_z = z.reshape(-1, z.size(-1)).cpu()
    flat_s = state.reshape(-1, state.size(-1)).cpu()
    n = flat_z.shape[0]
    seq_len = state.size(1)

    z_dist = torch.cdist(flat_z, flat_z)
    s_norm = (flat_s - flat_s.mean(0, keepdim=True)) / flat_s.std(0, keepdim=True).clamp_min(1e-6)
    s_dist = torch.cdist(s_norm, s_norm)

    eye = torch.eye(n, dtype=torch.bool)
    z_dist[eye] = float("inf")
    s_dist[eye] = float("inf")

    anchors = torch.linspace(0, n - 1, steps=min(n_anchors, n)).long().unique()
    report: List[Dict[str, Any]] = []
    for anchor in anchors.tolist():
        z_nn = torch.topk(z_dist[anchor], k=min(k, n - 1), largest=False).indices.tolist()
        s_nn = torch.topk(s_dist[anchor], k=min(k, n - 1), largest=False).indices.tolist()
        item = {
            "anchor_index": anchor,
            "anchor_seq_id": anchor // seq_len,
            "anchor_time_id": anchor % seq_len,
            "anchor_state": flat_s[anchor].tolist(),
            "overlap": len(set(z_nn) & set(s_nn)) / float(min(k, n - 1)),
            "latent_nn": [],
            "state_nn": [],
        }
        for idx in z_nn:
            item["latent_nn"].append({
                "index": idx,
                "seq_id": idx // seq_len,
                "time_id": idx % seq_len,
                "latent_distance": float(z_dist[anchor, idx]),
                "state_distance": float(s_dist[anchor, idx]),
                "state": flat_s[idx].tolist(),
            })
        for idx in s_nn:
            item["state_nn"].append({
                "index": idx,
                "seq_id": idx // seq_len,
                "time_id": idx % seq_len,
                "latent_distance": float(z_dist[anchor, idx]),
                "state_distance": float(s_dist[anchor, idx]),
                "state": flat_s[idx].tolist(),
            })
        report.append(item)
    return report


def make_interpretation(dataset: str, result: Dict[str, Dict[str, float]]) -> List[str]:
    """Environment-aware interpretation hints."""
    dataset_l = dataset.lower()
    emb = result["embedding"]
    topo = result["topology"]
    emb_raw = result.get("embedding_raw")
    topo_raw = result.get("topology_raw")
    dyn = result["dynamics"]
    act = result["action_effect"]
    topo_corr = topo.get("distance_corr_cross_seq", topo["distance_corr"])
    topo_knn = topo.get("knn_overlap_cross_seq", topo["knn_overlap"])
    topo_rank = topo.get("distance_rank_corr_cross_seq", topo["distance_rank_corr"])

    hints: List[str] = []

    if emb["effective_rank"] < 4:
        hints.append("Embedding rank is very low; the representation may still be partially collapsed.")
    elif emb["effective_rank"] < 12:
        hints.append("Embedding rank is moderate; anti-collapse works, but capacity usage may still be limited.")
    else:
        hints.append("Embedding rank looks healthy; remaining issues likely come from geometry or dynamics rather than collapse.")

    if topo_corr < 0.2:
        hints.append("Latent distances do not track state distances well; geometry is a likely bottleneck.")
    elif topo_corr < 0.5:
        hints.append("Latent geometry partially tracks state geometry, but there is still clear distortion.")
    else:
        hints.append("Latent geometry tracks state geometry reasonably well.")

    if topo_knn < 0.2:
        hints.append("Local neighborhoods disagree strongly between latent and state spaces.")
    elif topo_knn > 0.5:
        hints.append("Local neighborhoods are preserved fairly well.")

    if isinstance(topo_raw, dict):
        raw_topo_rank = topo_raw.get("distance_rank_corr_cross_seq", topo_raw["distance_rank_corr"])
        if raw_topo_rank > topo_rank + 0.1:
            hints.append(
                "Raw latent geometry is noticeably stronger than normalized geometry; "
                "if planning rolls out on normalized states, that branch may be the bottleneck."
            )
        elif raw_topo_rank < 0.1 and topo_rank < 0.1:
            hints.append(
                "Both normalized and raw latent geometry are weak; switching only the planning "
                "space is unlikely to rescue performance."
            )

    if isinstance(emb_raw, dict):
        raw_cos = emb_raw.get("inter_sample_cosine_mean", 0.0)
        norm_cos = emb.get("inter_sample_cosine_mean", 0.0)
        if norm_cos > 0.9 and raw_cos < 0.5:
            hints.append(
                "Normalized embeddings are much more concentrated than raw embeddings; "
                "L2 normalization may be discarding useful amplitude or angular separation."
            )

    if dyn["latent_state_step_corr"] < 0.2:
        hints.append("Latent motion is weakly aligned with true state motion; predictor/action conditioning is a good next target.")
    elif dyn["latent_state_step_corr"] > 0.5:
        hints.append("Latent motion is fairly well aligned with true state motion.")

    if act["action_perturb_pred_shift_corr"] < 0.2:
        hints.append("Action perturbations do not produce structured latent branching; action usage may be weak.")
    elif act["action_perturb_pred_shift_corr"] > 0.5:
        hints.append("Action perturbations produce structured latent branching.")

    if act["interpolation_monotonicity"] < 0.75:
        hints.append("Action interpolation is not very smooth; local latent dynamics may be jagged.")

    if "tworoom" in dataset_l:
        hints.append("For TwoRoom, inspect whether door-adjacent states remain connected instead of splitting into disconnected islands.")
    elif "pusht" in dataset_l:
        hints.append("For PushT, inspect whether agent/block position changes form a smooth sheet instead of fragmented clusters.")
    elif "cube" in dataset_l or "ogb" in dataset_l:
        hints.append("For Cube, inspect whether contact transitions bend the manifold smoothly instead of tearing local neighborhoods.")
    elif "reacher" in dataset_l or "dmc" in dataset_l:
        hints.append("For Reacher, inspect whether continuous arm poses remain continuous in the latent projection.")

    return hints


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def metric_spec(section: str, key: str) -> Dict[str, str]:
    return METRIC_SPECS.get(section, {}).get(key, {})


def metric_entries(result: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, Any]]:
    """Return scalar-like result entries in a stable order for table export."""
    entries: List[Tuple[str, str, Any]] = []
    for section in SECTION_ORDER:
        metrics = result.get(section)
        if not isinstance(metrics, dict):
            continue
        preferred_keys = list(METRIC_SPECS.get(section, {}).keys())
        extra_keys = [key for key in metrics if key not in preferred_keys]
        for key in preferred_keys + extra_keys:
            if key in metrics:
                entries.append((section, key, metrics[key]))
    return entries


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_section(title: str, metrics: Dict[str, Any]) -> str:
    lines = [f"\n{'=' * 72}", title, f"{'=' * 72}"]
    for key, value in metrics.items():
        lines.append(f"{key}: {format_metric_value(value)}")
    return "\n".join(lines)


def format_hints(hints: List[str]) -> str:
    lines = [f"\n{'=' * 72}", "Interpretation", f"{'=' * 72}"]
    for hint in hints:
        lines.append(f"- {hint}")
    return "\n".join(lines)


def format_analysis_report(result: Dict[str, Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for section in SECTION_ORDER:
        metrics = result.get(section)
        if isinstance(metrics, dict) and metrics:
            chunks.append(format_section(SECTION_TITLES.get(section, section.title()), metrics))
    hints = result.get("interpretation")
    if isinstance(hints, list):
        chunks.append(format_hints(hints))
    return "\n".join(chunks).lstrip()


def run_analysis(
    *,
    ckpt: str,
    dataset: str,
    state_key: str,
    frameskip: int,
    img_size: int,
    n_sequences: int,
    max_points: int,
    knn_k: int,
    action_trials: int,
    interp_steps: int,
    perturb_scale: float,
    seed: int,
    device: str,
    log: Callable[[str], None] | None = print,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Run the full representation analysis and return metrics plus tensors."""

    if log is not None:
        log(f"[analyze_repr] loading model: {ckpt}")
    model = load_model(ckpt, device)
    history_size = infer_history_size(model)
    if log is not None:
        log(f"[analyze_repr] dataset={dataset} state_key={state_key} history_size={history_size}")

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=history_size,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )
    outputs = encode_sequences(model, batch)

    result = {
        "meta": {
            "ckpt": ckpt,
            "dataset": dataset,
            "state_key": state_key,
            "n_sequences": n_sequences,
            "history_size": history_size,
            "device": device,
            "analysis_prediction_space": getattr(model, "analysis_prediction_space", "normalized"),
            "inference_rollout_state_space": getattr(model, "inference_rollout_state_space", "normalized"),
            "inference_cost_space": getattr(model, "inference_cost_space", "normalized"),
            "inference_cost_type": getattr(model, "inference_cost_type", "cosine"),
        },
        "embedding": analyze_embedding(outputs["emb"]),
        "embedding_raw": analyze_embedding(outputs["emb_raw"]),
        "topology": analyze_topology(
            outputs["emb"],
            outputs["state"],
            k=knn_k,
            max_points=max_points,
            seed=seed,
        ),
        "topology_raw": analyze_topology(
            outputs["emb_raw"],
            outputs["state"],
            k=knn_k,
            max_points=max_points,
            seed=seed,
        ),
        "dynamics": analyze_dynamics(outputs["emb"], outputs["state"], outputs["pred"], outputs["tgt"]),
        "action_effect": analyze_action_effect(
            model,
            outputs["emb"],
            outputs["action"],
            n_trials=action_trials,
            interp_steps=interp_steps,
            perturb_scale=perturb_scale,
        ),
    }
    result["interpretation"] = make_interpretation(dataset, result)
    return result, outputs


def save_projection_rows(proj: torch.Tensor, state: torch.Tensor, save_path: Path):
    flat_s = state.reshape(-1, state.size(-1)).cpu()
    rows = []
    seq_len = state.size(1)
    for idx in range(proj.size(0)):
        seq_id = idx // seq_len
        time_id = idx % seq_len
        row = {
            "seq_id": seq_id,
            "time_id": time_id,
            "x": float(proj[idx, 0]),
            "y": float(proj[idx, 1]),
        }
        for dim in range(flat_s.size(1)):
            row[f"state_{dim}"] = float(flat_s[idx, dim])
        rows.append(row)

    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)


def save_metric_guide(save_path: Path):
    """Save the metric glossary so result folders remain self-explanatory."""
    with open(save_path, "w") as f:
        json.dump(to_serializable(METRIC_SPECS), f, indent=2)


def save_outputs(
    save_dir: Path,
    result: Dict[str, Dict[str, Any]],
    z: torch.Tensor,
    z_raw: torch.Tensor,
    state: torch.Tensor,
    *,
    export_tsne: bool,
    tsne_perplexity: float,
    seed: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Persist both normalized and raw projections so downstream debugging can
    # inspect exactly the same sampled states in each latent space.
    flat_z = z.reshape(-1, z.size(-1)).cpu()
    pca_proj = pca_projection(flat_z, out_dim=2)
    save_projection_rows(pca_proj, state, save_dir / "pca_projection.json")

    flat_z_raw = z_raw.reshape(-1, z_raw.size(-1)).cpu()
    pca_proj_raw = pca_projection(flat_z_raw, out_dim=2)
    save_projection_rows(pca_proj_raw, state, save_dir / "pca_projection_raw.json")

    if export_tsne:
        try:
            tsne_proj = tsne_projection(
                flat_z,
                out_dim=2,
                perplexity=tsne_perplexity,
                random_state=seed,
            )
            save_projection_rows(tsne_proj, state, save_dir / "tsne_projection.json")
            result.setdefault("visualization", {})["tsne_exported"] = True
            result["visualization"]["tsne_perplexity"] = float(
                min(tsne_perplexity, max(1.0, float((flat_z.size(0) - 1) // 3)))
            )

            tsne_proj_raw = tsne_projection(
                flat_z_raw,
                out_dim=2,
                perplexity=tsne_perplexity,
                random_state=seed,
            )
            save_projection_rows(tsne_proj_raw, state, save_dir / "tsne_projection_raw.json")
        except Exception as exc:
            result.setdefault("visualization", {})["tsne_exported"] = False
            result["visualization"]["tsne_error"] = str(exc)

    with open(save_dir / "local_neighbors.json", "w") as f:
        json.dump(build_local_neighbor_report(z, state, k=8, n_anchors=12), f, indent=2)
    with open(save_dir / "local_neighbors_raw.json", "w") as f:
        json.dump(build_local_neighbor_report(z_raw, state, k=8, n_anchors=12), f, indent=2)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(to_serializable(result), f, indent=2)
    save_metric_guide(save_dir / "metric_guide.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Object checkpoint path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. tworoom")
    parser.add_argument("--state-key", type=str, default="proprio", help="State proxy key in dataset")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-sequences", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--action-trials", type=int, default=8)
    parser.add_argument("--interp-steps", type=int, default=9)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--export-tsne", action="store_true", help="Export t-SNE projection if sklearn is installed")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    result, outputs = run_analysis(
        ckpt=args.ckpt,
        dataset=args.dataset,
        state_key=args.state_key,
        frameskip=args.frameskip,
        img_size=args.img_size,
        n_sequences=args.n_sequences,
        max_points=args.max_points,
        knn_k=args.knn_k,
        action_trials=args.action_trials,
        interp_steps=args.interp_steps,
        perturb_scale=args.perturb_scale,
        seed=args.seed,
        device=args.device,
        log=print,
    )

    if args.export_tsne:
        print(
            "\n[analyze_repr] t-SNE requested: this is for local-neighborhood visualisation only; "
            "do not use it as a quantitative judge of global latent geometry."
        )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_outputs(
            save_dir,
            result,
            outputs["emb"],
            outputs["emb_raw"],
            outputs["state"],
            export_tsne=args.export_tsne,
            tsne_perplexity=args.tsne_perplexity,
            seed=args.seed,
        )
        print(f"\n[analyze_repr] saved outputs to: {save_dir}")

    print(f"\n{format_analysis_report(result)}")


if __name__ == "__main__":
    main()
