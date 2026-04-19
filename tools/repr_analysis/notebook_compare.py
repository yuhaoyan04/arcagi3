"""
Notebook-friendly utilities for representation analysis.

This module supports two interactive workflows:

1. load an existing `batch_repr_analysis.py` output directory
2. run `analyze_repr.run_analysis()` directly from a notebook and compare
   checkpoints in memory without going through a shell script first
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    import torch


SECTION_ORDER = ["meta", "embedding", "topology", "dynamics", "action_effect", "visualization"]
METRIC_SECTION_ORDER = ["embedding", "topology", "dynamics", "action_effect", "visualization"]


DEFAULT_KEY_METRICS: List[Dict[str, str]] = [
    {
        "group": "Embedding",
        "metric_name": "embedding.effective_rank",
        "label": "effective_rank",
        "short_label": "rank",
        "better": "higher",
        "brief": "used latent directions",
        "summary": "How many latent directions carry meaningful variance.",
        "diagnosis": "Low means the code underuses capacity or is partially collapsed.",
    },
    {
        "group": "Embedding",
        "metric_name": "embedding.inter_sample_cosine_mean",
        "label": "inter_sample_cosine_mean",
        "short_label": "inter_cos",
        "better": "lower_abs",
        "brief": "sample spread",
        "summary": "Average alignment between different samples.",
        "diagnosis": "Large magnitude means samples crowd into similar directions instead of spreading cleanly.",
    },
    {
        "group": "Topology",
        "metric_name": "topology.distance_rank_corr_cross_seq",
        "label": "distance_rank_corr_cross_seq",
        "short_label": "rank_corr_xseq",
        "better": "higher",
        "brief": "global cross-trajectory geometry",
        "summary": "Whether cross-trajectory distance ordering is preserved.",
        "diagnosis": "Low means the global planning geometry is distorted across trajectories.",
    },
    {
        "group": "Topology",
        "metric_name": "topology.knn_overlap_cross_seq",
        "label": "knn_overlap_cross_seq",
        "short_label": "knn_xseq",
        "better": "higher",
        "brief": "local cross-trajectory neighborhoods",
        "summary": "Whether cross-trajectory local neighbors stay neighbors in latent space.",
        "diagnosis": "Low means local neighborhoods are scrambled even when states should be close.",
    },
    {
        "group": "Dynamics",
        "metric_name": "dynamics.pred_target_cosine_distance_mean",
        "label": "pred_target_cosine_distance_mean",
        "short_label": "pred_cos_dist",
        "better": "lower",
        "brief": "one-step predictor accuracy",
        "summary": "Scale-free one-step prediction error.",
        "diagnosis": "High means the predictor is missing the next latent target.",
    },
    {
        "group": "Dynamics",
        "metric_name": "dynamics.latent_state_step_corr",
        "label": "latent_state_step_corr",
        "short_label": "step_corr",
        "better": "higher",
        "brief": "latent/state motion alignment",
        "summary": "Whether larger state moves also produce larger latent moves.",
        "diagnosis": "Low means latent motion is poorly calibrated to environment motion.",
    },
    {
        "group": "Action Effect",
        "metric_name": "action_effect.mean_pred_shift_norm",
        "label": "mean_pred_shift_norm",
        "short_label": "pred_shift",
        "better": "higher",
        "brief": "action-induced shift size",
        "summary": "How much action perturbations move the predicted latent state.",
        "diagnosis": "Too low means actions barely change the rollout, even if correlation looks decent.",
    },
    {
        "group": "Action Effect",
        "metric_name": "action_effect.action_perturb_pred_shift_corr",
        "label": "action_perturb_pred_shift_corr",
        "short_label": "act_shift_corr",
        "better": "higher",
        "brief": "action magnitude consistency",
        "summary": "Whether larger action perturbations cause larger latent shifts.",
        "diagnosis": "Low means action magnitude is not translated into latent dynamics reliably.",
    },
]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "notebook_compare.py requires pandas. Install it with `pip install pandas`."
        ) from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "notebook_compare.py requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Live notebook analysis requires torch. Install the project runtime "
            "or use the saved-results workflow only."
        ) from exc
    return torch


def _default_device() -> str:
    torch = _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _require_analyze_repr():
    try:
        from tools.repr_analysis.analyze_repr import (
            format_analysis_report,
            metric_entries,
            pca_projection,
            run_analysis,
            save_outputs,
            tsne_projection,
        )
    except ImportError as exc:
        raise ImportError(
            "Live notebook analysis requires the local project runtime and "
            "`tools.repr_analysis.analyze_repr` dependencies."
        ) from exc
    return {
        "format_analysis_report": format_analysis_report,
        "metric_entries": metric_entries,
        "pca_projection": pca_projection,
        "run_analysis": run_analysis,
        "save_outputs": save_outputs,
        "tsne_projection": tsne_projection,
    }


def load_batch_summary(batch_dir: str | Path) -> Dict[str, Any]:
    path = Path(batch_dir) / "batch_summary.json"
    with open(path, "r") as f:
        return json.load(f)


def load_metrics_wide(batch_dir: str | Path):
    pd = _require_pandas()
    path = Path(batch_dir) / "metrics_wide.csv"
    return pd.read_csv(path)


def analysis_dirs_by_label(batch_dir: str | Path) -> Dict[str, Path]:
    summary = load_batch_summary(batch_dir)
    return {
        record["label"]: Path(record["analysis_dir"])
        for record in summary.get("models", [])
    }


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return slug or "model"


def make_unique_slug(label: str, used: set[str]) -> str:
    base = slugify(label)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def parse_model_spec(spec: str | Sequence[str]) -> Tuple[str, str]:
    if isinstance(spec, str):
        if "=" in spec:
            label, ckpt = spec.split("=", 1)
            label = label.strip()
            ckpt = ckpt.strip()
            if not label or not ckpt:
                raise ValueError(f"Invalid model spec: {spec}")
            return label, ckpt
        ckpt = spec.strip()
        if not ckpt:
            raise ValueError("Empty model spec.")
        return Path(ckpt).stem, ckpt

    if len(spec) != 2:
        raise ValueError(f"Expected (label, ckpt) pair, got: {spec}")
    label, ckpt = str(spec[0]).strip(), str(spec[1]).strip()
    if not label or not ckpt:
        raise ValueError(f"Invalid model spec: {spec}")
    return label, ckpt


def _apply_model_order(df, model_order: Sequence[str] | None):
    if not model_order:
        return df
    pd = _require_pandas()
    order_map = {label: idx for idx, label in enumerate(model_order)}
    tmp = df.copy()
    tmp["_model_order"] = tmp["model_label"].map(order_map).fillna(len(order_map))
    tmp = tmp.sort_values(["_model_order", "model_label"]).drop(columns="_model_order")
    return pd.DataFrame(tmp)


def _key_metric_specs(metrics: Sequence[Mapping[str, str]] | None) -> List[Mapping[str, str]]:
    return list(metrics or DEFAULT_KEY_METRICS)


def _wide_table_from_df(
    wide,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
    include_analysis_dir: bool = False,
):
    pd = _require_pandas()
    specs = _key_metric_specs(metrics)
    table = wide.copy()

    if eval_scores is not None:
        table["eval_score"] = table["model_label"].map(eval_scores)

    columns = ["model_label"]
    rename_map: Dict[str, str] = {}
    if include_analysis_dir and "analysis_dir" in table.columns:
        columns.append("analysis_dir")
    if "eval_score" in table.columns:
        columns.append("eval_score")

    for spec in specs:
        metric_name = spec["metric_name"]
        if metric_name not in table.columns:
            table[metric_name] = float("nan")
        columns.append(metric_name)
        rename_map[metric_name] = spec["label"]

    out = table[columns].rename(columns=rename_map)
    out = _apply_model_order(out, model_order)
    return pd.DataFrame(out)


def _long_table_from_df(
    wide,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
):
    pd = _require_pandas()
    specs = _key_metric_specs(metrics)
    table = _wide_table_from_df(
        wide,
        metrics=specs,
        eval_scores=eval_scores,
        model_order=model_order,
        include_analysis_dir=True,
    )

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        label = spec["label"]
        for _, row in table.iterrows():
            rows.append(
                {
                    "model_label": row["model_label"],
                    "analysis_dir": row.get("analysis_dir", ""),
                    "group": spec["group"],
                    "metric": label,
                    "metric_name": spec["metric_name"],
                    "better": spec["better"],
                    "value": row.get(label),
                    "eval_score": row.get("eval_score"),
                }
            )
    return pd.DataFrame(rows)


def build_metric_reference_table(
    metrics: Sequence[Mapping[str, str]] | None = None,
):
    pd = _require_pandas()
    rows = []
    for spec in _key_metric_specs(metrics):
        rows.append(
            {
                "group": spec["group"],
                "metric": spec["label"],
                "short_label": spec.get("short_label", spec["label"]),
                "better": _metric_goal_text(spec.get("better", "higher")),
                "summary": spec.get("summary", ""),
                "diagnosis": spec.get("diagnosis", ""),
            }
        )
    return pd.DataFrame(rows)


def save_metric_reference_table(
    table,
    *,
    output_dir: str | Path,
    stem: str = "key_metric_reference",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    html_path = output_dir / f"{stem}.html"
    table.to_csv(csv_path, index=False)
    html_path.write_text(table.to_html(index=False))
    return {
        "csv": csv_path,
        "html": html_path,
    }


def build_key_metric_table(
    batch_dir: str | Path,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
    include_analysis_dir: bool = False,
):
    wide = load_metrics_wide(batch_dir)
    return _wide_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
        include_analysis_dir=include_analysis_dir,
    )


def build_key_metric_long_table(
    batch_dir: str | Path,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
):
    wide = load_metrics_wide(batch_dir)
    return _long_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
    )


def records_to_metrics_wide(records: Sequence[Mapping[str, Any]]):
    pd = _require_pandas()

    rows: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {
            "model_label": record["label"],
            "model_slug": record.get("slug", slugify(record["label"])),
            "analysis_dir": record.get("analysis_dir", ""),
            "ckpt": record["result"]["meta"]["ckpt"],
        }
        for section, key, value in _metric_entries_from_result(record["result"]):
            row[f"{section}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def records_to_metrics_long(records: Sequence[Mapping[str, Any]]):
    pd = _require_pandas()
    wide = records_to_metrics_wide(records)
    rows: List[Dict[str, Any]] = []
    for _, row in wide.iterrows():
        for key, value in row.items():
            if "." not in key or key in {"analysis_dir"}:
                continue
            section, metric = key.split(".", 1)
            rows.append(
                {
                    "model_label": row["model_label"],
                    "analysis_dir": row.get("analysis_dir", ""),
                    "metric_name": key,
                    "section": section,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def build_key_metric_table_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
    include_analysis_dir: bool = False,
):
    wide = records_to_metrics_wide(records)
    return _wide_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
        include_analysis_dir=include_analysis_dir,
    )


def build_key_metric_long_table_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
):
    wide = records_to_metrics_wide(records)
    return _long_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
    )


def save_key_metric_tables(
    wide_table,
    long_table,
    *,
    output_dir: str | Path,
    stem: str = "key_metrics",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = output_dir / f"{stem}_wide.csv"
    long_csv = output_dir / f"{stem}_long.csv"
    html_path = output_dir / f"{stem}_wide.html"

    wide_table.to_csv(wide_csv, index=False)
    long_table.to_csv(long_csv, index=False)
    html_path.write_text(wide_table.to_html(index=False))
    return {
        "wide_csv": wide_csv,
        "long_csv": long_csv,
        "wide_html": html_path,
    }


def _normalize_metric(values, better: str):
    clean = [float(v) for v in values if v == v]
    if not clean:
        return [float("nan")] * len(values)

    if better == "lower_abs":
        transformed = [abs(float(v)) if v == v else float("nan") for v in values]
        clean = [v for v in transformed if v == v]
        lo, hi = min(clean), max(clean)
        if math.isclose(lo, hi):
            return [0.5 if v == v else float("nan") for v in transformed]
        return [1.0 - ((v - lo) / (hi - lo)) if v == v else float("nan") for v in transformed]

    lo, hi = min(clean), max(clean)
    if math.isclose(lo, hi):
        return [0.5 if v == v else float("nan") for v in values]

    if better == "lower":
        return [1.0 - ((float(v) - lo) / (hi - lo)) if v == v else float("nan") for v in values]

    return [((float(v) - lo) / (hi - lo)) if v == v else float("nan") for v in values]


def _metric_arrow(better: str) -> str:
    if better in {"lower", "lower_abs"}:
        return "↓"
    return "↑"


def _metric_goal_text(better: str) -> str:
    if better == "lower":
        return "lower is better"
    if better == "lower_abs":
        return "closer to 0 is better"
    return "higher is better"


def _metric_title(spec: Mapping[str, str]) -> str:
    short_label = spec.get("short_label", spec["label"])
    brief = spec.get("brief", spec.get("summary", ""))
    return (
        f"{spec['group']} | {short_label} {_metric_arrow(spec.get('better', 'higher'))}\n"
        f"{brief}"
    )


def _metric_note_lines(specs: Sequence[Mapping[str, str]], *, notes_per_line: int = 2) -> List[str]:
    entries = [
        (
            f"{spec.get('short_label', spec['label'])} {_metric_arrow(spec.get('better', 'higher'))}: "
            f"{spec.get('summary', '')} "
            f"{spec.get('diagnosis', '')}"
        ).strip()
        for spec in specs
    ]
    lines: List[str] = []
    for idx in range(0, len(entries), notes_per_line):
        lines.append("  |  ".join(entries[idx : idx + notes_per_line]))
    return lines


def _apply_metric_notes(fig, specs: Sequence[Mapping[str, str]]):
    lines = _metric_note_lines(specs)
    if not lines:
        return
    bottom_margin = min(0.38, 0.08 + 0.045 * len(lines))
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        family="monospace",
    )


def _metric_entries_from_result(result: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    entries: List[Tuple[str, str, Any]] = []
    for section in METRIC_SECTION_ORDER:
        section_value = result.get(section)
        if not isinstance(section_value, Mapping):
            continue
        for key, value in section_value.items():
            entries.append((section, key, value))
    return entries


def plot_metric_bars(
    table,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    output: str | Path | None = None,
    ncols: int = 3,
    annotate: bool = True,
    figsize_scale: float = 4.0,
):
    plt = _require_matplotlib()
    specs = [spec for spec in _key_metric_specs(metrics) if spec["label"] in table.columns]
    if not specs:
        raise ValueError("No requested metrics are present in the table.")
    nplots = len(specs)
    ncols = max(1, min(ncols, nplots))
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    x_labels = table["model_label"].tolist()
    for ax, spec in zip(axes_flat, specs):
        col = spec["label"]
        values = table[col].tolist()
        bars = ax.bar(x_labels, values, color="#4C78A8")
        ax.set_title(_metric_title(spec))
        ax.tick_params(axis="x", rotation=30)
        if annotate:
            for bar, value in zip(bars, values):
                if value == value:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    for ax in axes_flat[nplots:]:
        ax.axis("off")

    _apply_metric_notes(fig, specs)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def plot_metric_heatmap(
    table,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    output: str | Path | None = None,
    annotate: bool = True,
):
    plt = _require_matplotlib()
    specs = [spec for spec in _key_metric_specs(metrics) if spec["label"] in table.columns]
    if not specs:
        raise ValueError("No requested metrics are present in the table.")
    metric_labels = [
        f"{spec.get('short_label', spec['label'])}\n{_metric_arrow(spec.get('better', 'higher'))}"
        for spec in specs
    ]
    normalized = [
        _normalize_metric(table[spec["label"]].tolist(), spec.get("better", "higher"))
        for spec in specs
    ]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(metric_labels)), max(3, 0.7 * len(table))))
    im = ax.imshow(list(zip(*normalized)), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["model_label"].tolist())
    ax.set_title("Direction-aware metric heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if annotate:
        for row_idx in range(len(table)):
            for col_idx, spec in enumerate(specs):
                value = table.iloc[row_idx][spec["label"]]
                if value == value:
                    ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

    _apply_metric_notes(fig, specs)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def _projection_rows_from_tensor(proj, state) -> List[Dict[str, float]]:
    flat_state = state.reshape(-1, state.size(-1)).cpu()
    seq_len = state.size(1)
    rows: List[Dict[str, float]] = []
    for idx in range(proj.size(0)):
        row: Dict[str, float] = {
            "seq_id": float(idx // seq_len),
            "time_id": float(idx % seq_len),
            "x": float(proj[idx, 0]),
            "y": float(proj[idx, 1]),
        }
        for dim in range(flat_state.size(1)):
            row[f"state_{dim}"] = float(flat_state[idx, dim])
        rows.append(row)
    return rows


def _projection_rows_from_outputs(
    outputs: Mapping[str, Any],
    *,
    projection: str,
    tsne_perplexity: float = 30.0,
    seed: int = 3072,
) -> List[Dict[str, float]]:
    analyze_repr = _require_analyze_repr()
    flat_z = outputs["emb"].reshape(-1, outputs["emb"].size(-1)).cpu()
    if projection == "pca":
        proj = analyze_repr["pca_projection"](flat_z, out_dim=2)
    elif projection == "tsne":
        proj = analyze_repr["tsne_projection"](
            flat_z,
            out_dim=2,
            perplexity=tsne_perplexity,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown projection type: {projection}")
    return _projection_rows_from_tensor(proj, outputs["state"])


def _load_projection_rows(path: Path) -> List[Dict[str, float]]:
    with open(path, "r") as f:
        return json.load(f)


def plot_projection_grid(
    batch_dir: str | Path,
    *,
    model_labels: Sequence[str] | None = None,
    projection: str = "pca",
    color_dims: Sequence[int] = (0, 1),
    output: str | Path | None = None,
    alpha: float = 0.7,
    size: float = 6.0,
):
    plt = _require_matplotlib()
    by_label = analysis_dirs_by_label(batch_dir)
    model_labels = list(model_labels or by_label.keys())
    if not model_labels:
        raise ValueError("No model labels available for projection plotting.")

    rows_by_label = {
        label: _load_projection_rows(by_label[label] / f"{projection}_projection.json")
        for label in model_labels
    }
    return _plot_projection_rows_grid(
        rows_by_label,
        model_labels=model_labels,
        projection=projection,
        color_dims=color_dims,
        output=output,
        alpha=alpha,
        size=size,
    )


def plot_projection_grid_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model_labels: Sequence[str] | None = None,
    projection: str = "pca",
    color_dims: Sequence[int] = (0, 1),
    output: str | Path | None = None,
    alpha: float = 0.7,
    size: float = 6.0,
    tsne_perplexity: float = 30.0,
    seed: int = 3072,
):
    model_labels = list(model_labels or [record["label"] for record in records])
    if not model_labels:
        raise ValueError("No model labels available for projection plotting.")
    record_by_label = {record["label"]: record for record in records}
    rows_by_label = {
        label: _projection_rows_from_outputs(
            record_by_label[label]["outputs"],
            projection=projection,
            tsne_perplexity=tsne_perplexity,
            seed=seed,
        )
        for label in model_labels
    }
    return _plot_projection_rows_grid(
        rows_by_label,
        model_labels=model_labels,
        projection=projection,
        color_dims=color_dims,
        output=output,
        alpha=alpha,
        size=size,
    )


def _plot_projection_rows_grid(
    rows_by_label: Mapping[str, List[Mapping[str, float]]],
    *,
    model_labels: Sequence[str],
    projection: str,
    color_dims: Sequence[int],
    output: str | Path | None = None,
    alpha: float = 0.7,
    size: float = 6.0,
):
    plt = _require_matplotlib()

    fig, axes = plt.subplots(
        len(model_labels),
        len(color_dims),
        figsize=(6 * len(color_dims), 4 * len(model_labels)),
        squeeze=False,
    )

    for col_idx, dim in enumerate(color_dims):
        key = f"state_{dim}"
        values = [row[key] for label in model_labels for row in rows_by_label[label]]
        vmin, vmax = min(values), max(values)

        for row_idx, label in enumerate(model_labels):
            rows = rows_by_label[label]
            ax = axes[row_idx][col_idx]
            sc = ax.scatter(
                [row["x"] for row in rows],
                [row["y"] for row in rows],
                c=[row[key] for row in rows],
                s=size,
                alpha=alpha,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label} | {projection.upper()} | {key}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def run_model_analysis(
    *,
    label: str,
    ckpt: str,
    dataset: str,
    state_key: str = "proprio",
    frameskip: int = 5,
    img_size: int = 224,
    n_sequences: int = 128,
    max_points: int = 512,
    knn_k: int = 10,
    action_trials: int = 8,
    interp_steps: int = 9,
    perturb_scale: float = 1.0,
    seed: int = 3072,
    device: str | None = None,
    save_dir: str | Path | None = None,
    export_tsne: bool = False,
    tsne_perplexity: float = 30.0,
    log=print,
):
    device = device or _default_device()
    analyze_repr = _require_analyze_repr()
    result, outputs = analyze_repr["run_analysis"](
        ckpt=ckpt,
        dataset=dataset,
        state_key=state_key,
        frameskip=frameskip,
        img_size=img_size,
        n_sequences=n_sequences,
        max_points=max_points,
        knn_k=knn_k,
        action_trials=action_trials,
        interp_steps=interp_steps,
        perturb_scale=perturb_scale,
        seed=seed,
        device=device,
        log=log,
    )

    analysis_dir = None
    if save_dir is not None:
        analysis_dir = Path(save_dir)
        analyze_repr["save_outputs"](
            analysis_dir,
            result,
            outputs["emb"],
            outputs["state"],
            export_tsne=export_tsne,
            tsne_perplexity=tsne_perplexity,
            seed=seed,
        )
        if log is not None:
            log(f"[notebook_compare] saved analysis dir: {analysis_dir}")

    report = analyze_repr["format_analysis_report"](result)
    return {
        "label": label,
        "slug": slugify(label),
        "analysis_dir": str(analysis_dir) if analysis_dir is not None else "",
        "result": result,
        "outputs": outputs,
        "report": report,
    }


def run_notebook_batch(
    model_specs: Sequence[str | Sequence[str]],
    *,
    dataset: str,
    state_key: str = "proprio",
    frameskip: int = 5,
    img_size: int = 224,
    n_sequences: int = 128,
    max_points: int = 512,
    knn_k: int = 10,
    action_trials: int = 8,
    interp_steps: int = 9,
    perturb_scale: float = 1.0,
    seed: int = 3072,
    device: str | None = None,
    save_dir: str | Path | None = None,
    export_tsne: bool = False,
    tsne_perplexity: float = 30.0,
    log=print,
):
    device = device or _default_device()
    records: List[Dict[str, Any]] = []
    used_slugs: set[str] = set()
    save_root = Path(save_dir) if save_dir is not None else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)

    parsed_specs = [parse_model_spec(spec) for spec in model_specs]
    if not parsed_specs:
        raise ValueError("run_notebook_batch requires at least one model spec.")
    total = len(parsed_specs)
    for index, (label, ckpt) in enumerate(parsed_specs, start=1):
        slug = make_unique_slug(label, used_slugs)
        model_dir = save_root / slug if save_root is not None else None
        if log is not None:
            log(f"[notebook_compare] ({index}/{total}) {label}")
        record = run_model_analysis(
            label=label,
            ckpt=ckpt,
            dataset=dataset,
            state_key=state_key,
            frameskip=frameskip,
            img_size=img_size,
            n_sequences=n_sequences,
            max_points=max_points,
            knn_k=knn_k,
            action_trials=action_trials,
            interp_steps=interp_steps,
            perturb_scale=perturb_scale,
            seed=seed,
            device=device,
            save_dir=model_dir,
            export_tsne=export_tsne,
            tsne_perplexity=tsne_perplexity,
            log=log,
        )
        record["slug"] = slug
        record["analysis_dir"] = str(model_dir) if model_dir is not None else ""
        records.append(record)
    return records
