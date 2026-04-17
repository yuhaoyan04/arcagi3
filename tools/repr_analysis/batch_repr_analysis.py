"""
batch_repr_analysis.py - Run representation analysis for multiple checkpoints.

This is the batch version of analyze_repr.py. It runs the same quantitative
analysis for several models, saves each model's usual output directory, writes
aggregate tables, and can optionally render pairwise comparison plots.

Typical usage:

  python -m tools.repr_analysis.batch_repr_analysis \
      --dataset /path/to/pusht_expert_train \
      --state-key proprio \
      --save-dir /tmp/repr_batch_pusht \
      --plot-projections pca \
      --compare-projections pca \
      --color-dims 0 1 \
      --model "SIGReg=/path/to/lewm.ckpt" \
      --model "BN+uniformity=/path/to/swm.ckpt"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from tools.repr_analysis.analyze_repr import (
    METRIC_SPECS,
    SECTION_ORDER,
    SECTION_TITLES,
    format_analysis_report,
    format_metric_value,
    metric_entries,
    metric_spec,
    run_analysis,
    save_metric_guide,
    save_outputs,
    to_serializable,
)
from tools.repr_analysis.compare_repr import load_rows as load_compare_rows
from tools.repr_analysis.compare_repr import load_summary, save_comparison_plot
from tools.repr_analysis.plot_repr import load_rows, save_projection_plot


class TeeLogger:
    """Mirror progress to stdout and to an in-memory log file."""

    def __init__(self):
        self.lines: List[str] = []

    def log(self, message: str):
        print(message)
        self.lines.append(message)

    def block(self, message: str):
        print(f"\n{message}")
        self.lines.append(message)

    def save(self, path: Path):
        text = "\n".join(self.lines).rstrip() + "\n"
        path.write_text(text)


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse `label=ckpt_path`; if label is omitted, derive one from the path."""
    if "=" in spec:
        label, ckpt = spec.split("=", 1)
        label = label.strip()
        ckpt = ckpt.strip()
        if not label or not ckpt:
            raise ValueError(f"Invalid --model spec: {spec}")
        return label, ckpt

    ckpt = spec.strip()
    if not ckpt:
        raise ValueError("Empty --model spec.")
    return Path(ckpt).stem, ckpt


def parse_compare_pair(spec: str) -> Tuple[str, str]:
    """Parse `left=right` or `left,right` pair spec using model labels."""
    if "=" in spec:
        left, right = spec.split("=", 1)
    elif "," in spec:
        left, right = spec.split(",", 1)
    else:
        raise ValueError(f"Invalid --compare-pair spec: {spec}")

    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Invalid --compare-pair spec: {spec}")
    return left, right


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


def projection_plot_name(projection: str, color_dims: List[int]) -> str:
    dims = "_".join(f"state{dim}" for dim in color_dims)
    return f"{projection}_{dims}.png"


def compare_plot_name(left_slug: str, right_slug: str, projection: str, color_dims: List[int]) -> str:
    dims = "_".join(f"state{dim}" for dim in color_dims)
    return f"{left_slug}__{right_slug}_{projection}_{dims}.png"


def write_wide_csv(path: Path, rows: List[Dict[str, Any]]):
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        raise ValueError("No rows to save in long CSV.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_cell(value: Any) -> str:
    return format_metric_value(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_report(path: Path, model_records: List[Dict[str, Any]]):
    labels = [record["label"] for record in model_records]
    lines = [
        "# Representation Batch Analysis",
        "",
        "This file compares the scalar metrics from multiple representation-analysis runs.",
        "",
        "## Models",
        "",
        "| label | slug | analysis_dir | ckpt |",
        "| --- | --- | --- | --- |",
    ]
    for record in model_records:
        lines.append(
            f"| {markdown_cell(record['label'])} | {markdown_cell(record['slug'])} | "
            f"{markdown_cell(record['analysis_dir'])} | {markdown_cell(record['result']['meta']['ckpt'])} |"
        )

    for section in SECTION_ORDER:
        if section == "meta":
            continue
        if not any(section in record["result"] for record in model_records):
            continue

        lines.extend([
            "",
            f"## {SECTION_TITLES.get(section, section.title())}",
            "",
            "| metric | better | summary | " + " | ".join(markdown_cell(label) for label in labels) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in labels) + " |",
        ])

        metric_keys: List[str] = []
        for record in model_records:
            metrics = record["result"].get(section, {})
            if not isinstance(metrics, dict):
                continue
            preferred_keys = list(METRIC_SPECS.get(section, {}).keys())
            extra_keys = [key for key in metrics.keys() if key not in preferred_keys]
            for key in preferred_keys + extra_keys:
                if key in metrics and key not in metric_keys:
                    metric_keys.append(key)

        for key in metric_keys:
            spec = metric_spec(section, key)
            row = [
                markdown_cell(key),
                markdown_cell(spec.get("better", "context")),
                markdown_cell(spec.get("summary", "")),
            ]
            for record in model_records:
                value = record["result"].get(section, {}).get(key, "")
                row.append(markdown_cell(value))
            lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines).rstrip() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="Repeatable `label=ckpt_path` entry.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name/path, e.g. pusht_expert_train")
    parser.add_argument("--state-key", type=str, default="proprio")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-sequences", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--action-trials", type=int, default=8)
    parser.add_argument("--interp-steps", type=int, default=9)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--save-dir", type=str, required=True, help="Root output directory for the batch run.")
    parser.add_argument("--export-tsne", action="store_true", help="Also export t-SNE projections for each model.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--plot-projections",
        type=str,
        nargs="*",
        default=[],
        choices=["pca", "tsne"],
        help="Projection types to render as per-model PNG files.",
    )
    parser.add_argument(
        "--compare-projections",
        type=str,
        nargs="*",
        default=[],
        choices=["pca", "tsne"],
        help="Projection types to render as pairwise comparison PNG files.",
    )
    parser.add_argument(
        "--compare-pair",
        action="append",
        default=[],
        help="Optional pair spec using model labels, e.g. `SIGReg=BN+uniformity`. Repeatable.",
    )
    parser.add_argument(
        "--compare-all-pairs",
        action="store_true",
        help="If set, render comparison plots for every model pair.",
    )
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions used for plot colors.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def resolve_compare_pairs(
    args: argparse.Namespace,
    model_records: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_label = {record["label"]: record for record in model_records}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[Tuple[str, str]] = set()

    if args.compare_all_pairs:
        requested = [(left["label"], right["label"]) for left, right in itertools.combinations(model_records, 2)]
    elif args.compare_pair:
        requested = [parse_compare_pair(spec) for spec in args.compare_pair]
    elif len(model_records) == 2 and args.compare_projections:
        requested = [(model_records[0]["label"], model_records[1]["label"])]
    else:
        requested = []

    for left_label, right_label in requested:
        if left_label not in by_label:
            raise KeyError(f"Unknown compare pair label: {left_label}")
        if right_label not in by_label:
            raise KeyError(f"Unknown compare pair label: {right_label}")
        if left_label == right_label:
            raise ValueError(f"Compare pair must use two different labels: {left_label}")

        left = by_label[left_label]
        right = by_label[right_label]
        pair_key = tuple(sorted((left["slug"], right["slug"])))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        pairs.append((left, right))

    return pairs


def render_pairwise_comparisons(
    *,
    args: argparse.Namespace,
    root_dir: Path,
    model_records: List[Dict[str, Any]],
    logger: TeeLogger,
) -> List[Dict[str, str]]:
    compare_pairs = resolve_compare_pairs(args, model_records)
    if not compare_pairs or not args.compare_projections:
        return []

    compare_dir = root_dir / "pairwise_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    manifests: List[Dict[str, str]] = []

    for left, right in compare_pairs:
        for projection in args.compare_projections:
            left_rows = load_compare_rows(Path(left["analysis_dir"]), projection)
            right_rows = load_compare_rows(Path(right["analysis_dir"]), projection)
            left_summary = load_summary(Path(left["analysis_dir"]))
            right_summary = load_summary(Path(right["analysis_dir"]))

            output = compare_dir / compare_plot_name(left["slug"], right["slug"], projection, args.color_dims)
            save_comparison_plot(
                left_rows=left_rows,
                right_rows=right_rows,
                left_summary=left_summary,
                right_summary=right_summary,
                left_label=left["label"],
                right_label=right["label"],
                projection=projection,
                color_dims=args.color_dims,
                output=output,
                title=f"{args.dataset} Representation Comparison",
                alpha=args.alpha,
                size=args.size,
            )
            logger.log(
                f"[batch_repr_analysis] saved compare plot: {output} "
                f"({left['label']} vs {right['label']}, {projection})"
            )
            manifests.append({
                "left_label": left["label"],
                "right_label": right["label"],
                "left_slug": left["slug"],
                "right_slug": right["slug"],
                "projection": projection,
                "output": str(output),
            })

    with open(compare_dir / "compare_manifest.json", "w") as f:
        json.dump(to_serializable(manifests), f, indent=2)
    return manifests


def main():
    parser = build_parser()
    args = parser.parse_args()

    if "tsne" in args.plot_projections and not args.export_tsne:
        parser.error("`--plot-projections tsne` requires `--export-tsne`.")
    if "tsne" in args.compare_projections and not args.export_tsne:
        parser.error("`--compare-projections tsne` requires `--export-tsne`.")
    if args.compare_pair and not args.compare_projections:
        parser.error("`--compare-pair` requires `--compare-projections`.")
    if args.compare_all_pairs and not args.compare_projections:
        parser.error("`--compare-all-pairs` requires `--compare-projections`.")

    root_dir = Path(args.save_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    save_metric_guide(root_dir / "metric_guide.json")

    logger = TeeLogger()
    model_specs = [parse_model_spec(spec) for spec in args.model]
    used_slugs: set[str] = set()
    model_records: List[Dict[str, Any]] = []

    logger.log(f"[batch_repr_analysis] models={len(model_specs)} save_dir={root_dir}")
    for index, (label, ckpt) in enumerate(model_specs, start=1):
        slug = make_unique_slug(label, used_slugs)
        model_dir = root_dir / slug

        logger.log(f"\n[batch_repr_analysis] ({index}/{len(model_specs)}) {label}")
        result, outputs = run_analysis(
            ckpt=ckpt,
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
            log=logger.log,
        )

        save_outputs(
            model_dir,
            result,
            outputs["emb"],
            outputs["state"],
            export_tsne=args.export_tsne,
            tsne_perplexity=args.tsne_perplexity,
            seed=args.seed,
        )
        logger.log(f"[batch_repr_analysis] saved analysis dir: {model_dir}")

        for projection in args.plot_projections:
            projection_json = model_dir / f"{projection}_projection.json"
            if not projection_json.exists():
                logger.log(f"[batch_repr_analysis] skip {projection} plot for {label}: {projection_json.name} not found")
                continue

            plot_path = model_dir / projection_plot_name(projection, args.color_dims)
            save_projection_plot(
                load_rows(projection_json),
                output=plot_path,
                color_dims=args.color_dims,
                title=f"{label} {projection.upper()}",
                alpha=args.alpha,
                size=args.size,
            )
            logger.log(f"[batch_repr_analysis] saved {projection} plot: {plot_path}")

        logger.block(format_analysis_report(result))
        model_records.append({
            "label": label,
            "slug": slug,
            "analysis_dir": str(model_dir),
            "result": result,
        })

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for record in model_records:
        wide_row: Dict[str, Any] = {
            "model_label": record["label"],
            "model_slug": record["slug"],
            "analysis_dir": record["analysis_dir"],
        }
        for section, key, value in metric_entries(record["result"]):
            metric_name = f"{section}.{key}"
            wide_row[metric_name] = value
            spec = metric_spec(section, key)
            long_rows.append({
                "model_label": record["label"],
                "model_slug": record["slug"],
                "analysis_dir": record["analysis_dir"],
                "section": section,
                "metric": key,
                "metric_name": metric_name,
                "value": value,
                "better": spec.get("better", "context"),
                "summary": spec.get("summary", ""),
                "use": spec.get("use", ""),
            })
        wide_rows.append(wide_row)

    compare_manifest = render_pairwise_comparisons(
        args=args,
        root_dir=root_dir,
        model_records=model_records,
        logger=logger,
    )

    wide_csv = root_dir / "metrics_wide.csv"
    long_csv = root_dir / "metrics_long.csv"
    markdown_report = root_dir / "metrics_table.md"
    batch_summary_json = root_dir / "batch_summary.json"
    batch_log = root_dir / "batch_report.txt"

    write_wide_csv(wide_csv, wide_rows)
    write_long_csv(long_csv, long_rows)
    write_markdown_report(markdown_report, model_records)
    with open(batch_summary_json, "w") as f:
        json.dump(to_serializable({"models": model_records, "compare_manifest": compare_manifest}), f, indent=2)

    logger.log(f"\n[batch_repr_analysis] saved wide table: {wide_csv}")
    logger.log(f"[batch_repr_analysis] saved long table: {long_csv}")
    logger.log(f"[batch_repr_analysis] saved markdown table: {markdown_report}")
    if compare_manifest:
        logger.log(f"[batch_repr_analysis] saved pairwise compares: {root_dir / 'pairwise_compare'}")
    logger.log(f"[batch_repr_analysis] saved batch summary: {batch_summary_json}")
    logger.save(batch_log)
    print(f"[batch_repr_analysis] saved batch log: {batch_log}")


if __name__ == "__main__":
    main()
