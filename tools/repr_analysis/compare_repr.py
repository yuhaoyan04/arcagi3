"""
compare_repr.py - Side-by-side comparison for two representation analysis runs.

Reads two analysis directories produced by analyze_repr.py and renders a
side-by-side scatter plot for quick comparison, e.g.:
  - SIGReg vs BN+uniformity
  - baseline vs new loss

Typical usage:

  python compare_repr.py \
      --left-dir /tmp/repr_sigreg \
      --right-dir /tmp/repr_bn_uniform \
      --projection tsne \
      --output /tmp/repr_compare_tsne.png \
      --color-dims 0 1 \
      --left-label SIGReg \
      --right-label BN+uniformity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(analysis_dir: Path, projection: str):
    file_name = f"{projection}_projection.json"
    path = analysis_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_summary(analysis_dir: Path):
    path = analysis_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def make_caption(summary: dict) -> str:
    if not summary:
        return ""
    parts = []
    meta = summary.get("meta", {})
    topo = summary.get("topology", {})
    dyn = summary.get("dynamics", {})
    if meta.get("dataset"):
        parts.append(f"dataset={meta['dataset']}")
    if "distance_corr" in topo:
        parts.append(f"dist_corr={topo['distance_corr']:.3f}")
    if "knn_overlap" in topo:
        parts.append(f"knn={topo['knn_overlap']:.3f}")
    if "latent_state_step_corr" in dyn:
        parts.append(f"dyn_corr={dyn['latent_state_step_corr']:.3f}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=str, required=True)
    parser.add_argument("--right-dir", type=str, required=True)
    parser.add_argument("--left-label", type=str, default="Left")
    parser.add_argument("--right-label", type=str, default="Right")
    parser.add_argument("--projection", type=str, default="tsne", choices=["pca", "tsne"])
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions to color by")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "compare_repr.py requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    left_rows = load_rows(left_dir, args.projection)
    right_rows = load_rows(right_dir, args.projection)
    left_summary = load_summary(left_dir)
    right_summary = load_summary(right_dir)

    ncols = len(args.color_dims)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10), squeeze=False)

    for col, dim in enumerate(args.color_dims):
        key = f"state_{dim}"
        if key not in left_rows[0] or key not in right_rows[0]:
            raise KeyError(f"{key} not found in both projection files")

        left_x = [row["x"] for row in left_rows]
        left_y = [row["y"] for row in left_rows]
        left_c = [row[key] for row in left_rows]

        right_x = [row["x"] for row in right_rows]
        right_y = [row["y"] for row in right_rows]
        right_c = [row[key] for row in right_rows]

        vmin = min(min(left_c), min(right_c))
        vmax = max(max(left_c), max(right_c))

        ax_l = axes[0][col]
        ax_r = axes[1][col]

        sc_l = ax_l.scatter(
            left_x, left_y, c=left_c, s=args.size, alpha=args.alpha, cmap="viridis", vmin=vmin, vmax=vmax
        )
        sc_r = ax_r.scatter(
            right_x, right_y, c=right_c, s=args.size, alpha=args.alpha, cmap="viridis", vmin=vmin, vmax=vmax
        )

        ax_l.set_title(f"{args.left_label} | {args.projection.upper()} | {key}")
        ax_r.set_title(f"{args.right_label} | {args.projection.upper()} | {key}")
        ax_l.set_xlabel("x")
        ax_l.set_ylabel("y")
        ax_r.set_xlabel("x")
        ax_r.set_ylabel("y")

        cap_l = make_caption(left_summary)
        cap_r = make_caption(right_summary)
        if cap_l:
            ax_l.text(0.01, -0.12, cap_l, transform=ax_l.transAxes, fontsize=9, va="top")
        if cap_r:
            ax_r.text(0.01, -0.12, cap_r, transform=ax_r.transAxes, fontsize=9, va="top")

        fig.colorbar(sc_l, ax=[ax_l, ax_r], fraction=0.03, pad=0.02)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[compare_repr] saved to {out}")


if __name__ == "__main__":
    main()
