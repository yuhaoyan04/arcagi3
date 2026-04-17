"""
plot_repr.py - Plot PCA / t-SNE exports from analyze_repr.py.

This script turns:
  - pca_projection.json
  - tsne_projection.json

into simple scatter plots that are easy to compare across checkpoints, e.g.
SIGReg vs BN+uniformity.

Typical usage:

  python plot_repr.py \
      --input /tmp/repr_analysis_pusht/tsne_projection.json \
      --output /tmp/repr_analysis_pusht/tsne_state0_state1.png \
      --color-dims 0 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Projection JSON from analyze_repr.py")
    parser.add_argument("--output", type=str, required=True, help="Output image path, e.g. out.png")
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions to color by")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_repr.py requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    rows = load_rows(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    x = [row["x"] for row in rows]
    y = [row["y"] for row in rows]

    n_panels = len(args.color_dims)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, args.color_dims):
        key = f"state_{dim}"
        if key not in rows[0]:
            raise KeyError(f"{key} not found in projection file.")
        c = [row[key] for row in rows]
        sc = ax.scatter(x, y, c=c, s=args.size, alpha=args.alpha, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{Path(args.input).stem} colored by {key}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[plot_repr] saved to {out}")


if __name__ == "__main__":
    main()
