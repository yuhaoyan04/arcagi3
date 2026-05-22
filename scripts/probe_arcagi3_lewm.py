#!/usr/bin/env python3
"""Train coordinate probes on LeWM representations for ARC-AGI-3 replays."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ALL_LAYERS = ["low", "mid", "high"]
ALL_PROBES = ["linear", "mlp"]
ALL_INITS = ["trained", "random"]


@dataclass
class SplitInfo:
    train_levels: list[int]
    val_levels: list[int]
    test_levels: list[int]
    train_size: int
    val_size: int
    test_size: int
    split_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe LeWM latents for player coordinates.")
    parser.add_argument("--h5", type=Path, default=Path("data/processed/arcagi3_human_replay.h5"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="LeWM object checkpoint saved by train.py, e.g. *_object.ckpt.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/arcagi3/probe"))
    parser.add_argument("--layers", default="low,mid,high", help="Comma list or all.")
    parser.add_argument("--probe-types", default="linear,mlp", help="Comma list or all.")
    parser.add_argument("--lewm-inits", default="trained,random", help="trained, random, both, or comma list.")
    parser.add_argument("--target-key", default="proprio", help="HDF5 target column. Default: proprio.")
    parser.add_argument("--split-key", default="level_uid", help="HDF5 column used for level-heldout splits.")
    parser.add_argument("--train-levels", default="", help="Explicit comma-separated level ids for train.")
    parser.add_argument("--val-levels", default="", help="Explicit comma-separated level ids for validation.")
    parser.add_argument("--test-levels", default="", help="Explicit comma-separated level ids for test.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-hidden-size", type=int, choices=[128, 256], default=128)
    parser.add_argument("--loss", choices=["mse", "cross_entropy"], default="mse")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--filter-lost", action="store_true", default=True)
    parser.add_argument("--keep-lost", dest="filter_lost", action="store_false")
    parser.add_argument("--save-heatmap", action="store_true", help="Save an optional RMSE heatmap PNG.")
    parser.add_argument("--swanlab", action="store_true", default=True)
    parser.add_argument("--no-swanlab", dest="swanlab", action="store_false")
    parser.add_argument("--swanlab-project", default="arcagi3-probe")
    parser.add_argument("--swanlab-workspace", default="")
    parser.add_argument("--swanlab-logdir", default="outputs/arcagi3/swanlab")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def parse_choice_list(value: str, all_values: list[str], both_alias: bool = False) -> list[str]:
    value = value.strip().lower()
    if value == "all" or (both_alias and value == "both"):
        return list(all_values)
    out = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in out if item not in all_values]
    if invalid:
        raise ValueError(f"Invalid values {invalid}; allowed: {all_values}")
    return out


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SwanLabSink:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]):
        self.enabled = bool(args.swanlab)
        self.swanlab = None
        if not self.enabled:
            return
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("swanlab is required unless --no-swanlab is set.") from exc

        self.swanlab = swanlab
        kwargs = {
            "project": args.swanlab_project,
            "experiment_name": args.run_name or "arcagi3_probe",
            "config": config,
            "logdir": args.swanlab_logdir,
        }
        if args.swanlab_workspace:
            kwargs["workspace"] = args.swanlab_workspace
        swanlab.init(**kwargs)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled and self.swanlab is not None:
            self.swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self.swanlab is not None and hasattr(self.swanlab, "finish"):
            self.swanlab.finish()


class H5FrameDataset(Dataset):
    def __init__(self, h5_path: Path, indices: np.ndarray, target_key: str):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_key = target_key
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        idx = int(self.indices[item])
        pixels = np.asarray(h5["pixels"][idx])
        target = np.asarray(h5[self.target_key][idx], dtype=np.float32).reshape(-1)
        return {
            "pixels": torch.from_numpy(pixels),
            "target": torch.from_numpy(target),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def valid_target_mask(h5: h5py.File, target_key: str, filter_lost: bool) -> np.ndarray:
    target = np.asarray(h5[target_key][:], dtype=np.float32).reshape(len(h5[target_key]), -1)
    mask = np.isfinite(target).all(axis=1)
    if filter_lost and "player_lost" in h5:
        lost = np.asarray(h5["player_lost"][:]).reshape(-1)
        mask &= lost == 0
    return mask


def split_indices(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], SplitInfo]:
    with h5py.File(args.h5, "r") as h5:
        if args.target_key not in h5:
            raise KeyError(f"Target key {args.target_key!r} not found in {args.h5}")
        if args.split_key in h5:
            level_values = np.asarray(h5[args.split_key][:]).reshape(-1).astype(np.int64)
            split_key = args.split_key
        else:
            lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
            level_values = np.concatenate(
                [np.full(length, ep_idx, dtype=np.int64) for ep_idx, length in enumerate(lengths)]
            )
            split_key = "episode_index"

        valid_mask = valid_target_mask(h5, args.target_key, args.filter_lost)

    unique_levels = sorted(int(x) for x in np.unique(level_values[valid_mask]))
    train_levels = parse_int_list(args.train_levels)
    val_levels = parse_int_list(args.val_levels)
    test_levels = parse_int_list(args.test_levels)

    if not (train_levels or val_levels or test_levels):
        rng = np.random.default_rng(args.seed)
        shuffled = np.asarray(unique_levels, dtype=np.int64)
        rng.shuffle(shuffled)
        n_levels = len(shuffled)
        if n_levels < 3:
            raise ValueError("Need at least 3 levels for train/val/test level-heldout splits.")
        n_train = max(1, int(round(n_levels * args.train_ratio)))
        n_val = max(1, int(round(n_levels * args.val_ratio)))
        if n_train + n_val >= n_levels:
            n_train = max(1, n_levels - 2)
            n_val = 1
        train_levels = sorted(int(x) for x in shuffled[:n_train])
        val_levels = sorted(int(x) for x in shuffled[n_train : n_train + n_val])
        test_levels = sorted(int(x) for x in shuffled[n_train + n_val :])
    else:
        assigned = set(train_levels) | set(val_levels) | set(test_levels)
        missing = [lvl for lvl in unique_levels if lvl not in assigned]
        if missing:
            train_levels = sorted(set(train_levels) | set(missing))
        if not train_levels or not val_levels or not test_levels:
            raise ValueError("Explicit splits must provide non-empty train, val, and test levels.")

    def select(levels: list[int]) -> np.ndarray:
        mask = valid_mask & np.isin(level_values, np.asarray(levels, dtype=np.int64))
        return np.nonzero(mask)[0].astype(np.int64)

    splits = {
        "train": select(train_levels),
        "val": select(val_levels),
        "test": select(test_levels),
    }
    info = SplitInfo(
        train_levels=train_levels,
        val_levels=val_levels,
        test_levels=test_levels,
        train_size=int(len(splits["train"])),
        val_size=int(len(splits["val"])),
        test_size=int(len(splits["test"])),
        split_key=split_key,
    )
    if min(info.train_size, info.val_size, info.test_size) <= 0:
        raise ValueError(f"Empty split after filtering: {info}")
    return splits, info


def make_loader(
    h5_path: Path,
    indices: np.ndarray,
    target_key: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = H5FrameDataset(h5_path, indices, target_key)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def preprocess_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    pixels = pixels.float()
    if pixels.ndim != 4:
        raise ValueError(f"Expected pixels with 4 dims, got {tuple(pixels.shape)}")
    if pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(0, 3, 1, 2)
    if pixels.shape[1] == 1:
        pixels = pixels.repeat(1, 3, 1, 1)
    if pixels.max() > 2.0:
        pixels = pixels / 255.0
    if pixels.shape[-2:] != (image_size, image_size):
        pixels = F.interpolate(pixels, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def scan_lewm_model(payload: Any) -> nn.Module | None:
    if isinstance(payload, nn.Module):
        if hasattr(payload, "encode") and hasattr(payload, "encoder"):
            return payload
        if hasattr(payload, "model"):
            found = scan_lewm_model(payload.model)
            if found is not None:
                return found
        for child in payload.children():
            found = scan_lewm_model(child)
            if found is not None:
                return found
    if isinstance(payload, dict):
        for key in ("model", "module", "state"):
            if key in payload:
                found = scan_lewm_model(payload[key])
                if found is not None:
                    return found
    return None


def load_lewm(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = scan_lewm_model(payload)
    if model is None:
        raise RuntimeError(
            f"No serialized LeWM object with encode()/encoder found in {checkpoint}. "
            "Use the *_object.ckpt produced by train.py."
        )
    return model.to(device).eval()


def reset_for_random_baseline(model: nn.Module, seed: int) -> nn.Module:
    seed_everything(seed)
    random_model = copy.deepcopy(model).cpu()

    def reset_module(module: nn.Module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            return
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                if param.ndim > 1:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(param)

    random_model.apply(reset_module)
    return random_model


def extract_latents(model: nn.Module, pixels: torch.Tensor, layer: str) -> torch.Tensor:
    if layer == "high":
        out = model.encode({"pixels": pixels.unsqueeze(1)})
        return out["emb"][:, 0]

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError("Loaded model has no encoder attribute.")

    try:
        out = encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
            return_dict=True,
        )
    except TypeError:
        out = encoder(pixels, output_hidden_states=True, return_dict=True)

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None and isinstance(out, dict):
        hidden_states = out.get("hidden_states")
    if hidden_states is None:
        raise RuntimeError("Encoder did not return hidden_states; low/mid probes require ViT hidden states.")

    if layer == "low":
        idx = 1 if len(hidden_states) > 1 else 0
    elif layer == "mid":
        idx = max(1, len(hidden_states) // 2)
    else:
        raise ValueError(f"Unsupported layer: {layer}")

    features = hidden_states[idx]
    if features.ndim == 3:
        return features[:, 0]
    if features.ndim == 4:
        return features.flatten(2).mean(dim=-1)
    return features.reshape(features.shape[0], -1)


class ProbeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, probe_type: str, hidden_size: int):
        super().__init__()
        if probe_type == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif probe_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim),
            )
        else:
            raise ValueError(f"Unsupported probe type: {probe_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infer_dims(
    model: nn.Module,
    loader: DataLoader,
    layer: str,
    device: torch.device,
    image_size: int,
    loss_name: str,
) -> tuple[int, int]:
    batch = next(iter(loader))
    pixels = preprocess_pixels(batch["pixels"].to(device), image_size)
    with torch.inference_mode():
        features = extract_latents(model, pixels, layer)
    target = batch["target"]
    if loss_name == "cross_entropy":
        if target.shape[-1] != 1:
            raise ValueError("cross_entropy requires a scalar integer target.")
        output_dim = int(target.max().item()) + 1
    else:
        output_dim = int(target.reshape(target.shape[0], -1).shape[-1])
    return int(features.shape[-1]), output_dim


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target.float())
    if loss_name == "cross_entropy":
        return F.cross_entropy(pred, target.reshape(-1).long())
    raise ValueError(loss_name)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_np = pred.detach().cpu().numpy().reshape(len(pred), -1)
    target_np = target.detach().cpu().numpy().reshape(len(target), -1)
    err = pred_np - target_np
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target_np - target_np.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_cls = pred.argmax(dim=-1)
    target_cls = target.reshape(-1).long()
    acc = (pred_cls == target_cls).float().mean().item()
    return {"accuracy": float(acc)}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    probe: nn.Module,
    loader: DataLoader,
    layer: str,
    device: torch.device,
    image_size: int,
    loss_name: str,
) -> dict[str, float]:
    probe.eval()
    losses = []
    preds = []
    targets = []
    for batch in loader:
        pixels = preprocess_pixels(batch["pixels"].to(device), image_size)
        target = batch["target"].to(device).float()
        features = extract_latents(model, pixels, layer)
        pred = probe(features)
        losses.append(compute_loss(pred, target, loss_name).detach().cpu())
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = {"loss": float(torch.stack(losses).mean().item())}
    if loss_name == "mse":
        metrics.update(regression_metrics(pred_all, target_all))
    else:
        metrics.update(classification_metrics(pred_all, target_all))
    return metrics


def train_one_probe(
    *,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    layer: str,
    probe_type: str,
    init_name: str,
    args: argparse.Namespace,
    device: torch.device,
    swan: SwanLabSink,
    combo_dir: Path,
    global_combo_idx: int,
) -> dict[str, Any]:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    input_dim, output_dim = infer_dims(model, loaders["train"], layer, device, args.image_size, args.loss)
    probe = ProbeHead(input_dim, output_dim, probe_type, args.mlp_hidden_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        probe.train()
        losses = []
        for batch in loaders["train"]:
            pixels = preprocess_pixels(batch["pixels"].to(device), args.image_size)
            target = batch["target"].to(device).float()
            with torch.inference_mode():
                features = extract_latents(model, pixels, layer)
            pred = probe(features)
            loss = compute_loss(pred, target, args.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())

        train_metrics = evaluate(model, probe, loaders["train"], layer, device, args.image_size, args.loss)
        val_metrics = evaluate(model, probe, loaders["val"], layer, device, args.image_size, args.loss)
        test_metrics = evaluate(model, probe, loaders["test"], layer, device, args.image_size, args.loss)
        train_metrics["loss"] = float(torch.stack(losses).mean().item())

        row = {
            "epoch": epoch,
            "lewm_init": init_name,
            "layer": layer,
            "probe_type": probe_type,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        history.append(row)

        metric_prefix = f"{init_name}/{probe_type}/{layer}"
        swan.log(
            {f"{metric_prefix}/{k}": float(v) for k, v in row.items() if isinstance(v, (int, float))},
            step=(global_combo_idx * args.epochs) + epoch,
        )

        val_key = "val_rmse" if args.loss == "mse" else "val_loss"
        if row[val_key] < best_val:
            best_val = float(row[val_key])
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())

    assert best_state is not None
    probe.load_state_dict(best_state)
    final_metrics = {
        split: evaluate(model, probe, loader, layer, device, args.image_size, args.loss)
        for split, loader in loaders.items()
    }

    combo_dir.mkdir(parents=True, exist_ok=True)
    weights_path = combo_dir / "probe_best.pt"
    torch.save(
        {
            "probe_state_dict": best_state,
            "layer": layer,
            "probe_type": probe_type,
            "lewm_init": init_name,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "best_epoch": best_epoch,
            "args": serializable_args(args),
        },
        weights_path,
    )
    write_csv(combo_dir / "history.csv", history)

    result = {
        "lewm_init": init_name,
        "layer": layer,
        "probe_type": probe_type,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "best_epoch": best_epoch,
        "weights": str(weights_path),
        **{f"{split}_{k}": v for split, metrics in final_metrics.items() for k, v in metrics.items()},
    }
    (combo_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib is not installed; skipping heatmap.")
        return

    rows = [f"{r['lewm_init']}/{r['probe_type']}" for r in results]
    cols = ALL_LAYERS
    row_names = sorted(set(rows))
    matrix = np.full((len(row_names), len(cols)), np.nan, dtype=np.float32)
    row_index = {name: i for i, name in enumerate(row_names)}
    col_index = {name: i for i, name in enumerate(cols)}
    for r in results:
        metric = r.get("test_rmse", r.get("test_loss", np.nan))
        matrix[row_index[f"{r['lewm_init']}/{r['probe_type']}"], col_index[r["layer"]]] = metric

    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(row_names))))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(row_names)), labels=row_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_title("ARC-AGI-3 probe test error")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "probe_test_error_heatmap.png", dpi=180)
    plt.close(fig)


def write_diagnostics(results: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# Probe Diagnostics", ""]
    metric = "test_rmse" if "test_rmse" in results[0] else "test_loss"
    for init_name in sorted({r["lewm_init"] for r in results}):
        subset = [r for r in results if r["lewm_init"] == init_name]
        best = min(subset, key=lambda r: r[metric])
        lines.append(
            f"- Best {init_name}: {best['probe_type']} on {best['layer']} "
            f"({metric}={best[metric]:.6f})."
        )

    for layer in sorted({r["layer"] for r in results}):
        trained_linear = next(
            (r for r in results if r["lewm_init"] == "trained" and r["probe_type"] == "linear" and r["layer"] == layer),
            None,
        )
        trained_mlp = next(
            (r for r in results if r["lewm_init"] == "trained" and r["probe_type"] == "mlp" and r["layer"] == layer),
            None,
        )
        if trained_linear and trained_mlp and trained_mlp[metric] > trained_linear[metric]:
            train_gap = trained_mlp.get("train_rmse", trained_mlp.get("train_loss", 0.0)) - trained_linear.get(
                "train_rmse", trained_linear.get("train_loss", 0.0)
            )
            if train_gap < 0:
                reason = "MLP fits train better but generalizes worse, which points to overfitting on held-out levels."
            else:
                reason = "MLP also fails to improve train error, which points to optimization or hidden-size sensitivity."
            lines.append(f"- Layer {layer}: MLP underperforms Linear on test. {reason}")

    lines.append("")
    lines.append("Use this file as a first-pass diagnosis; confirm with multiple seeds before drawing conclusions.")
    (output_dir / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    layers = parse_choice_list(args.layers, ALL_LAYERS)
    probe_types = parse_choice_list(args.probe_types, ALL_PROBES)
    lewm_inits = parse_choice_list(args.lewm_inits, ALL_INITS, both_alias=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, split_info = split_indices(args)
    loaders = {
        split: make_loader(
            args.h5,
            indices,
            args.target_key,
            args.batch_size,
            args.num_workers,
            shuffle=(split == "train"),
        )
        for split, indices in splits.items()
    }

    config = {
        "args": serializable_args(args),
        "layers": layers,
        "probe_types": probe_types,
        "lewm_inits": lewm_inits,
        "split": asdict(split_info),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "splits.json").write_text(json.dumps(asdict(split_info), indent=2), encoding="utf-8")

    swan = SwanLabSink(args, config)
    trained_model = load_lewm(args.checkpoint, device)
    models: dict[str, nn.Module] = {}
    if "trained" in lewm_inits:
        models["trained"] = trained_model
    if "random" in lewm_inits:
        models["random"] = reset_for_random_baseline(trained_model, args.seed + 17).to(device).eval()

    results = []
    combo_idx = 0
    try:
        for init_name in lewm_inits:
            for layer in layers:
                for probe_type in probe_types:
                    combo_idx += 1
                    combo_name = f"{init_name}_{layer}_{probe_type}"
                    print(f"[probe] Running {combo_name}")
                    result = train_one_probe(
                        model=models[init_name],
                        loaders=loaders,
                        layer=layer,
                        probe_type=probe_type,
                        init_name=init_name,
                        args=args,
                        device=device,
                        swan=swan,
                        combo_dir=args.output_dir / combo_name,
                        global_combo_idx=combo_idx - 1,
                    )
                    results.append(result)
                    swan.log({f"summary/{combo_name}/test_rmse": float(result.get("test_rmse", result["test_loss"]))})
    finally:
        swan.finish()

    write_csv(args.output_dir / "comparison.csv", results)
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_diagnostics(results, args.output_dir)
    if args.save_heatmap:
        save_heatmap(results, args.output_dir)
    print(f"[probe] Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
