#!/usr/bin/env python3
"""Convert ARC-AGI-3 human replay folders to stable-worldmodel HDF5.

The produced file follows the stable-worldmodel HDF5 reader convention:
flat per-frame datasets plus `ep_len` and `ep_offset`.  Each level folder is
written as one episode so that train/val/test splits can hold out full levels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARC_AGI_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [30, 147, 255],
        [220, 50, 47],
        [64, 178, 77],
        [255, 213, 79],
        [158, 158, 158],
        [177, 64, 201],
        [255, 142, 42],
        [64, 224, 208],
        [128, 0, 32],
        [255, 255, 255],
        [96, 96, 96],
        [0, 92, 197],
        [120, 44, 138],
        [247, 247, 247],
        [44, 44, 44],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ARC-AGI-3 replay folders into stable-worldmodel HDF5."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data"),
        help="Dataset root. Can be wm_exp/data or the arcpipeline/data directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/arcagi3_human_replay.h5"),
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=["by_level", "world_model"],
        default="by_level",
        help="Source folder to convert. by_level is required for coordinate probes.",
    )
    parser.add_argument(
        "--pixel-source",
        choices=["render_grids", "frames", "png"],
        default="render_grids",
        help="Source for pixels. render_grids stores compact 64x64 RGB frames.",
    )
    parser.add_argument(
        "--action-encoding",
        choices=["onehot", "scalar"],
        default="onehot",
        help="Encode discrete actions as one-hot vectors or scalar ids.",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=None,
        help="One-hot action dimension. Defaults to max observed action id + 1.",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="HDF5 compression filter.",
    )
    parser.add_argument(
        "--skip-lost-coords",
        action="store_true",
        help="Drop frames whose coordinate extraction is marked lost.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print detected dataset statistics; do not write HDF5.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_source_root(input_root: Path, dataset_kind: str) -> Path:
    folder_name = f"dataset_{dataset_kind}" if dataset_kind != "by_level" else "dataset_by_level"
    input_root = input_root.expanduser().resolve()

    if input_root.name == folder_name:
        return input_root

    direct = input_root / folder_name
    if direct.exists():
        return direct

    candidates = sorted(p for p in input_root.rglob(folder_name) if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"Could not find {folder_name!r} under {input_root}")
    if len(candidates) > 1:
        print(f"[convert] Multiple {folder_name} roots found; using {candidates[0]}")
    return candidates[0]


def discover_episode_dirs(source_root: Path, dataset_kind: str) -> list[Path]:
    if dataset_kind == "by_level":
        dirs = sorted(
            p
            for p in source_root.glob("*/*")
            if p.is_dir() and p.name.startswith("level_") and (p / "episode.npz").exists()
        )
    else:
        dirs = sorted(p for p in source_root.iterdir() if p.is_dir() and (p / "episode.npz").exists())
    if not dirs:
        raise FileNotFoundError(f"No episode.npz files found under {source_root}")
    return dirs


def stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def parse_level_id(level_dir: Path) -> int:
    if not level_dir.name.startswith("level_"):
        return -1
    try:
        return int(level_dir.name.split("_", 1)[1])
    except ValueError:
        return -1


def grids_to_rgb(grids: np.ndarray) -> np.ndarray:
    grids = np.asarray(grids)
    if grids.ndim == 4:
        grids = grids[:, 0]
    if grids.ndim != 3:
        raise ValueError(f"Expected grid shape (N,H,W) or (N,C,H,W), got {grids.shape}")

    max_value = int(np.nanmax(grids)) if grids.size else 0
    palette = ARC_AGI_PALETTE
    if max_value >= len(palette):
        extra = np.zeros((max_value + 1 - len(palette), 3), dtype=np.uint8)
        for i in range(len(palette), max_value + 1):
            extra[i - len(palette)] = [
                (37 * i) % 256,
                (67 * i + 29) % 256,
                (97 * i + 53) % 256,
            ]
        palette = np.concatenate([palette, extra], axis=0)

    safe = np.clip(grids.astype(np.int64), 0, len(palette) - 1)
    return palette[safe].astype(np.uint8)


def load_png_pixels(level_dir: Path, source_root: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("--pixel-source png requires Pillow. Install it with `pip install pillow`.") from exc

    frames_meta = read_jsonl(level_dir / "frames.jsonl")
    images = []
    for meta in frames_meta:
        rel = meta.get("png")
        if rel is None:
            raise KeyError(f"Missing png field in {level_dir / 'frames.jsonl'}")
        path = source_root / rel
        if not path.exists():
            path = level_dir / rel
        with Image.open(path) as img:
            images.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    return np.stack(images, axis=0)


def transition_aligned_actions(npz: np.lib.npyio.NpzFile, n_frames: int) -> np.ndarray:
    if "transition_actions" in npz and len(npz["transition_actions"]) > 0:
        transition = np.asarray(npz["transition_actions"], dtype=np.int64)
        pad = transition[-1:] if len(transition) else np.zeros((1,), dtype=np.int64)
        actions = np.concatenate([transition, pad], axis=0)
    else:
        actions = np.asarray(npz["actions"], dtype=np.int64)
        if len(actions) > 1:
            actions = np.concatenate([actions[1:], actions[-1:]], axis=0)
    if len(actions) != n_frames:
        raise ValueError(f"Action/frame length mismatch: actions={len(actions)} frames={n_frames}")
    return actions


def load_coords(level_dir: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_path = level_dir / "coords.jsonl"
    if not coords_path.exists():
        pos = np.full((n_frames, 2), np.nan, dtype=np.float32)
        bbox = np.full((n_frames, 4), np.nan, dtype=np.float32)
        lost = np.ones((n_frames, 1), dtype=np.float32)
        return pos, bbox, lost

    rows = read_jsonl(coords_path)
    by_idx = {int(row["frame_idx"]): row for row in rows}
    pos = np.full((n_frames, 2), np.nan, dtype=np.float32)
    bbox = np.full((n_frames, 4), np.nan, dtype=np.float32)
    lost = np.ones((n_frames, 1), dtype=np.float32)
    for i in range(n_frames):
        row = by_idx.get(i)
        if row is None:
            continue
        r = float(row.get("row", np.nan))
        c = float(row.get("col", np.nan))
        h = float(row.get("height", np.nan))
        w = float(row.get("width", np.nan))
        pos[i] = [r, c]
        bbox[i] = [r, c, h, w]
        lost[i, 0] = float(bool(row.get("lost", False)))
    return pos, bbox, lost


def encode_actions(action_ids: np.ndarray, encoding: str, num_actions: int) -> np.ndarray:
    if encoding == "scalar":
        return action_ids.astype(np.float32)[:, None]
    out = np.zeros((len(action_ids), num_actions), dtype=np.float32)
    valid = (action_ids >= 0) & (action_ids < num_actions)
    out[np.arange(len(action_ids))[valid], action_ids[valid]] = 1.0
    return out


def collect_episode(
    level_dir: Path,
    source_root: Path,
    level_uid: int,
    episode_uid: int,
    args: argparse.Namespace,
    num_actions: int,
) -> dict[str, np.ndarray]:
    npz_path = level_dir / "episode.npz"
    with np.load(npz_path, allow_pickle=False) as npz:
        if args.pixel_source == "render_grids":
            pixels = grids_to_rgb(npz["render_grids"])
        elif args.pixel_source == "frames":
            pixels = grids_to_rgb(npz["frames"])
        else:
            pixels = load_png_pixels(level_dir, source_root)

        n_frames = int(pixels.shape[0])
        action_ids = transition_aligned_actions(npz, n_frames)
        action = encode_actions(action_ids, args.action_encoding, num_actions)
        frame_action_id = np.asarray(npz["actions"], dtype=np.int16)
        if len(frame_action_id) != n_frames:
            frame_action_id = np.resize(frame_action_id, n_frames)

    proprio, player_bbox, player_lost = load_coords(level_dir, n_frames)
    keep = np.ones(n_frames, dtype=bool)
    if args.skip_lost_coords:
        keep &= player_lost[:, 0] == 0.0
        keep &= np.isfinite(proprio).all(axis=1)

    level_id = parse_level_id(level_dir)
    episode_name = level_dir.parent.name

    return {
        "pixels": pixels[keep].astype(np.uint8),
        "action": action[keep].astype(np.float32),
        "action_id": action_ids[keep, None].astype(np.int16),
        "frame_action_id": frame_action_id[keep, None].astype(np.int16),
        "proprio": proprio[keep].astype(np.float32),
        "player_bbox": player_bbox[keep].astype(np.float32),
        "player_lost": player_lost[keep].astype(np.float32),
        "level_id": np.full((int(keep.sum()), 1), level_id, dtype=np.int16),
        "level_uid": np.full((int(keep.sum()), 1), level_uid, dtype=np.int32),
        "episode_uid": np.full((int(keep.sum()), 1), episode_uid, dtype=np.int32),
        "episode_hash": np.full((int(keep.sum()), 1), stable_hash(episode_name), dtype=np.uint32),
    }


def infer_num_actions(episode_dirs: list[Path]) -> int:
    max_action = 0
    for level_dir in episode_dirs:
        with np.load(level_dir / "episode.npz", allow_pickle=False) as npz:
            for key in ("actions", "transition_actions"):
                if key in npz and npz[key].size:
                    max_action = max(max_action, int(np.nanmax(npz[key])))
    return max_action + 1


def write_hdf5(
    output_path: Path,
    episodes: list[dict[str, np.ndarray]],
    compression: str,
    attrs: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression_arg = None if compression == "none" else compression
    keys = list(episodes[0].keys())
    total = sum(len(ep["pixels"]) for ep in episodes)

    with h5py.File(output_path, "w", libver="latest") as h5:
        h5.create_dataset("ep_len", data=np.asarray([len(ep["pixels"]) for ep in episodes], dtype=np.int32))
        offsets = np.cumsum([0] + [len(ep["pixels"]) for ep in episodes[:-1]])
        h5.create_dataset("ep_offset", data=offsets.astype(np.int64))

        for key in keys:
            sample = episodes[0][key]
            shape = (total, *sample.shape[1:])
            chunk0 = min(max(1, len(sample)), 256)
            chunks = (chunk0, *sample.shape[1:])
            ds = h5.create_dataset(
                key,
                shape=shape,
                dtype=sample.dtype,
                chunks=chunks,
                compression=compression_arg,
                shuffle=compression_arg is not None,
            )
            cursor = 0
            for ep in episodes:
                values = ep[key]
                ds[cursor : cursor + len(values)] = values
                cursor += len(values)

        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.swmr_mode = True


def summarize(source_root: Path, episode_dirs: list[Path]) -> dict[str, Any]:
    action_counts: Counter[int] = Counter()
    frame_counts = []
    coord_files = 0
    lost_count = 0
    for level_dir in episode_dirs:
        with np.load(level_dir / "episode.npz", allow_pickle=False) as npz:
            frame_counts.append(int(npz["render_grids"].shape[0]))
            if "transition_actions" in npz:
                action_counts.update(int(x) for x in npz["transition_actions"].reshape(-1))
        coords_path = level_dir / "coords.jsonl"
        if coords_path.exists():
            coord_files += 1
            for row in read_jsonl(coords_path):
                lost_count += int(bool(row.get("lost", False)))

    return {
        "source_root": str(source_root),
        "episodes": len(episode_dirs),
        "frames": int(sum(frame_counts)),
        "min_frames_per_episode": int(min(frame_counts)),
        "max_frames_per_episode": int(max(frame_counts)),
        "coord_files": coord_files,
        "lost_coord_frames": lost_count,
        "action_counts": dict(sorted(action_counts.items())),
    }


def main() -> None:
    args = parse_args()
    source_root = find_source_root(args.input_root, args.dataset_kind)
    episode_dirs = discover_episode_dirs(source_root, args.dataset_kind)
    stats = summarize(source_root, episode_dirs)
    print("[convert] Detected dataset:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.inspect_only:
        return

    num_actions = args.num_actions or infer_num_actions(episode_dirs)
    print(f"[convert] action_encoding={args.action_encoding} num_actions={num_actions}")

    episode_uid_by_name: dict[str, int] = {}
    episodes = []
    manifest = []
    for level_uid, level_dir in enumerate(episode_dirs):
        episode_name = level_dir.parent.name
        episode_uid = episode_uid_by_name.setdefault(episode_name, len(episode_uid_by_name))
        ep = collect_episode(level_dir, source_root, level_uid, episode_uid, args, num_actions)
        if len(ep["pixels"]) == 0:
            print(f"[convert] Skipping empty episode after filtering: {level_dir}")
            continue
        episodes.append(ep)
        manifest.append(
            {
                "level_uid": level_uid,
                "episode_uid": episode_uid,
                "episode_id": episode_name,
                "level_id": parse_level_id(level_dir),
                "source_dir": str(level_dir),
                "frames": int(len(ep["pixels"])),
            }
        )

    if not episodes:
        raise RuntimeError("No episodes left to write.")

    attrs = {
        "format": "stable_worldmodel_hdf5",
        "source": "arcagi3_human_replay",
        "pixel_source": args.pixel_source,
        "action_encoding": args.action_encoding,
        "num_actions": int(num_actions),
        "proprio_description": "player row,col in rendered grid coordinates",
        "split_key": "level_uid",
    }
    write_hdf5(args.output.expanduser(), episodes, args.compression, attrs)

    sidecar = args.output.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps({"stats": stats, "levels": manifest}, indent=2), encoding="utf-8")
    print(f"[convert] Wrote {args.output}")
    print(f"[convert] Wrote {sidecar}")


if __name__ == "__main__":
    main()
