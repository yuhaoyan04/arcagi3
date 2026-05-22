#!/usr/bin/env bash
set -euo pipefail

H5_PATH="data/processed/arcagi3_human_replay.h5"
CHECKPOINT=""
OUTPUT_DIR="outputs/arcagi3/probe/$(date +%Y%m%d_%H%M%S)"
LAYERS="low,mid,high"
PROBE_TYPES="linear,mlp"
LEWM_INITS="trained,random"
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCHS=50
MLP_HIDDEN_SIZE=128
DEVICE="auto"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-arcagi3-probe}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_FLAG="--swanlab"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: scripts/run_probe_arcagi3.sh --checkpoint PATH [options]

Options:
  --h5 PATH                 Converted HDF5 dataset path.
  --checkpoint PATH         LeWM *_object.ckpt checkpoint. Required.
  --output-dir PATH         Probe output directory.
  --layers LIST             low,mid,high or all. Default: low,mid,high.
  --probe-types LIST        linear,mlp or all. Default: linear,mlp.
  --lewm-inits LIST         trained,random,both. Default: trained,random.
  --batch-size N            Batch size. Default: 128.
  --learning-rate LR        Probe learning rate. Default: 0.001.
  --epochs N                Probe epochs. Default: 50.
  --mlp-hidden-size N       128 or 256. Default: 128.
  --device VALUE            cuda, cpu, or auto. Default: auto.
  --no-swanlab              Disable SwanLab logging.
  --save-heatmap            Save optional test error heatmap.
  -h, --help                Show this message.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --h5) H5_PATH="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --probe-types) PROBE_TYPES="$2"; shift 2 ;;
    --lewm-inits) LEWM_INITS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --mlp-hidden-size) MLP_HIDDEN_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-swanlab) SWANLAB_FLAG="--no-swanlab"; shift ;;
    --save-heatmap) EXTRA_ARGS+=("--save-heatmap"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required." >&2
  usage
  exit 2
fi
if [[ ! -f "$H5_PATH" ]]; then
  echo "HDF5 dataset not found: $H5_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[probe] output_dir=$OUTPUT_DIR"
ARGS=(
  --h5 "$H5_PATH"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --probe-types "$PROBE_TYPES"
  --lewm-inits "$LEWM_INITS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --epochs "$EPOCHS"
  --mlp-hidden-size "$MLP_HIDDEN_SIZE"
  --swanlab-project "$SWANLAB_PROJECT"
  --swanlab-workspace "$SWANLAB_WORKSPACE"
  "$SWANLAB_FLAG"
)
if [[ "$DEVICE" != "auto" ]]; then
  ARGS+=(--device "$DEVICE")
fi
ARGS+=("${EXTRA_ARGS[@]}")

"$PYTHON_BIN" scripts/probe_arcagi3_lewm.py "${ARGS[@]}"
