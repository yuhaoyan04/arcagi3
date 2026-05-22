#!/usr/bin/env bash
set -euo pipefail

H5_PATH="data/processed/arcagi3_human_replay.h5"
BATCH_SIZE=128
LEARNING_RATE="5e-5"
NUM_EPOCHS=10
FRAMESKIP=1
HISTORY_SIZE=3
NUM_PREDS=1
EXP_NAME="arcagi3_lewm"
RUN_DIR=""
ACCELERATOR="gpu"
DEVICES="auto"
PRECISION="bf16"
LOGGER_BACKEND="swanlab"
SWANLAB_ENABLED="true"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-arcagi3-lewm}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: scripts/run_train_lewm_arcagi3.sh [options]

Options:
  --h5 PATH                 Converted HDF5 dataset path.
  --batch-size N            Batch size. Default: 128.
  --learning-rate LR        Optimizer learning rate. Default: 5e-5.
  --num-epochs N            Training epochs. Default: 10.
  --frameskip N             Dataset frameskip. Default: 1.
  --history-size N          LeWM context length. Default: 3.
  --num-preds N             Prediction offset. Default: 1.
  --exp-name NAME           Experiment name. Default: arcagi3_lewm.
  --run-dir PATH            Output directory for config/checkpoints/logs.
  --accelerator VALUE       Lightning accelerator. Default: gpu.
  --devices VALUE           Lightning devices. Default: auto.
  --precision VALUE         Lightning precision. Default: bf16.
  --logger-backend VALUE    swanlab, wandb, or none. Default: swanlab.
  --swanlab-project NAME    SwanLab project. Default: arcagi3-lewm.
  --swanlab-workspace NAME  SwanLab workspace. Default: account default.
  --no-swanlab              Disable SwanLab logging.
  -h, --help                Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --h5) H5_PATH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --frameskip) FRAMESKIP="$2"; shift 2 ;;
    --history-size) HISTORY_SIZE="$2"; shift 2 ;;
    --num-preds) NUM_PREDS="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --logger-backend) LOGGER_BACKEND="$2"; shift 2 ;;
    --swanlab-project) SWANLAB_PROJECT="$2"; shift 2 ;;
    --swanlab-workspace) SWANLAB_WORKSPACE="$2"; shift 2 ;;
    --no-swanlab) SWANLAB_ENABLED="false"; LOGGER_BACKEND="none"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$H5_PATH" ]]; then
  echo "HDF5 dataset not found: $H5_PATH" >&2
  echo "Run scripts/convert_arcagi3_to_hdf5.py first." >&2
  exit 1
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="outputs/arcagi3/lewm/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN_DIR"
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/outputs/stablewm}"

ARGS=(
  "data=arcagi3"
  "data.dataset.path=$H5_PATH"
  "data.dataset.frameskip=$FRAMESKIP"
  "loader.batch_size=$BATCH_SIZE"
  "optimizer.lr=$LEARNING_RATE"
  "trainer.max_epochs=$NUM_EPOCHS"
  "trainer.accelerator=$ACCELERATOR"
  "trainer.devices=$DEVICES"
  "trainer.precision=$PRECISION"
  "wm.history_size=$HISTORY_SIZE"
  "wm.num_preds=$NUM_PREDS"
  "exp_name=$EXP_NAME"
  "subdir=$RUN_DIR"
  "logger_backend=$LOGGER_BACKEND"
  "swanlab.enabled=$SWANLAB_ENABLED"
  "swanlab.config.project=$SWANLAB_PROJECT"
  "swanlab.config.experiment_name=$EXP_NAME"
  "swanlab.config.logdir=$RUN_DIR/swanlab"
)

if [[ -n "$SWANLAB_WORKSPACE" ]]; then
  ARGS+=("swanlab.config.workspace=$SWANLAB_WORKSPACE")
else
  ARGS+=("swanlab.config.workspace=null")
fi

if [[ "$LOGGER_BACKEND" == "none" ]]; then
  ARGS+=("wandb.enabled=false")
fi

echo "[train] STABLEWM_HOME=$STABLEWM_HOME"
echo "[train] run_dir=$RUN_DIR"
"$PYTHON_BIN" train.py "${ARGS[@]}"
