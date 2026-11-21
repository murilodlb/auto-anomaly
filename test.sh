#!/usr/bin/env bash
#SBATCH --job-name=auto-anomaly
#SBATCH --partition=gpu-4-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# sbatch --export=ALL,CATEGORY=all,LOSS=l1_ssim,NUM_STAGES=5,BASE_CHANNELS=64,LATENT_DIM=128,BATCH_SIZE=16,EPOCHS=100,GAUSSIAN_SIGMA=10.0,TOP_K_PCT=0.001,SAVE_TIFF=1,EXP_NAME=mvtec_fullres_l1ssim_s5 test.sh
# sbatch --export=ALL,CATEGORY=carpet,EXTRACTOR=dinov3,LOSS=cosine,NUM_STAGES=3,BASE_CHANNELS=128,LATENT_DIM=64,BATCH_SIZE=16,EPOCHS=100,GAUSSIAN_SIGMA=3.0,TOP_K_PCT=0.001,LOSS_FLOOR=0.0,DINOV3_STRIDE=16,DINOV3_PRUNE_REGISTERS=1,EXP_NAME=carpet_dinov3_cosine_s3_c128_z64 test.sh
# Compare DCAE vs Bottleneck-WTA ConvAE (wta_ae) on carpet at 1024x1024:
# sbatch --export=ALL,CATEGORY=carpet,MODEL=dcae,LOSS=ms_l1_ssim,NUM_STAGES=6,BASE_CHANNELS=64,LATENT_DIM=64,EPOCHS=70,SPATIAL_DROPOUT=0.25,TOP_K_PCT=0.0004,GAUSSIAN_SIGMA=6.0 test.sh
# sbatch --export=ALL,CATEGORY=carpet,MODEL=wta_ae,LOSS=ms_l1_ssim,NUM_STAGES=6,BASE_CHANNELS=64,LATENT_DIM=64,EPOCHS=70,WTA_SPATIAL_WINDOW=4,WTA_LIFETIME_SPARSITY=0.2,WTA_EVAL_SCALE=0.8,TOP_K_PCT=0.0004,GAUSSIAN_SIGMA=6.0 test.sh
# sbatch --export=ALL,CATEGORY=carpet,MODEL=wta_ae,LOSS=l1_ssim,NUM_STAGES=5,BASE_CHANNELS=64,LATENT_DIM=128,EPOCHS=100,WTA_SPATIAL_WINDOW=4,WTA_LIFETIME_SPARSITY=0.1,WTA_EVAL_SCALE=0.8,TOP_K_PCT=0.0005,GAUSSIAN_SIGMA=8.0 test.sh
# sbatch --export=ALL,CATEGORY=carpet,MODEL=wta_ae,LOSS=ms_l1_ssim,NUM_STAGES=6,BASE_CHANNELS=64,LATENT_DIM=128,EPOCHS=80,WTA_SPATIAL_WINDOW=2,WTA_LIFETIME_SPARSITY=0.15,WTA_EVAL_SCALE=0.8,TOP_K_PCT=0.0004,GAUSSIAN_SIGMA=6.0 test.sh

set -euo pipefail

mkdir -p logs results

echo "============================================================"
echo "Job ID           : ${SLURM_JOB_ID:-local}"
echo "Node             : $(hostname)"
echo "Start Time       : $(date --iso-8601=seconds)"
echo "Working Dir      : $(pwd)"
echo "CPUs Allocated   : ${SLURM_CPUS_PER_TASK:-8}"
echo "============================================================"

nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv

if command -v uv >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
    PYTHON_RUN=(uv run python)
    PYTEST_RUN=(uv run pytest)
elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PYTHON_RUN=(python)
    PYTEST_RUN=(pytest)
else
    PYTHON_RUN=(python3)
    PYTEST_RUN=(pytest)
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "[0/2] Verifying PyTorch CUDA availability..."
"${PYTHON_RUN[@]}" -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CUDA unavailable in PyTorch; run uv sync to install cu126 wheels compatible with NVIDIA driver 560.x'"

echo "[1/2] Running pipeline smoke tests..."
"${PYTEST_RUN[@]}" -q

# Configurable hyperparameters (override via sbatch --export=ALL,VAR=val)
DATA_DIR="${DATA_DIR:-data}"
CATEGORY="${CATEGORY:-carpet}"
EXTRACTOR="${EXTRACTOR:-raw}"
MODEL="${MODEL:-dcae}"
EPOCHS="${EPOCHS:-70}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LATENT_DIM="${LATENT_DIM:-64}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
if [[ "${EXTRACTOR}" == "dinov3" ]]; then
    LOSS="${LOSS:-cosine}"
    LOSS_FLOOR="${LOSS_FLOOR:-0.0}"
    NUM_STAGES="${NUM_STAGES:-3}"
    GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-3.0}"
else
    LOSS="${LOSS:-ms_l1_ssim}"
    LOSS_FLOOR="${LOSS_FLOOR:-0.006}"
    NUM_STAGES="${NUM_STAGES:-6}"
    GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-6.0}"
fi
TOP_K_PCT="${TOP_K_PCT:-0.0004}"
SPATIAL_DROPOUT="${SPATIAL_DROPOUT:-0.25}"
DENOISE_PROB="${DENOISE_PROB:-0.5}"
TEXTURE_DAMPING="${TEXTURE_DAMPING:-2.0}"
GUIDED_FILTER_RADIUS="${GUIDED_FILTER_RADIUS:-8}"
GUIDED_FILTER_EPS="${GUIDED_FILTER_EPS:-0.01}"
WTA_SPATIAL_WINDOW="${WTA_SPATIAL_WINDOW:-4}"
WTA_LIFETIME_SPARSITY="${WTA_LIFETIME_SPARSITY:-0.2}"
WTA_EVAL_SCALE="${WTA_EVAL_SCALE:-0.8}"
SEED="${SEED:-42}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
if [[ "${EXTRACTOR}" == "raw" ]]; then
    EXP_NAME="${EXP_NAME:-${CATEGORY}_${MODEL}_fullres_${LOSS}_s${NUM_STAGES}_c${BASE_CHANNELS}_z${LATENT_DIM}}"
else
    EXP_NAME="${EXP_NAME:-${CATEGORY}_${MODEL}_fullres_${EXTRACTOR}_${LOSS}_s${NUM_STAGES}_c${BASE_CHANNELS}_z${LATENT_DIM}}"
fi

EXTRA_ARGS=()
if [[ "${FULL_RES:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--full-res)
fi
if [[ "${SAVE_TIFF:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--save-tiff)
fi
if [[ "${EXTRACTOR}" == "dinov3" ]]; then
    if [[ -n "${DINOV3_MODEL:-}" ]]; then
        EXTRA_ARGS+=(--dinov3-model "${DINOV3_MODEL}")
    fi
    if [[ -n "${DINOV3_STRIDE:-}" ]]; then
        EXTRA_ARGS+=(--dinov3-stride "${DINOV3_STRIDE}")
    fi
    if [[ -n "${DINOV3_LAYERS:-}" ]]; then
        read -r -a _dinov3_layers_arr <<< "${DINOV3_LAYERS}"
        EXTRA_ARGS+=(--dinov3-layers "${_dinov3_layers_arr[@]}")
    fi
    if [[ "${DINOV3_PRUNE_REGISTERS:-0}" == "1" ]]; then
        EXTRA_ARGS+=(--dinov3-prune-registers)
    fi
    if [[ -n "${DINOV3_MEAN_CENTER_ALPHA:-}" ]]; then
        EXTRA_ARGS+=(--dinov3-mean-center-alpha "${DINOV3_MEAN_CENTER_ALPHA}")
    fi
    if [[ -n "${DINOV3_TARGET_SHORT_EDGE:-}" ]]; then
        EXTRA_ARGS+=(--dinov3-target-short-edge "${DINOV3_TARGET_SHORT_EDGE}")
    fi
    if [[ "${DINOV3_PLAIN_LAST_HIDDEN_STATE:-0}" == "1" ]]; then
        EXTRA_ARGS+=(--dinov3-plain-last-hidden-state)
    fi
fi

echo "[2/2] Launching Experiment: ${EXP_NAME} (Category: ${CATEGORY}, Model: ${MODEL}, Extractor: ${EXTRACTOR})"
"${PYTHON_RUN[@]}" -m src \
    --data-dir "${DATA_DIR}" \
    --category "${CATEGORY}" \
    --extractor "${EXTRACTOR}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --latent-dim "${LATENT_DIM}" \
    --base-channels "${BASE_CHANNELS}" \
    --num-stages "${NUM_STAGES}" \
    --gaussian-sigma "${GAUSSIAN_SIGMA}" \
    --top-k-pct "${TOP_K_PCT}" \
    --loss "${LOSS}" \
    --spatial-dropout "${SPATIAL_DROPOUT}" \
    --denoise-prob "${DENOISE_PROB}" \
    --texture-damping "${TEXTURE_DAMPING}" \
    --loss-floor "${LOSS_FLOOR}" \
    --guided-filter-radius "${GUIDED_FILTER_RADIUS}" \
    --guided-filter-eps "${GUIDED_FILTER_EPS}" \
    --wta-spatial-window "${WTA_SPATIAL_WINDOW}" \
    --wta-lifetime-sparsity "${WTA_LIFETIME_SPARSITY}" \
    --wta-eval-scale "${WTA_EVAL_SCALE}" \
    --num-workers "${NUM_WORKERS}" \
    --output "${EXP_NAME}" \
    --device cuda \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "Finished at      : $(date --iso-8601=seconds)"
echo "============================================================"
