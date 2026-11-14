#!/usr/bin/env bash
#SBATCH --job-name=auto-anomaly-fullres
#SBATCH --nodelist=gpu-4-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# sbatch --export=ALL,CATEGORY=carpet,LOSS=l1_ssim,NUM_STAGES=4,BASE_CHANNELS=64,LATENT_DIM=256,BATCH_SIZE=16,EPOCHS=200,GAUSSIAN_SIGMA=8.0,TOP_K_PCT=0.001 test.sh
# sbatch --export=ALL,CATEGORY=hazelnut,LOSS=l1_ssim,NUM_STAGES=5,BASE_CHANNELS=48,LATENT_DIM=128,BATCH_SIZE=16,EPOCHS=200,GAUSSIAN_SIGMA=12.0,TOP_K_PCT=0.001 test.sh
# sbatch --export=ALL,CATEGORY=bottle,LOSS=l1_ssim,NUM_STAGES=5,BASE_CHANNELS=48,LATENT_DIM=128,BATCH_SIZE=16,EPOCHS=200,GAUSSIAN_SIGMA=12.0,TOP_K_PCT=0.001 test.sh
# sbatch --export=ALL,CATEGORY=all,LOSS=l1_ssim,NUM_STAGES=5,BASE_CHANNELS=64,LATENT_DIM=128,BATCH_SIZE=16,EPOCHS=200,GAUSSIAN_SIGMA=10.0,TOP_K_PCT=0.001,SAVE_TIFF=1,EXP_NAME=mvtec_fullres_l1ssim_s5 test.sh
# for loss_fn in l2 l1 l1_ssim; do
    # sbatch --export=ALL,CATEGORY=carpet,LOSS="${loss_fn}",NUM_STAGES=4,BASE_CHANNELS=64,LATENT_DIM=256,BATCH_SIZE=16,EPOCHS=200,GAUSSIAN_SIGMA=8.0,TOP_K_PCT=0.001 test.sh
# done

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

echo "[1/2] Running pipeline smoke tests..."
"${PYTEST_RUN[@]}" -q

# Configurable hyperparameters (override via sbatch --export=ALL,VAR=val)
DATA_DIR="${DATA_DIR:-data}"
CATEGORY="${CATEGORY:-carpet}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LATENT_DIM="${LATENT_DIM:-256}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
NUM_STAGES="${NUM_STAGES:-4}"
GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-8.0}"
TOP_K_PCT="${TOP_K_PCT:-0.001}"
LOSS="${LOSS:-l1_ssim}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
EXP_NAME="${EXP_NAME:-${CATEGORY}_fullres_${LOSS}_s${NUM_STAGES}_c${BASE_CHANNELS}_z${LATENT_DIM}}"

EXTRA_ARGS=()
if [[ "${FULL_RES:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--full-res)
fi
if [[ "${SAVE_TIFF:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--save-tiff)
fi

echo "[2/2] Launching Experiment: ${EXP_NAME} (Category: ${CATEGORY})"
"${PYTHON_RUN[@]}" -m src \
    --data-dir "${DATA_DIR}" \
    --category "${CATEGORY}" \
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
    --num-workers "${NUM_WORKERS}" \
    --output "${EXP_NAME}" \
    --device cuda \
    --seed 42 \
    "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "Finished at      : $(date --iso-8601=seconds)"
echo "============================================================"
