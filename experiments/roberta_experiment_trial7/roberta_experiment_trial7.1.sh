#!/bin/bash
#SBATCH --job-name=ro_aa_7_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0
#SBATCH --output=logs_new/%x_%A_%a.out
#SBATCH --error=logs_new/%x_%A_%a.err
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=ALL

set -euo pipefail

# ==== modules & env ====
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
source activate myspark

mkdir -p logs_new

# >>> 절대경로 Python 고정 (환경 맞게 확인)
PY="/users/acp24hj/.conda/envs/myspark/bin/python"

# threads & misc
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_NO_TORCHVISION=1   # torchvision 충돌 방지

# seeds (array index로 선택)
SEEDS=(7 1001 1211)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

TRAIN_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/silver_AA_train.jsonl"
VAL_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/silver_AA_val.jsonl"
TEST_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/gold_AA_test.jsonl"

OUTDIR="$HOME/models/roberta/article_article_trial7_seed${SEED}"
mkdir -p "${OUTDIR}"

# 디버그: 실제 파이썬 확인 (로그에 남김)
srun -u bash -lc "$PY -V && which $PY"

# seed=7 에서만 SHAP
if [ "$SEED" -eq 7 ]; then
  srun -u "$PY" "$HOME/msc-dissertation/experiments/roberta_experiment_trial7/roberta_experiment_trial7.py" \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl   "${VAL_JSONL}" \
    --test_jsonl  "${TEST_JSONL}" \
    --model_name roberta-base \
    --output_dir "${OUTDIR}" \
    --max_length 512 \
    --batch_size 4 \
    --num_train_epochs 3 \
    --seed ${SEED} \
    --logging_steps 2000 \
    --run_shap
else
  srun -u "$PY" "$HOME/msc-dissertation/experiments/roberta_experiment_trial7/roberta_experiment_trial7.py" \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl   "${VAL_JSONL}" \
    --test_jsonl  "${TEST_JSONL}" \
    --model_name roberta-base \
    --output_dir "${OUTDIR}" \
    --max_length 512 \
    --batch_size 4 \
    --num_train_epochs 3 \
    --seed ${SEED} \
    --logging_steps 2000
fi