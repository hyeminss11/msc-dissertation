#!/bin/bash
#SBATCH --job-name=bb_aa_9_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
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

PY="/users/acp24hj/.conda/envs/myspark/bin/python"

# threads & misc
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_NO_TORCHVISION=1
# help avoid CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 디버그: 실제 파이썬 확인
$PY -V && which $PY

# === array index로 seed 선택 ===
SEEDS=(7 1001 1211)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

TRAIN_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/silver_AA_train.jsonl"
VAL_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/silver_AA_val.jsonl"
TEST_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/gold_AA_test.jsonl"

OUTDIR="$HOME/models/bigbird/article_article_trial9_seed${SEED}"
mkdir -p "${OUTDIR}"

echo "========= Running experiment with SEED: ${SEED} ========="

# seed=7 은 SHAP 추가
if [ "$SEED" -eq 7 ]; then
  $PY msc-dissertation/experiments/bigbird_experiment_trial9/bigbird_experiment_trial9.py \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl   "${VAL_JSONL}" \
    --test_jsonl  "${TEST_JSONL}" \
    --model_name google/bigbird-roberta-base \
    --output_dir "${OUTDIR}" \
    --max_length 1024 \
    --batch_size 2 \
    --num_train_epochs 3 \
    --seed ${SEED} \
    --logging_steps 5000 \
    --run_shap
else
  $PY msc-dissertation/experiments/bigbird_experiment_trial9/bigbird_experiment_trial9.py \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl   "${VAL_JSONL}" \
    --test_jsonl  "${TEST_JSONL}" \
    --model_name google/bigbird-roberta-base \
    --output_dir "${OUTDIR}" \
    --max_length 1024 \
    --batch_size 2 \
    --num_train_epochs 3 \
    --seed ${SEED} \
    --logging_steps 5000
fi

echo "========= Finished SEED: ${SEED} ========="