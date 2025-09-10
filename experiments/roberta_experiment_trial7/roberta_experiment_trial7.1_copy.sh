#!/bin/bash
#SBATCH --job-name=ro_at_7_2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs_new/%x.out
#SBATCH --error=logs_new/%x.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=ALL

# --- Modules & env ---
module load Anaconda3/2024.02-1
module load CUDA/12.1.1
source activate myspark

# quieter & a bit safer defaults
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p logs_new

SEEDS="7 1001 1211"

TRAIN_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_AA_train.jsonl"
VAL_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_AA_val.jsonl"
TEST_JSONL="msc-dissertation/experiments/pairs_balanced_json/gold_AT_test.jsonl"

for SEED in $SEEDS; do
  echo "========= Running experiment with SEED: ${SEED} ========="

  OUTDIR="./models/roberta/article_tweet_trial7_seed${SEED}"
  mkdir -p "${OUTDIR}"

  # seed=7 SHAP
  if [ "$SEED" -eq 7 ]; then
    python msc-dissertation/experiments/roberta_experiment_trial7/roberta_experiment_trial7.py \
      --train_jsonl "${TRAIN_JSONL}" \
      --val_jsonl   "${VAL_JSONL}" \
      --test_jsonl  "${TEST_JSONL}" \
      --model_name roberta-base \
      --output_dir "${OUTDIR}" \
      --max_length 512 \
      --batch_size 4 \
      --num_train_epochs 3 \
      --seed ${SEED} \
      --run_shap
  else
    python msc-dissertation/experiments/roberta_experiment_trial7/roberta_experiment_trial7.py \
      --train_jsonl "${TRAIN_JSONL}" \
      --val_jsonl   "${VAL_JSONL}" \
      --test_jsonl  "${TEST_JSONL}" \
      --model_name roberta-base \
      --output_dir "${OUTDIR}" \
      --max_length 512 \
      --batch_size 4 \
      --num_train_epochs 3 \
      --seed ${SEED}
  fi

  echo "========= Finished SEED: ${SEED} ========="
done