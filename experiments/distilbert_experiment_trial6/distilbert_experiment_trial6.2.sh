#!/bin/bash
#SBATCH --job-name=dtb_at_6_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs_new/%x.out
#SBATCH --error=logs_new/%x.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
module load CUDA/12.1.1
source activate myspark

mkdir -p logs_new

SEEDS="7 1001 1211"

for SEED in $SEEDS
do
  echo "========= Running experiment with SEED: ${SEED} ========="

  OUTDIR="./models/distilbert/article_tweet_trial6_seed${SEED}"
  mkdir -p "${OUTDIR}"

  TRAIN_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_AA_train.jsonl"
  VAL_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_AA_val.jsonl"
  TEST_JSONL="msc-dissertation/experiments/pairs_balanced_json/gold_AT_test.jsonl"

  if [ "$SEED" -eq 7 ]; then
    # Run with SHAP
    python msc-dissertation/experiments/distilbert_experiment_trial6/distilbert_experiment_trial6.py \
      --train_jsonl "${TRAIN_JSONL}" \
      --val_jsonl   "${VAL_JSONL}" \
      --test_jsonl  "${TEST_JSONL}" \
      --model_name distilbert-base-uncased \
      --output_dir "${OUTDIR}" \
      --max_length 512 \
      --batch_size 4 \
      --num_train_epochs 3 \
      --seed ${SEED} \
      --run_shap
  else
    # Run without SHAP
    python msc-dissertation/experiments/distilbert_experiment_trial6/distilbert_experiment_trial6.py \
      --train_jsonl "${TRAIN_JSONL}" \
      --val_jsonl   "${VAL_JSONL}" \
      --test_jsonl  "${TEST_JSONL}" \
      --model_name distilbert-base-uncased \
      --output_dir "${OUTDIR}" \
      --max_length 512 \
      --batch_size 4 \
      --num_train_epochs 3 \
      --seed ${SEED}
  fi

  echo "========= Finished SEED: ${SEED} ========="
done