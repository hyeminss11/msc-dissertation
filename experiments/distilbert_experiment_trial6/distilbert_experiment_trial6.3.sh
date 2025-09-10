#!/bin/bash
#SBATCH --job-name=dtb_tt_tr_6_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs_new/%x.out
#SBATCH --error=logs_new/%x.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
module load CUDA/12.1.1
source activate myspark

mkdir -p logs_new

SEED=7
SHAPK=8

OUTDIR="./models/distilbert/tweet_tweet_trial6_with_shap_seed${SEED}"
mkdir -p "${OUTDIR}"

TRAIN_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_TT_train.jsonl"
VAL_JSONL="msc-dissertation/experiments/pairs_balanced_json/silver_TT_val.jsonl"
TEST_JSONL="msc-dissertation/experiments/pairs_balanced_json/gold_TT_test.jsonl"

python msc-dissertation/experiments/distilbert_experiment_trial6/distilbert_experiment_trial6.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl   "${VAL_JSONL}" \
  --test_jsonl  "${TEST_JSONL}" \
  --model_name distilbert-base-uncased \
  --output_dir "${OUTDIR}" \
  --max_length 512 \
  --batch_size 16 \
  --num_train_epochs 3 \
  --seed ${SEED} \
  --run_shap