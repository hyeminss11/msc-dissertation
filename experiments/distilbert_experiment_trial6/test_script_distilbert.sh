#!/bin/bash
#SBATCH --job-name=distilbert_at_imp_cm
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err

# ---- modules & env ----
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

PY="$HOME/.conda/envs/myspark/bin/python"

# ---- paths (바꿀 부분은 여기) ----
MODEL_DIR="$HOME/models/distilbert/article_article_trial6_seed1001/best_model"
TEST_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_imp_pairs_test.jsonl"
VAL_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_imp_pairs_val.jsonl"   # tau(EER) 계산용

# ---- run ----
$PY msc-dissertation/experiments/distilbert_experiment_trial6/eval_bi_encoder_with_shap.py \
  --model_dir "$MODEL_DIR" \
  --test_jsonl "$TEST_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --batch_size 256 \
  --no_shap