#!/bin/bash
#SBATCH --job-name=roberta_at_imp_cm
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err

module load Anaconda3/2024.02-1
module load CUDA/11.8.0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

PY="$HOME/.conda/envs/myspark/bin/python"

# quiet & safer defaults
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1

mkdir -p logs_new

$PY $HOME/msc-dissertation/experiments/distilbert_experiment_trial6/eval_bi_encoder_with_shap.py \
  --model_dir "$HOME/models/roberta/article_article_trial7_seed1001/best_model" \
  --val_jsonl "$HOME/msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_imp_pairs_val.jsonl" \
  --test_jsonl "$HOME/msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_imp_pairs_test.jsonl" \
  --batch_size 256 \
  --no_shap