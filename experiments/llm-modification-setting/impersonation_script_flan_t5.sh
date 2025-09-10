#!/bin/bash
#SBATCH --job-name=imp_flan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjeong17@sheffield.ac.uk

set -euo pipefail

# ---- Modules & Conda ----
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

PY="$HOME/.conda/envs/myspark/bin/python"

# ---- Paths ----
DOCS_JSON="$HOME/msc-dissertation/experiments/llm-modification-setting/crossnews_gold_clean_anon.json"
PAIRS_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/gold_AT_test.jsonl"
OUT_DIR="$HOME/msc-dissertation/experiments/llm-modification-setting"
OUT_JSONL="${OUT_DIR}/gold_imp_flan_final.jsonl"

# ---- Run ----
$PY "$HOME/msc-dissertation/experiments/llm-modification-setting/impersonation_script_flan_t5.py" \
  --docs_json "$DOCS_JSON" \
  --pairs "$PAIRS_JSONL" \
  --out_jsonl "$OUT_JSONL" \
  --model_name "google/flan-t5-large" \
  --max_input_words 500 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --support_k 3 \
  --num_shards 1 \
  --shard_id 0