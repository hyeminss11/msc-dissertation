#!/bin/bash
#SBATCH --job-name=obf_flan_from_orig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjeong17@sheffield.ac.uk

set -euo pipefail

# -------- Paths --------
DOCS_JSON="$HOME/msc-dissertation/experiments/llm-modification-setting/crossnews_gold_clean_anon.json"
OUT_DIR="$HOME/msc-dissertation/experiments/llm-modification-setting"

# impersonation result(JSONL): 이 파일의 id들만 대상으로 원문 기사 obfuscation 수행
IMP_JSONL="${OUT_DIR}/gold_imp_flan_final.jsonl"

# output
OUT_JSONL="${OUT_DIR}/gold_obf_flan_final_re.jsonl"

mkdir -p "$OUT_DIR" logs_new

# -------- Modules & Conda --------
module purge
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

# 안전/조용한 기본값
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1

NUM_SHARDS=1
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

# -------- Run obfuscation (from ORIGINAL articles) --------
python "$HOME/msc-dissertation/experiments/llm-modification-setting/obfuscation_script_flan_t5.py" \
  --docs_json "$DOCS_JSON" \
  --impersonation_jsonl "$IMP_JSONL" \
  --out_jsonl "$OUT_JSONL" \
  --model_name "google/flan-t5-large" \
  --max_input_words 800 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --progress_every 50 \
  --num_shards "$NUM_SHARDS" \
  --shard_id "$SHARD_ID" \
  --resume_jsonl "$OUT_DIR/gold_obf_flan_final.jsonl"