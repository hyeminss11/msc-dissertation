#!/bin/bash
#SBATCH --job-name=obf_flan_from_orig
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --array=0-19
#SBATCH --output=logs_new/%x_%A_%a.out
#SBATCH --error=logs_new/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjeong17@sheffield.ac.uk

set -euo pipefail

# -------- Paths --------
DOCS_JSON="$HOME/msc-dissertation/experiments/llm-modification-setting/crossnews_gold_clean_anon.json"
OUT_DIR="$HOME/msc-dissertation/experiments/llm-modification-setting/obf_from_orig_cpu"
IMP_JSONL="$HOME/msc-dissertation/experiments/llm-modification-setting/gold_imp_flan_final.jsonl"
RESUME_JSONL="$HOME/msc-dissertation/experiments/llm-modification-setting/gold_obf_flan_final.jsonl"

mkdir -p "$OUT_DIR" logs_new

# -------- Modules & Conda --------
module purge
module load Anaconda3/2024.02-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

# -------- Env vars --------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1

NUM_SHARDS=20
SHARD_ID=${SLURM_ARRAY_TASK_ID}
OUT_JSONL="${OUT_DIR}/out_shard_${SHARD_ID}.jsonl"

# -------- Run obfuscation (from ORIGINAL articles) --------
python "$HOME/msc-dissertation/experiments/llm-modification-setting/obfuscation_script_flan_t5.py" \
  --docs_json           "$DOCS_JSON" \
  --impersonation_jsonl "$IMP_JSONL" \
  --out_jsonl           "$OUT_JSONL" \
  --model_name          "google/flan-t5-large" \
  --max_input_words     800 \
  --max_new_tokens      256 \
  --temperature         0.7 \
  --top_p               0.9 \
  --progress_every      50 \
  --num_shards          "$NUM_SHARDS" \
  --shard_id            "$SHARD_ID" \
  --resume_jsonl        "$RESUME_JSONL"

echo "[✓] Shard ${SHARD_ID} complete -> ${OUT_JSONL}"