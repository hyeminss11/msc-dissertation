#!/bin/bash
#SBATCH --job-name=obf_pair_gen_at        # Job name
#SBATCH --partition=sheffield             # Partition (CPU queue)
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=32G                         # Memory
#SBATCH --time=20:00:00                   # Time limit
#SBATCH --output=logs_new/%x_%j.out       # Stdout log
#SBATCH --error=logs_new/%x_%j.err        # Stderr log

set -euo pipefail

# -------- Paths --------
DOCS_JSON="$HOME/msc-dissertation/experiments/llm-modification-setting/crossnews_gold_clean_anon.json"
MODIFIED_JSONL="$HOME/msc-dissertation/experiments/llm-modification-setting/gold_obf_flan_final.jsonl"
OUT_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/at_obf_pairs.jsonl"

mkdir -p "$(dirname "$OUT_JSONL")" logs_new

# -------- Modules & Conda --------
module purge
module load Anaconda3/2024.02-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

# -------- Run --------
python "$HOME/msc-dissertation/experiments/llm-modification-setting/imp_pair_gen_at.py" \
  --docs_json "$DOCS_JSON" \
  --modified_jsonl "$MODIFIED_JSONL" \
  --out_jsonl "$OUT_JSONL" \
  --pair_tag "AT-OBF" \
  --neg_ratio 1 \
  --neg_per_author_cap 50

echo "[✓] AT pair generation complete -> ${OUT_JSONL}"