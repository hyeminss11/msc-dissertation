#!/bin/bash
#SBATCH --job-name=imp_pair_gen_aa
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjeong17@sheffield.ac.uk

set -euo pipefail

# -------- Paths --------
DOCS_JSON="$HOME/msc-dissertation/experiments/llm-modification-setting/crossnews_gold_clean_anon.json"
MODIFIED_JSONL="$HOME/msc-dissertation/experiments/llm-modification-setting/gold_imp_flan_final.jsonl"
OUT_JSONL="$HOME/msc-dissertation/experiments/pairs_balanced_json/aa_imp_pairs.jsonl"

mkdir -p "$(dirname "$OUT_JSONL")" logs_new

# -------- Modules & conda --------
module purge
module load Anaconda3/2024.02-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

# -------- Run pair generation --------
python "$HOME/msc-dissertation/experiments/llm-modification-setting/imp_pair_gen_aa.py" \
  --docs_json "$DOCS_JSON" \
  --modified_jsonl "$MODIFIED_JSONL" \
  --out_jsonl "$OUT_JSONL" \
  --pair_tag "AA-IMP" \
  --neg_ratio 1 \
  --neg_per_author_cap 50 \
  --swap_prob 0.5

echo "[✓] Pair generation complete -> ${OUT_JSONL}"