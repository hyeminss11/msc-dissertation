#!/bin/bash
#SBATCH --job-name=finderr
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs_new/%x_%j.out
#SBATCH --error=logs_new/%x_%j.err
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Activate conda environment (adjust path as needed)
module load Anaconda3/2024.02-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate myspark

# Create output directory
mkdir -p error_analysis_results

# Define model paths and corresponding test files
declare -a MODELS=(
   "models/distilbert/article_article_trial6_seed7/best_model"
   "models/roberta/article_article_trial7_seed7/best_model"
   "models/bigbird/article_article_trial9_seed7/best_model"
)

declare -a MODEL_NAMES=(
   "distilbert"
   "roberta"
   "bigbird"
)

declare -a TEST_FILES=(
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/gold_AA_test.jsonl"
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/gold_AT_test.jsonl"
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/aa_obf_pairs_test.jsonl"
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/aa_imp_pairs_test.jsonl"
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_obf_pairs_test.jsonl"
   "msc-dissertation/experiments/pairs_balanced_json_flan_and_ori/at_imp_pairs_test.jsonl"
)

declare -a TEST_NAMES=(
   "AA"
   "AT"
   "AA_obf"
   "AA_imp"
   "AT_obf"
   "AT_imp"
)

# Tau values (you need to adjust these based on your validation results)
declare -A TAUS=(
   ["distilbert"]=0.5
   ["roberta"]=0.5
   ["bigbird"]=0.5
)

# Run error analysis for each model and test set combination
for i in "${!MODELS[@]}"; do
   MODEL_PATH="${MODELS[$i]}"
   MODEL_NAME="${MODEL_NAMES[$i]}"
   TAU="${TAUS[$MODEL_NAME]}"
   
   echo "Processing $MODEL_NAME..."
   
   for j in "${!TEST_FILES[@]}"; do
       TEST_FILE="${TEST_FILES[$j]}"
       TEST_NAME="${TEST_NAMES[$j]}"
       
       OUTPUT_FILE="error_analysis_results/${MODEL_NAME}_${TEST_NAME}_errors.json"
       
       echo "  Analyzing ${TEST_NAME}..."
       
       python extract_errors.py \
           --model_dir "$MODEL_PATH" \
           --test_file "$TEST_FILE" \
           --tau "$TAU" \
           --n_samples 500 \
           --output "$OUTPUT_FILE"
       
       if [ $? -ne 0 ]; then
           echo "  Error processing ${MODEL_NAME} on ${TEST_NAME}"
       else
           echo "  Completed ${MODEL_NAME} on ${TEST_NAME}"
       fi
   done
done

# Generate summary report
echo "Generating summary report..."
python - <<EOF
import json
import os
from collections import defaultdict

results_dir = "error_analysis_results"
summary = defaultdict(dict)

# Load all results
for filename in os.listdir(results_dir):
   if filename.endswith("_errors.json"):
       parts = filename.replace("_errors.json", "").split("_")
       model = parts[0]
       test_set = "_".join(parts[1:])
       
       with open(os.path.join(results_dir, filename), 'r') as f:
           data = json.load(f)
           summary[model][test_set] = {
               'error_rate': data['error_rate'],
               'patterns': data['patterns'],
               'fp_count': data['category_distribution']['FP'],
               'fn_count': data['category_distribution']['FN']
           }

# Create markdown report
report = "# Error Analysis Summary\n\n"

for model in summary:
   report += f"## {model.upper()}\n\n"
   report += "| Test Set | Error Rate | FP | FN | Platform Markers | Topic Overlap | Length Imbalance |\n"
   report += "|----------|------------|----|----|------------------|---------------|------------------|\n"
   
   for test_set in summary[model]:
       data = summary[model][test_set]
       patterns = data['patterns']
       report += f"| {test_set} | {data['error_rate']:.3f} | {data['fp_count']} | {data['fn_count']} | "
       report += f"{patterns.get('platform_markers', 0)} | {patterns.get('topic_overlap', 0)} | {patterns.get('length_imbalance', 0)} |\n"
   
   report += "\n"

with open(os.path.join(results_dir, "SUMMARY.md"), 'w') as f:
   f.write(report)

print("Summary report saved to error_analysis_results/SUMMARY.md")
EOF

echo "All error analyses complete!"
echo "Results saved in error_analysis_results/"