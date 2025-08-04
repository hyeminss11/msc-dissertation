#!/bin/bash
#SBATCH --job-name=bb_ev_aa
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/eval_aa_log.txt
#SBATCH --error=./logs/eval_aa_error.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
source activate myspark

# Run your script
python msc-dissertation/experiments/bigbird_experiment_trial2/article-article_trial2.1/bigbird_eval_trial2_1_aa.py