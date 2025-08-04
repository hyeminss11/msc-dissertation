#!/bin/bash
#SBATCH --job-name=bigbird_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=26:00:00
#SBATCH --output=./logs/eval_log_2.4.txt
#SBATCH --error=./logs/eval_error_2.4.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
source activate myspark

# Run your script
python msc-dissertation/tools/bigbird_eval_trial2.4.py