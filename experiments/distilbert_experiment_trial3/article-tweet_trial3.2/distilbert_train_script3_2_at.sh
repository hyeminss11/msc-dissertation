#!/bin/bash
#SBATCH --job-name=dtb_tr_at
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=26:00:00
#SBATCH --output=./logs/train_distil_at_log.txt
#SBATCH --error=./logs/train_distil_at_error.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
source activate myspark

# Run your script
python msc-dissertation/experiments/distilbert_experiment_trial3/article-tweet_trial3.2/distilbert_train_trial3_2_at.py