#!/bin/bash
#SBATCH --job-name=bb_tr_tt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=26:00:00
#SBATCH --output=./logs/train_tt_log.txt
#SBATCH --error=./logs/train_tt_error.txt
#SBATCH --mail-user=hjeong17@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

# Load modules and activate conda env
module load Anaconda3/2024.02-1
source activate myspark

# Run your script
python msc-dissertation/experiments/bigbird_experiment_trial2/tweet-tweet_trial2.3/bigbird_train_trial2_3_tt.py