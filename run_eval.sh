#!/bin/bash

#SBATCH --job-name=gsm8k_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB


source ~/anaconda3/etc/profile.d/conda.sh
conda activate searchr12

cd /home/xhyin/search/training-free

# Full test set (1319 samples)
python eval_gsm8k.py \
    --data_path ../Search-R1/data/gsm8k_calc/test.parquet \
    --calc_url http://192.168.102.17:8000/retrieve \
    --model qwen2.5-omni-7b \
    --max_turns 3 \
    --output_dir ./results
