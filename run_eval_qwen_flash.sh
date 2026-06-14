#!/bin/bash

#SBATCH --job-name=gsm8k_flash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate searchr12

cd /home/xhyin/search/training-free

python eval_gsm8k.py \
    --data_path ../Search-R1/data/gsm8k_calc/test.parquet \
    --calc_url http://192.168.102.14:8000/retrieve \
    --model qwen-flash \
    --max_turns 3 \
    --output_dir ./results_qwen_flash
