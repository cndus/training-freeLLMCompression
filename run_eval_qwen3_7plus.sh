#!/bin/bash

#SBATCH --job-name=gsm8k_q37plus
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --qos=ddl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source ~/anaconda3/etc/profile.d/conda.sh
conda activate searchr12

cd /home/xhyin/search/training-free

python eval_gsm8k.py \
    --data_path ../Search-R1/data/gsm8k_calc/test.parquet \
    --calc_url http://192.168.102.16:8000/retrieve \
    --model qwen3.7-plus \
    --max_turns 3 \
    --output_dir ./results_qwen3_7plus
