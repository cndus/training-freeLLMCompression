#!/bin/bash

#SBATCH --job-name=gsm8k_eval_debug
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --gres=gpu:2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate searchr12

cd /home/xhyin/search/training-free

echo "=== Testing API connection ==="
python -c "
from openai import OpenAI
client = OpenAI(
    api_key='sk-e4bca3558ebf46fb95f8850dc5caf152',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)
completion = client.chat.completions.create(
    model='qwen2.5-omni-7b',
    messages=[{'role': 'user', 'content': 'What is 2+2? Answer with just the number.'}],
    modalities=['text'],
    stream=True,
    stream_options={'include_usage': True},
)
chunks = []
for chunk in completion:
    if chunk.choices and chunk.choices[0].delta.content:
        chunks.append(chunk.choices[0].delta.content)
print('API OK, response:', ''.join(chunks))
"

echo ""
echo "=== Testing calc server connection ==="
python -c "
import requests
resp = requests.post(
    'http://192.168.102.17:8000/retrieve',
    json={'queries': ['12 * 8 + 5'], 'topk': 1, 'return_scores': True},
    timeout=5,
)
print('Calc server OK, result:', resp.json()['result'][0][0]['document']['contents'])
"

echo ""
echo "=== Running 5-sample evaluation ==="
python eval_gsm8k.py \
    --data_path ../Search-R1/data/gsm8k_calc/test.parquet \
    --calc_url http://192.168.102.17:8000/retrieve \
    --model qwen2.5-omni-7b \
    --max_turns 3 \
    --num_samples 5 \
    --output_dir ./results_debug

echo ""
echo "=== Done ==="
