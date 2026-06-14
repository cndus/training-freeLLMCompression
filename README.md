# Training-Free GSM8K Evaluation

A baseline for comparing against Search-R1 on the GSM8K math benchmark.
Uses the Qwen2.5-Omni-7B API with an optional calculator tool — **no fine-tuning or RL**.

## How it works

```
User prompt (same as Search-R1)
        │
        ▼
  ┌─────────────┐   <search>expr</search>   ┌──────────────┐
  │  Qwen2.5-   │ ─────────────────────────▶│  Calc Server │
  │  Omni-7B    │ ◀─────────────────────────│  (port 8000) │
  │  (API call) │   = result                └──────────────┘
  └─────────────┘
        │                 ▲
        │  context so far + calc result
        │                 │
        ▼           ┌─────────────┐
  LLM compresses ──▶│  Compressed │──▶ continue reasoning
  context            │  summary   │
                     └─────────────┘
        │
        ▼  <answer>N</answer>
      Score (exact float match)
```

**Key design:**
- The model is **not forced** to call the calculator. It calls it only if it decides to.
- If it does call `<calculate>`, generation is **intercepted** at `</calculate>`. The calc server evaluates the expression, then the LLM **compresses** the full conversation + result into a concise reasoning summary. This replaces the growing message history, keeping context short across turns.
- Stop tokens `["</search>", "</answer>"]` ensure the model cannot hallucinate tool results.

## Comparison with Search-R1

| Aspect | Search-R1 | Training-Free |
|--------|-----------|---------------|
| Model | Qwen2.5-1.5B-Instruct (RL fine-tuned) | Qwen API (no fine-tuning) |
| Tool use | Learned via PPO | Prompted |
| Context management | Fixed sliding window | LLM compression after each tool call |
| Prompt | Identical | Identical |
| Calc server | Same (`/retrieve` endpoint) | Same (`/retrieve` endpoint) |
| Metrics | accuracy, avg\_turns, context length | accuracy, avg\_turns, avg\_compressions, context length |

## Setup

The calculator server must be running before evaluation:
```bash
# In Search-R1 directory
sbatch calc_launch.sh
# Note the node IP from the job output, update --calc_url below
```

Install dependencies (in `searchr12` conda env):
```bash
pip install openai pandas requests
```

## Usage

**Quick debug (5 samples):**
```bash
sbatch run_eval_debug.sh
```

**Full evaluation (1319 test samples):**
```bash
sbatch run_eval.sh
```

**Custom run:**
```bash
python eval_gsm8k.py \
    --data_path ../Search-R1/data/gsm8k_calc/test.parquet \
    --calc_url http://<node_ip>:8000/retrieve \
    --model qwen2.5-omni-7b \
    --max_turns 3 \
    --num_samples 100 \        # omit for full test set
    --output_dir ./results
```

Evaluation supports **resume** — if interrupted, re-running the same command skips already-completed samples.

## Output

Results are written to `results/gsm8k_qwen2.5-omni-7b_turns3.jsonl` (one JSON per line):
```json
{
  "index": 0,
  "ground_truth": {"target": "18"},
  "score": 1.0,
  "num_turns": 2,
  "compressions": 1,
  "turn_stats": [
    {"turn": 0, "context_chars": 855, "response_chars": 312,
     "compressed": true, "calc_query": "16 - 3 - 4", "calc_result": "= 9"},
    {"turn": 1, "context_chars": 620, "response_chars": 89, "compressed": false}
  ],
  "messages": [...],
  "full_text": "..."
}
```

A summary is saved to `results/summary.json`:
```json
{
  "accuracy": 0.812,
  "avg_turns": 1.73,
  "avg_compressions": 0.65,
  "turn_distribution": {"1": 420, "2": 580, "3": 319},
  "avg_context_chars_per_turn": {"0": 855, "1": 620, "2": 710}
}
```

## Files

| File | Description |
|------|-------------|
| `eval_gsm8k.py` | Main evaluation script |
| `run_eval_debug.sh` | sbatch script for 5-sample debug run |
| `run_eval.sh` | sbatch script for full 1319-sample evaluation |
| `apicall_eg.py` | API usage example |
| `results/` | Output directory (created automatically) |
