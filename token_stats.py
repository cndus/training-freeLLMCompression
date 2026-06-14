"""
Compute per-turn token length stats for math agent (response) and
summary agent (compression) outputs.

Uses the Qwen2.5 tokenizer for accurate token counts.
Run:
    python3 token_stats.py --csv results_qwen3_7plus/gsm8k_qwen3.7-plus_turns3.csv
"""

import csv, re, argparse, json, os, sys
from collections import defaultdict

TOKENIZER_PATH = "/data1/public/hf/Qwen/Qwen2.5-1.5B-Instruct"

# load tokenizer once at module level
sys.path.insert(0, "/home/xhyin/search/Search-R1")
try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    def token_count(text: str) -> int:
        return len(_tokenizer.encode(text, add_special_tokens=False))
except Exception as e:
    print(f"[WARN] tokenizer load failed ({e}), falling back to char/4")
    def token_count(text: str) -> int:
        return max(1, len(text) // 4)


def stats(values):
    if not values:
        return {"count": 0, "mean": 0, "max": 0, "min": 0, "median": 0}
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "mean": round(sum(values) / n, 1),
        "max": values[-1],
        "min": values[0],
        "median": values[n // 2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"Loaded {len(rows)} rows from {os.path.basename(args.csv)}")

    # per-turn collections
    response_lens = defaultdict(list)   # math agent output
    compress_lens = defaultdict(list)   # summary agent output
    all_response_lens = []
    all_compress_lens = []

    for row in rows:
        effective_turns = int(row.get("num_turns") or args.max_turns)
        for t in range(args.max_turns + 1):
            resp = row.get(f"turn_{t}_response", "")
            comp = row.get(f"turn_{t}_compression", "")

            # only count turns up to the effective end turn
            if t >= effective_turns and not resp.strip():
                continue

            if resp.strip():
                n = token_count(resp)
                response_lens[t].append(n)
                all_response_lens.append(n)

            if comp.strip():
                n = token_count(comp)
                compress_lens[t].append(n)
                all_compress_lens.append(n)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Token length stats (Qwen2.5 tokenizer)")
    print(f"{'='*60}")

    print(f"\n  [Math Agent — response per turn]")
    print(f"  {'Turn':<8} {'Count':>6} {'Mean':>8} {'Median':>8} {'Max':>8} {'Min':>6}")
    print(f"  {'-'*50}")
    for t in sorted(response_lens.keys()):
        s = stats(response_lens[t])
        print(f"  turn {t:<4} {s['count']:>6} {s['mean']:>8.0f} {s['median']:>8} {s['max']:>8} {s['min']:>6}")
    s_all = stats(all_response_lens)
    print(f"  {'ALL':<8} {s_all['count']:>6} {s_all['mean']:>8.0f} {s_all['median']:>8} {s_all['max']:>8} {s_all['min']:>6}")

    print(f"\n  [Summary Agent — compression per turn]")
    print(f"  {'Turn':<8} {'Count':>6} {'Mean':>8} {'Median':>8} {'Max':>8} {'Min':>6}")
    print(f"  {'-'*50}")
    for t in sorted(compress_lens.keys()):
        s = stats(compress_lens[t])
        print(f"  turn {t:<4} {s['count']:>6} {s['mean']:>8.0f} {s['median']:>8} {s['max']:>8} {s['min']:>6}")
    if all_compress_lens:
        s_all_c = stats(all_compress_lens)
        print(f"  {'ALL':<8} {s_all_c['count']:>6} {s_all_c['mean']:>8.0f} {s_all_c['median']:>8} {s_all_c['max']:>8} {s_all_c['min']:>6}")
    else:
        print(f"  (no compressions found)")

    # compression ratio
    if all_compress_lens and all_response_lens:
        ratio = sum(all_compress_lens) / len(all_compress_lens) / (sum(all_response_lens) / len(all_response_lens))
        print(f"\n  Compression/Response ratio (mean chars): {ratio:.3f}x")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.save_json:
        result = {
            "file": os.path.basename(args.csv),
            "math_agent_response": {
                f"turn_{t}": stats(v) for t, v in sorted(response_lens.items())
            },
            "math_agent_response_all": stats(all_response_lens),
            "summary_agent_compression": {
                f"turn_{t}": stats(v) for t, v in sorted(compress_lens.items())
            },
            "summary_agent_compression_all": stats(all_compress_lens),
        }
        out = args.csv.replace(".csv", "_token_stats.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
