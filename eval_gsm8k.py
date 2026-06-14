"""
Training-free GSM8K evaluation: Qwen2.5-Omni-7B API + calculator tool + LLM context compression.
Output: one CSV row per sample, columns = input, per-turn outputs, LLM compressions, gt, correct.
"""

import re, os, json, csv, argparse, requests, time
from typing import List, Dict
import pandas as pd
from openai import OpenAI
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Search-R1'))
from verl.utils.reward_score.gsm8k import compute_score_answer_tag


# ── API ───────────────────────────────────────────────────────────────────────

def make_client():
    return OpenAI(
        api_key="sk-e4bca3558ebf46fb95f8850dc5caf152",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def call_api(client, messages: List[Dict], model: str, stop: List[str] = None) -> str:
    is_omni = "omni" in model.lower()
    kwargs = dict(model=model, messages=messages,
                  stream=True, stream_options={"include_usage": True})
    if is_omni:
        kwargs["modalities"] = ["text"]
    else:
        kwargs["extra_body"] = {"enable_thinking": True}
    if stop:
        kwargs["stop"] = stop
    completion = client.chat.completions.create(**kwargs)
    think_chunks, answer_chunks = [], []
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            think_chunks.append(delta.reasoning_content)
        if hasattr(delta, "content") and delta.content:
            answer_chunks.append(delta.content)
    raw = "".join(answer_chunks)
    # wrap thinking in <think> tags so scoring and compression can use it
    if think_chunks:
        raw = "<think>\n" + "".join(think_chunks) + "\n</think>\n" + raw
    if stop:
        # </answer> takes priority: if answer tag is open anywhere, close it and ignore calculate
        if re.search(r'<answer>[^<]*$', raw):
            raw += "</answer>"
        elif re.search(r'<calculate>[^<]*$', raw):
            # only attach </calculate> if there's no complete <answer> earlier in the text
            if not re.search(r'<answer>.*?</answer>', raw, re.DOTALL):
                raw += "</calculate>"
            else:
                # answer already complete earlier — strip the dangling <calculate>
                raw = re.sub(r'<calculate>[^<]*$', '', raw)
    return raw


# ── Calculator ────────────────────────────────────────────────────────────────

def call_calc(calc_url: str, expression: str) -> str:
    try:
        resp = requests.post(
            calc_url,
            json={"queries": [expression], "topk": 1, "return_scores": True},
            timeout=10,
        )
        return resp.json()["result"][0][0]["document"]["contents"]
    except Exception as e:
        return f"error: {e}"


# ── Context compression ───────────────────────────────────────────────────────

COMPRESS_PROMPT = """\
Math problem in progress. Summarize the reasoning so far in 2-3 sentences, \
incorporating the calculator result below. State only the key numbers found \
and what remains to be computed. Be extremely concise.

Calculator result: {calc_result}

Conversation so far:
{history}"""


def compress_context(client, model: str, messages: List[Dict], calc_result: str) -> str:
    # Only pass the last assistant response (truncated to 500 chars) to keep prompt short
    last_asst = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
    )
    last_asst_short = last_asst[:500] + ("..." if len(last_asst) > 500 else "")
    history = f"[ASSISTANT]: {last_asst_short}"
    raw = call_api(client, [{"role": "user", "content":
                              COMPRESS_PROMPT.format(history=history, calc_result=calc_result)}], model)
    # extract only the last <think>...</think> block if present, else use raw
    think_blocks = re.findall(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_blocks:
        return "<think>\n" + think_blocks[-1].strip() + "\n</think>"
    # fallback: strip to first 300 chars if no think block (avoid bloat)
    return raw.strip()[:300]


# ── Prompt patching ───────────────────────────────────────────────────────────

def patch_prompt(prompt: str) -> str:
    """Replace <search> tags with <calculate> tags in the prompt text."""
    prompt = re.sub(r'<search>', '<calculate>', prompt)
    prompt = re.sub(r'</search>', '</calculate>', prompt)
    prompt = re.sub(r'call a search engine by', 'call the calculator by', prompt)
    prompt = re.sub(r'You can search as many times as your want\.', '', prompt)
    return prompt


INVALID_MSG = (
    "\nNo valid action detected. "
    "If you want to calculate an expression, wrap it in <calculate> and </calculate>. "
    "If you have the final answer, wrap it in <answer> and </answer>.\n"
)


# ── Single-sample generation loop ─────────────────────────────────────────────

def run_single(client, model: str, calc_url: str, prompt: str, max_turns: int) -> Dict:
    original_prompt = prompt
    messages = [{"role": "user", "content": prompt}]
    # turn_data: list of {response, calc_query, calc_result, compression, context_chars}
    turn_data = []
    num_turns = 0
    compressions = 0

    for turn in range(max_turns + 1):
        context_chars = sum(len(m["content"]) for m in messages)
        response = call_api(client, messages, model, stop=["</calculate>", "</answer>"])
        messages.append({"role": "assistant", "content": response})

        td = {
            "response": response,
            "context_chars": context_chars,
            "calc_query": "",
            "calc_result": "",
            "compression": "",
        }
        num_turns = turn + 1

        if re.search(r'<answer>.*?</answer>', response, re.DOTALL):
            turn_data.append(td)
            break

        calc_match = re.search(r'<calculate>(.*?)</calculate>', response, re.DOTALL)
        if calc_match and turn < max_turns:
            query = calc_match.group(1).strip()
            result = call_calc(calc_url, query)
            compressed = compress_context(client, model, messages, result)
            compressions += 1
            td["calc_query"] = query
            td["calc_result"] = result
            td["compression"] = compressed
            messages = [
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": compressed},
            ]
        elif turn < max_turns:
            messages.append({"role": "user", "content": INVALID_MSG})

        turn_data.append(td)

    last_asst = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
    )
    return {
        "last_assistant": last_asst,
        "num_turns": num_turns,
        "compressions": compressions,
        "turn_data": turn_data,
        "avg_context_chars": round(
            sum(td["context_chars"] for td in turn_data) / len(turn_data)
        ) if turn_data else 0,
    }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def build_fieldnames(max_turns: int) -> List[str]:
    base = ["index", "question", "ground_truth", "correct",
            "num_turns", "compressions", "avg_context_chars"]
    for t in range(max_turns + 1):
        base += [f"turn_{t}_response", f"turn_{t}_calc_query",
                 f"turn_{t}_calc_result", f"turn_{t}_compression"]
    return base


def build_row(idx, question, gt, score, stats, max_turns) -> Dict:
    gt_str = gt.get("target", str(gt)) if isinstance(gt, dict) else str(gt)
    row = {
        "index": idx,
        "question": question,
        "ground_truth": gt_str,
        "correct": int(score > 0),
        "num_turns": stats["num_turns"],
        "compressions": stats["compressions"],
        "avg_context_chars": stats["avg_context_chars"],
    }
    for t in range(max_turns + 1):
        if t < len(stats["turn_data"]):
            td = stats["turn_data"][t]
            row[f"turn_{t}_response"] = td["response"]
            row[f"turn_{t}_calc_query"] = td["calc_query"]
            row[f"turn_{t}_calc_result"] = td["calc_result"]
            row[f"turn_{t}_compression"] = td["compression"]
        else:
            row[f"turn_{t}_response"] = ""
            row[f"turn_{t}_calc_query"] = ""
            row[f"turn_{t}_calc_result"] = ""
            row[f"turn_{t}_compression"] = ""
    return row


def extract_question(prompt: str) -> str:
    """Extract just the question text from the full prompt."""
    m = re.search(r'Question:\s*(.+?)$', prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt[:200]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../Search-R1/data/gsm8k_calc/test.parquet")
    parser.add_argument("--calc_url", default="http://192.168.102.17:8000/retrieve")
    parser.add_argument("--model", default="qwen2.5-omni-7b")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--retry_delay", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = f"gsm8k_{args.model.replace('/', '_')}_turns{args.max_turns}"
    csv_path = os.path.join(args.output_dir, f"{base}.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)
    print(f"Evaluating {len(df)} samples | model={args.model} | max_turns={args.max_turns}")
    print(f"Calc: {args.calc_url}  →  {csv_path}")

    client = make_client()
    fieldnames = build_fieldnames(args.max_turns)

    done_indices = set()
    if os.path.exists(csv_path):
        done_df = pd.read_csv(csv_path)
        done_indices = set(done_df["index"].tolist())
        print(f"Resuming: {len(done_indices)} samples already done")

    write_header = not os.path.exists(csv_path)
    results = []

    with open(csv_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for _, row in df.iterrows():
            idx = int(row["extra_info"]["index"])
            if idx in done_indices:
                continue

            raw_prompt = row["prompt"][0]["content"]
            prompt = patch_prompt(raw_prompt)
            question = extract_question(raw_prompt)
            ground_truth = row["reward_model"]["ground_truth"]

            for attempt in range(5):
                try:
                    stats = run_single(client, args.model, args.calc_url, prompt, args.max_turns)
                    break
                except Exception as e:
                    print(f"  [#{idx}] error attempt {attempt+1}: {e}")
                    time.sleep(args.retry_delay * (attempt + 1))
            else:
                print(f"  [#{idx}] SKIPPED")
                continue

            score = compute_score_answer_tag(stats["last_assistant"], ground_truth)
            results.append({
                "correct": float(score > 0),
                "num_turns": stats["num_turns"],
                "compressions": stats["compressions"],
                "avg_context_chars": stats["avg_context_chars"],
            })

            writer.writerow(build_row(idx, question, ground_truth, score, stats, args.max_turns))
            fout.flush()

            if len(results) % 50 == 0 or len(results) == 1:
                acc = sum(r["correct"] for r in results) / len(results)
                avg_t = sum(r["num_turns"] for r in results) / len(results)
                avg_c = sum(r["compressions"] for r in results) / len(results)
                print(f"  [{len(results)}/{len(df)}] acc={acc:.3f}  avg_turns={avg_t:.2f}"
                      f"  avg_comp={avg_c:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    full = pd.read_csv(csv_path)
    n = len(full)
    turn_dist = full["num_turns"].value_counts().sort_index().to_dict()

    summary = {
        "model": args.model,
        "max_turns": args.max_turns,
        "num_samples": n,
        "accuracy": float(full["correct"].mean()),
        "avg_turns": float(full["num_turns"].mean()),
        "avg_compressions": float(full["compressions"].mean()),
        "avg_context_chars_per_sample": float(full["avg_context_chars"].mean()),
        "turn_distribution": {str(k): int(v) for k, v in turn_dist.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Accuracy:                  {summary['accuracy']:.4f}  "
          f"({int(full['correct'].sum())}/{n})")
    print(f"  Avg turns:                 {summary['avg_turns']:.3f}")
    print(f"  Avg compressions:          {summary['avg_compressions']:.3f}")
    print(f"  Avg context chars/sample:  {summary['avg_context_chars_per_sample']:.0f}")
    print(f"  Turn distribution:         {turn_dist}")
    print(f"\nSaved: {csv_path}")
    print(f"       {summary_path}")


if __name__ == "__main__":
    main()
