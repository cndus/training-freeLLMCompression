"""
Training-free GSM8K evaluation: Qwen2.5-Omni-7B API + calculator tool + LLM context compression.
Output: one CSV row per sample, columns = input, per-turn outputs, LLM compressions, gt, correct.
"""

import re, os, json, csv, argparse, requests, time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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
COMPRESS_PROMPT2 = '''
You are a Working Memory Manager for a math agent. Your ONLY task is to compress the reasoning history into a highly concise memory state. 
You are NOT the problem solver. DO NOT solve the next step. DO NOT output the final answer.

[Original Question]: 
{question}

[Previous history]: 
{history}

[Calculator Result]: 
{calc_result}

INSTRUCTIONS:
Merge the [Previous history] and the [Calculator Result] into an updated Working Memory.
1. Output exactly 2 to 3 bullet points.
2. The bullets must contain ONLY explicit numbers, established facts, and what the latest calculation yielded.
3. STRICT RESTRICTIONS: Keep it strictly under 40 words total. Do NOT use `<think>` tags if possible. Direct output the bullet points.

Working Memory:
'''


# ── Math agent prompts ────────────────────────────────────────────────────────
# Turn 1: no working memory yet
MATH_PROMPT_INIT = """\
You are an efficient mathematical problem solver.
You are given a [Question]. Solve it step-by-step.

RULES:
1. CRITICAL: Your `<think>` process MUST be extremely concise (under 3 sentences or 50 words). Do not re-explain the problem.
2. NO DOUBLE-CHECKING: Trust your logic. Do not verify or double-check answers. Once you know the next step, act immediately.
3. After thinking, you MUST choose EXACTLY ONE of the following two actions:
   - Option A (Needs Calculation): Output EXACTLY ONE mathematical expression enclosed in `<calculate>...</calculate>`. DO NOT output anything else after this tag.
   - Option B (Final Answer Reached): Output ONLY the final numerical value enclosed in `<answer>...</answer>`.

[Question]:
{question}

Now, generate your next step:"""

# Turn 2+: has working memory from previous compression
MATH_PROMPT_WITH_MEMORY = """\
You are an efficient mathematical problem solver.
You are given a [Question] and your [Current Working Memory] which contains the summarized results of your previous steps.

Your goal is to solve the problem step-by-step.

RULES:
1. CRITICAL: Your `<think>` process MUST be extremely concise (under 3 sentences or 50 words). Do not recount the whole history.
2. NO DOUBLE-CHECKING: Trust the calculator and previous memory. Do not verify or double-check. 
3. After thinking, you MUST choose EXACTLY ONE of the following two actions:
   - Option A (Needs Calculation): Output EXACTLY ONE mathematical expression enclosed in `<calculate>...</calculate>`. DO NOT output anything else after this tag.
   - Option B (Final Answer Reached): Output ONLY the final numerical value enclosed in `<answer>...</answer>`.

[Question]:
{question}

[Current Working Memory]:
{working_memory}

Now, generate your next step:"""


def compress_context(client, model: str, messages: List[Dict], calc_result: str, question: str) -> str:
    last_asst = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
    )
    last_asst_short = last_asst[:500] + ("..." if len(last_asst) > 500 else "")
    history = f"[ASSISTANT]: {last_asst_short}"
    raw = call_api(client, [{"role": "user", "content":
                              COMPRESS_PROMPT2.format(history=history, calc_result=calc_result,
                                                      question=question)}], model)
    
    # 修复：移除所有 <think>...</think> 内容，只保留真正的 summary
    summary = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    
    # 兜底截断，防止异常生成
    return summary[:300]


INVALID_MSG = (
    "\nNo valid action detected. "
    "If you want to calculate an expression, wrap it in <calculate> and </calculate>. "
    "If you have the final answer, wrap it in <answer> and </answer>.\n"
)

def run_single(client, model: str, calc_url: str, question: str, max_turns: int) -> Dict:
    working_memory = None  # None on turn 1, set after each compression
    turn_data = []
    num_turns = 0
    compressions = 0

    for turn in range(max_turns + 1):
        # build fresh single-turn prompt each round
        if working_memory is None:
            prompt = MATH_PROMPT_INIT.format(question=question)
        else:
            prompt = MATH_PROMPT_WITH_MEMORY.format(question=question, working_memory=working_memory)

        messages = [{"role": "user", "content": prompt}]
        context_chars = len(prompt)
        response = call_api(client, messages, model, stop=["</calculate>", "</answer>"])

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
            compressed = compress_context(
                client, model,
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                result, question
            )
            compressions += 1
            td["calc_query"] = query
            td["calc_result"] = result
            td["compression"] = compressed
            working_memory = compressed
        elif turn < max_turns:
            # nudge with INVALID_MSG in same turn, don't consume a full turn slot
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": INVALID_MSG},
            ]
            retry = call_api(client, messages, model, stop=["</calculate>", "</answer>"])
            td["response"] = response + "\n[retry]\n" + retry
            if re.search(r'<answer>.*?</answer>', retry, re.DOTALL):
                turn_data.append(td)
                break
            calc_match2 = re.search(r'<calculate>(.*?)</calculate>', retry, re.DOTALL)
            if calc_match2 and turn < max_turns:
                query = calc_match2.group(1).strip()
                result = call_calc(calc_url, query)
                compressed = compress_context(
                    client, model,
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": retry}],
                    result, question
                )
                compressions += 1
                td["calc_query"] = query
                td["calc_result"] = result
                td["compression"] = compressed
                working_memory = compressed

        turn_data.append(td)

    last_asst = turn_data[-1]["response"] if turn_data else ""
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

    # build work list
    todo = []
    for _, row in df.iterrows():
        idx = int(row["extra_info"]["index"])
        if idx not in done_indices:
            todo.append((idx, row["question"].strip(), row["reward_model"]["ground_truth"]))

    print(f"Remaining: {len(todo)} samples to evaluate")

    write_lock = threading.Lock()
    results = []
    write_header = not os.path.exists(csv_path)

    def process_one(item):
        idx, question, ground_truth = item
        for attempt in range(5):
            try:
                stats = run_single(client, args.model, args.calc_url, question, args.max_turns)
                break
            except Exception as e:
                print(f"  [#{idx}] error attempt {attempt+1}: {e}")
                time.sleep(args.retry_delay * (attempt + 1))
        else:
            print(f"  [#{idx}] SKIPPED")
            return None
        score = compute_score_answer_tag(stats["last_assistant"], ground_truth)
        return (idx, question, ground_truth, score, stats)

    with open(csv_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                idx, question, ground_truth, score, stats = result
                results.append({
                    "correct": float(score > 0),
                    "num_turns": stats["num_turns"],
                    "compressions": stats["compressions"],
                    "avg_context_chars": stats["avg_context_chars"],
                })
                with write_lock:
                    writer.writerow(build_row(idx, question, ground_truth, score, stats, args.max_turns))
                    fout.flush()
                n_done = len(results)
                if n_done % 50 == 0 or n_done == 1:
                    acc = sum(r["correct"] for r in results) / n_done
                    avg_t = sum(r["num_turns"] for r in results) / n_done
                    avg_c = sum(r["compressions"] for r in results) / n_done
                    print(f"  [{n_done}/{len(todo)}] acc={acc:.3f}  avg_turns={avg_t:.2f}"
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
