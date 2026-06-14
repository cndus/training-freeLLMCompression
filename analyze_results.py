"""
Analyze eval CSV outputs from training-free experiments.
Computes accuracy, avg turns, compression length vs response length, per-turn stats.

Re-analysis mode (--reanalyze): treat first <answer> as the terminal turn —
discard subsequent turns for stats and re-score from the first answer found.
Saves a cleaned CSV alongside the original.

Usage:
  python analyze_results.py --csv results/gsm8k_*.csv
  python analyze_results.py --csv results_qwen3_6flash/gsm8k_qwen3.6-flash_turns3.csv --reanalyze
"""

import csv, re, argparse, json, os
from collections import defaultdict


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def get_turn_indices(rows, max_turns=4):
    if not rows:
        return []
    keys = rows[0].keys()
    return [t for t in range(max_turns + 1) if f'turn_{t}_response' in keys]


def extract_answer(text):
    """Return content of first <answer>...</answer>, or None."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip().replace(',', '').replace('$', '').strip() if m else None


def score_answer(pred_str, gt):
    if pred_str is None:
        return 0
    if isinstance(gt, dict):
        gt = gt.get('target', str(gt))
    gt_str = str(gt).strip().replace(',', '').replace('$', '').strip()
    try:
        return 1 if abs(float(pred_str) - float(gt_str)) < 1e-6 else 0
    except ValueError:
        return 1 if pred_str == gt_str else 0


def find_first_answer_turn(row, turns):
    """Return (turn_index, answer_str) for the first turn that has <answer>."""
    for t in turns:
        resp = row.get(f'turn_{t}_response', '')
        ans = extract_answer(resp)
        if ans is not None:
            return t, ans
    return None, None


def reanalyze_row(row, turns):
    """
    Return a modified row dict where:
    - correct/score are re-computed from the first answer turn
    - turns/compressions/avg_context_chars reflect only up to (and including) that turn
    - columns after the first-answer turn are cleared
    """
    first_ans_turn, ans_str = find_first_answer_turn(row, turns)
    new_row = dict(row)

    if first_ans_turn is None:
        # no answer found at all
        new_row['correct'] = 0
        new_row['score'] = 0.0
        new_row['num_turns'] = len(turns)
        return new_row

    gt = row.get('ground_truth', '')
    new_row['correct'] = score_answer(ans_str, gt)
    new_row['score'] = float(new_row['correct'])
    new_row['num_turns'] = first_ans_turn + 1

    # recount compressions up to first_ans_turn (inclusive)
    compressions = sum(
        1 for t in turns[:first_ans_turn + 1]
        if row.get(f'turn_{t}_calc_query', '')
    )
    new_row['compressions'] = compressions

    # recompute avg_context_chars up to first_ans_turn (inclusive)
    ctx_vals = [
        int(row.get(f'turn_{t}_context_chars') or 0)
        for t in turns[:first_ans_turn + 1]
        if row.get(f'turn_{t}_context_chars')
    ]
    if ctx_vals:
        new_row['avg_context_chars'] = round(sum(ctx_vals) / len(ctx_vals))

    # clear columns after first_ans_turn
    for t in turns[first_ans_turn + 1:]:
        new_row[f'turn_{t}_response'] = ''
        new_row[f'turn_{t}_calc_query'] = ''
        new_row[f'turn_{t}_calc_result'] = ''
        new_row[f'turn_{t}_compression'] = ''

    return new_row


def analyze(rows, turns):
    n = len(rows)
    correct = sum(int(r['correct']) for r in rows)

    turn_response_lens = defaultdict(list)
    turn_compression_lens = defaultdict(list)
    turn_calc_count = defaultdict(int)
    turn_answer_count = defaultdict(int)

    for r in rows:
        effective_turns = int(r.get('num_turns') or len(turns))
        for t in turns:
            if t >= effective_turns:
                break
            resp = r.get(f'turn_{t}_response', '')
            comp = r.get(f'turn_{t}_compression', '')
            calc_q = r.get(f'turn_{t}_calc_query', '')
            if resp:
                turn_response_lens[t].append(len(resp))
            if comp:
                turn_compression_lens[t].append(len(comp))
            if calc_q:
                turn_calc_count[t] += 1
            if re.search(r'<answer>.*?</answer>', resp, re.DOTALL):
                turn_answer_count[t] += 1

    all_comp = [v for lst in turn_compression_lens.values() for v in lst]
    all_resp = [v for lst in turn_response_lens.values() for v in lst]

    num_turns_dist = defaultdict(int)
    for r in rows:
        num_turns_dist[int(r.get('num_turns', 0))] += 1

    compressions_dist = defaultdict(int)
    for r in rows:
        compressions_dist[int(r.get('compressions', 0))] += 1

    avg_ctx = [float(r['avg_context_chars']) for r in rows if r.get('avg_context_chars')]

    result = {
        'num_samples': n,
        'accuracy': correct / n,
        'correct': correct,
        'avg_turns': sum(int(r.get('num_turns', 0)) for r in rows) / n,
        'avg_compressions': sum(int(r.get('compressions', 0)) for r in rows) / n,
        'avg_context_chars_per_sample': sum(avg_ctx) / len(avg_ctx) if avg_ctx else 0,
        'turn_distribution': dict(sorted(num_turns_dist.items())),
        'compressions_distribution': dict(sorted(compressions_dist.items())),
        'avg_compression_len': sum(all_comp) / len(all_comp) if all_comp else 0,
        'avg_response_len': sum(all_resp) / len(all_resp) if all_resp else 0,
        'compression_ratio': (sum(all_comp) / len(all_comp)) / (sum(all_resp) / len(all_resp))
                              if all_comp and all_resp else 0,
        'per_turn': {}
    }

    for t in turns:
        r_lens = turn_response_lens[t]
        c_lens = turn_compression_lens[t]
        result['per_turn'][t] = {
            'samples_reaching_turn': len(r_lens),
            'calc_calls': turn_calc_count[t],
            'answer_outputs': turn_answer_count[t],
            'avg_response_len': sum(r_lens) / len(r_lens) if r_lens else 0,
            'avg_compression_len': sum(c_lens) / len(c_lens) if c_lens else 0,
        }

    return result


def print_report(stats, label, reanalyzed=False):
    tag = " [re-analyzed]" if reanalyzed else ""
    print(f"\n{'='*55}")
    print(f"  {label}{tag}")
    print(f"{'='*55}")
    print(f"  Samples:                  {stats['num_samples']}")
    print(f"  Accuracy:                 {stats['accuracy']:.4f}  ({stats['correct']}/{stats['num_samples']})")
    print(f"  Avg turns:                {stats['avg_turns']:.3f}")
    print(f"  Avg compressions:         {stats['avg_compressions']:.3f}")
    print(f"  Avg context chars/sample: {stats['avg_context_chars_per_sample']:.0f}")
    print(f"  Avg response len (chars): {stats['avg_response_len']:.0f}")
    print(f"  Avg compression len:      {stats['avg_compression_len']:.0f}")
    print(f"  Compression/response:     {stats['compression_ratio']:.2f}x")
    print(f"\n  Turn distribution:        {stats['turn_distribution']}")
    print(f"  Compressions dist:        {stats['compressions_distribution']}")
    print(f"\n  Per-turn breakdown:")
    for t, ts in sorted(stats['per_turn'].items()):
        if ts['samples_reaching_turn'] == 0:
            continue
        print(f"    turn {t}: n={ts['samples_reaching_turn']:4d}  "
              f"calc={ts['calc_calls']:3d}  ans={ts['answer_outputs']:3d}  "
              f"avg_resp={ts['avg_response_len']:5.0f}  "
              f"avg_comp={ts['avg_compression_len']:5.0f}")
    print()


def save_reanalyzed_csv(rows, src_path):
    if not rows:
        return
    out_path = src_path.replace('.csv', '_reanalyzed.csv')
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved re-analyzed CSV: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs='+', required=True)
    parser.add_argument('--reanalyze', action='store_true',
                        help='Re-score using first <answer> turn only, discard later turns')
    parser.add_argument('--save_json', action='store_true')
    args = parser.parse_args()

    all_stats = {}
    for path in args.csv:
        if not os.path.exists(path):
            print(f'File not found: {path}')
            continue
        rows = load_csv(path)
        turns = get_turn_indices(rows)

        if args.reanalyze:
            rows = [reanalyze_row(r, turns) for r in rows]
            save_reanalyzed_csv(rows, path)

        stats = analyze(rows, turns)
        label = os.path.basename(path)
        print_report(stats, label, reanalyzed=args.reanalyze)
        all_stats[path] = stats

        if args.save_json:
            suffix = '_reanalyzed_analysis.json' if args.reanalyze else '_analysis.json'
            out = path.replace('.csv', suffix)
            with open(out, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f'  Saved JSON: {out}')

    if len(all_stats) > 1:
        print(f"\n{'='*55}")
        print("  Comparison")
        print(f"{'='*55}")
        print(f"  {'Model':<40} {'Acc':>6} {'Turns':>6} {'Comp':>5} {'CompLen':>8} {'RespLen':>8}")
        for path, s in all_stats.items():
            name = os.path.basename(path)[:40]
            print(f"  {name:<40} {s['accuracy']:>6.3f} {s['avg_turns']:>6.2f} "
                  f"{s['avg_compressions']:>5.2f} {s['avg_compression_len']:>8.0f} "
                  f"{s['avg_response_len']:>8.0f}")


if __name__ == '__main__':
    main()

