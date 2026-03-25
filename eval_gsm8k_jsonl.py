"""
Evaluate GSM8K JSONL (refine results) using the same judgment as evaluate_gsm8k.get_acc.
"""

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.math_normalization import normalize_final_answer, check_sympy_equivalence


NUMBER_LIKE_RE = re.compile(
    r"[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:/\d+(?:\.\d+)?)?"
)


def extract_number_for_eval(text: str) -> str:
    """
    Robustly extract a number-like token for evaluation.
    Priority:
    1) first number-like token after the LAST '####'
    2) first number-like token from full text
    3) legacy first-whitespace-token fallback after LAST '####'
    """
    s = str(text).strip() if text else ""
    if not s:
        return ""

    tail = s.rsplit("####", 1)[1] if "####" in s else s
    m_tail = NUMBER_LIKE_RE.search(tail)
    if m_tail:
        return normalize_final_answer(m_tail.group(0).replace(",", ""))

    m_all = NUMBER_LIKE_RE.search(s)
    if m_all:
        return normalize_final_answer(m_all.group(0).replace(",", ""))

    # Legacy fallback (kept for compatibility with old outputs)
    pattern = r"####\s*(.*)$"
    preds = re.findall(pattern, s)
    rest = preds[-1].strip() if preds else ""
    tok = rest.split()[0].strip() if rest else ""
    return normalize_final_answer(tok)


def get_acc(pred, right_answer):
    """
    Extract number after last #### from pred and gt, normalize, then compare.
    gt may be full chain (<<...>> #### 18) or just the number ("18"); pred is refined_completion.
    """
    pred_num = extract_number_for_eval(pred)

    right_answer = str(right_answer).strip() if right_answer is not None else ""
    if "####" in right_answer:
        gt_num = extract_number_for_eval(right_answer)
    else:
        gt_num = extract_number_for_eval(right_answer)

    # Numeric equivalence so 18 vs 18.0 and string vs int don't cause false fail
    try:
        if pred_num and gt_num and float(pred_num) == float(gt_num):
            return True
    except (ValueError, TypeError):
        pass
    return check_sympy_equivalence(pred_num, gt_num)


def load_task_to_gt(initial_dataset_path: str) -> dict:
    """task_id -> ground truth (answer / canonical_answer / target)."""
    task_to_gt = {}
    path = Path(initial_dataset_path)
    if not path.exists():
        return task_to_gt
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tid = item.get("task_id")
            if tid is None:
                continue
            gt = item.get("answer") or item.get("canonical_answer") or item.get("target") or ""
            if gt is not None:
                task_to_gt[tid] = gt
    return task_to_gt


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K JSONL (same judgment as evaluate_gsm8k)")
    parser.add_argument("--results_file", type=str, required=True, help="JSONL with refined_completion / completion / buggy_answer")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Unused; for compatibility with callers")
    parser.add_argument("--initial_dataset", type=str, default=None,
                        help="JSONL for task_id -> gt; default = results_file")
    parser.add_argument("--skip_if_exist", action="store_true", help="Skip if _evaluated.jsonl exists")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    initial_path = args.initial_dataset or args.results_file
    task_to_gt = load_task_to_gt(initial_path)
    if not task_to_gt:
        print(f"Warning: no task_id -> gt from {initial_path}")

    out_path = results_path.parent / f"{results_path.stem}_evaluated.jsonl"
    if args.skip_if_exist and out_path.exists():
        print(f"Output exists, skipping: {out_path}")
        return

    passed = 0
    total = 0
    with open(results_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_id = item.get("task_id", "")
            pred_text = item.get("refined_completion") or item.get("completion") or item.get("buggy_answer") or ""
            gt_text = task_to_gt.get(task_id, item.get("answer") or item.get("canonical_answer") or "")

            ok = get_acc(pred_text, gt_text)
            if ok:
                passed += 1
            total += 1
            f_out.write(json.dumps({**item, "test_passed": ok, "error_message": "" if ok else "answer mismatch"}) + "\n")

    print(f"Evaluated {total} samples, passed {passed}, acc = {passed / total if total else 0:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
