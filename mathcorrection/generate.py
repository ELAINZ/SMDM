# Adding number errors to GSM8K answers
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
import random
import argparse
import os
import torch

def check_token_consistency(buggy_ids, tokenizer) -> bool:
    """
    Check if decoding and re-encoding the buggy IDs results in the same token sequence.
    This ensures token positions are preserved when the code is processed downstream.
    Must use add_special_tokens=False when re-tokenizing to match how buggy_ids were built.
    """
    try:
        buggy_code = tokenizer.decode(buggy_ids, skip_special_tokens=True)
        retokenized_ids = tokenizer(
            buggy_code, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        return buggy_ids == retokenized_ids
    except Exception:
        return False

def _safe_eval_expr(expr: str):
    """Evaluate expr containing only digits, + - * / and spaces. Return str of result or None."""
    expr = expr.strip().replace(" ", "")
    if not expr or not re.match(r"^[\d+\-*/.()]+$", expr):
        return None
    try:
        # Avoid division by zero; use float for decimals
        val = eval(expr)
        return str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
    except Exception:
        return None


# Pairs of operators to swap for symbol errors (result changes)
_OPERATOR_SWAPS = [("+", "-"), ("-", "+"), ("*", "/"), ("/", "*")]


def _find_operator_positions(expr: str):
    """Find (start, end, op) for each + - * / in expr (no spaces in expr)."""
    expr = expr.strip().replace(" ", "")
    positions = []
    for m in re.finditer(r"([+\-*/])", expr):
        positions.append((m.start(), m.end(), m.group(1)))
    return positions


def _swap_one_operator(expr: str, op_index: int):
    """
    In expr, swap the operator at the given index (into _OPERATOR_SWAPS).
    Returns (new_expr, new_result_str) or (None, None) if invalid.
    """
    expr = expr.strip().replace(" ", "")
    positions = _find_operator_positions(expr)
    if op_index < 0 or op_index >= len(positions):
        return None, None
    start, end, op = positions[op_index]
    for a, b in _OPERATOR_SWAPS:
        if op == a:
            new_expr = expr[:start] + b + expr[end:]
            res = _safe_eval_expr(new_expr)
            if res is not None:
                return new_expr, res
            break
    return None, None


def _recompute_reasoning_chain(answer: str, replacement_map: dict = None, step_overrides: dict = None) -> str:
    """
    Recompute <<expr=result>> chain so it stays consistent.

    - replacement_map (number errors): original_number -> replacement. Re-eval each step's expr;
      if the step's result is one of replacement_map.values(), keep it (injected wrong value).
    - step_overrides (symbol errors): index -> (new_expr, new_result). Use that step's new expr/result,
      then for each later step substitute prior steps' new results into the expr and re-eval (propagate).
    """
    replacement_map = replacement_map or {}
    step_overrides = step_overrides or {}
    injected_values = set(replacement_map.values())
    if "<<" not in answer or ">>" not in answer or "####" not in answer:
        return answer
    text = answer.strip()
    final_num = ""
    if "####" in text:
        idx = text.rfind("####")
        rest = text[idx + 4 :].strip()
        final_num = rest.split()[0] if rest else ""
        text = text[:idx].strip()
    steps = re.findall(r"<<([^>]+)>>", text)
    if not steps:
        return answer

    # Parse to (expr, result) list
    steps_with_results = []
    for s in steps:
        if "=" not in s:
            steps_with_results.append((s, ""))
            continue
        expr, res = s.split("=", 1)
        steps_with_results.append((expr.strip(), res.strip()))

    # Symbol propagation: apply step_overrides then substitute and re-eval later steps
    if step_overrides:
        out = list(steps_with_results)
        orig_results = [steps_with_results[j][1] for j in range(len(steps_with_results))]
        for i, (new_expr, new_result) in step_overrides.items():
            if 0 <= i < len(out):
                out[i] = (new_expr, new_result)
        changed_indices = sorted(step_overrides.keys())
        if changed_indices:
            first_changed = min(changed_indices)
            for i in range(first_changed + 1, len(out)):
                expr = steps_with_results[i][0]
                for j in range(first_changed, i):
                    expr = re.sub(r"\b" + re.escape(orig_results[j]) + r"\b", out[j][1], expr)
                res_i = _safe_eval_expr(expr)
                if res_i is not None:
                    out[i] = (expr, res_i)
        new_parts = [f"<<{e}={r}>>" for e, r in out]
        new_final = out[-1][1] if out else final_num
        return " ".join(new_parts) + " #### " + new_final

    # Number case: re-eval each step, keep injected values
    new_parts = []
    for s in steps:
        if "=" not in s:
            new_parts.append(f"<<{s}>>")
            continue
        expr, old_res = s.split("=", 1)
        expr, old_res = expr.strip(), old_res.strip()
        new_res = _safe_eval_expr(expr)
        if old_res in injected_values and new_res is not None:
            new_parts.append(f"<<{expr}={old_res}>>")
        elif new_res is not None:
            new_parts.append(f"<<{expr}={new_res}>>")
        else:
            new_parts.append(f"<<{s}>>")
    last_step_res = new_parts[-1].split("=")[-1].rstrip(">>") if new_parts else final_num
    if final_num in injected_values:
        new_final = final_num
    else:
        new_final = last_step_res
    return " ".join(new_parts) + " #### " + new_final


def number_bug(answer: str, n_replace: int, tokenizer, propagate_mode: bool = False):
    """
    Inject errors by randomly replacing numbers in GSM8K answer.
    
    Args:
        answer: GSM8K answer text (format: "... #### number")
        n_replace: Number of numbers to replace (count of numbers, not tokens)
        tokenizer: Tokenizer to use
        propagate_mode: If True, propagate errors through reasoning chain
            (replace all occurrences of changed numbers in the reasoning chain)
            If False, only replace n random numbers (each number replaced once)
    
    Returns:
        tuple: (buggy_answer, error_positions, original_tokens) or (None, None, None)
    """
    n_replace = int(n_replace)
    
    if n_replace == 0:
        return answer, [], []
    
    # Find all numbers in the answer
    number_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    number_matches = list(number_pattern.finditer(answer))
    
    if len(number_matches) == 0:
        return None, None, None
    
    # Extract unique numbers (by text value) and their occurrences
    unique_numbers = {}
    for match in number_matches:
        number_text = match.group(1)
        if number_text not in unique_numbers:
            number_ids = tokenizer(number_text, add_special_tokens=False)["input_ids"]
            unique_numbers[number_text] = {
                'text': number_text,
                'token_len': len(number_ids),
                'token_ids': number_ids,
                'occurrences': []
            }
        unique_numbers[number_text]['occurrences'].append(match)
    
    # Convert to list for easier selection
    numbers_list = list(unique_numbers.values())
    
    if len(numbers_list) < n_replace:
        return None, None, None
    
    # Select n_replace unique numbers to replace
    selected_numbers_info = random.sample(numbers_list, min(n_replace, len(numbers_list)))
    
    # Build replacement map: original_number -> replacement_number
    replacement_map = {}
    original_tokens = []
    
    # Tokenize the full answer to track positions (no special tokens for alignment with buggy)
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    
    for num_info in selected_numbers_info:
        original_num = num_info['text']
        token_len = num_info['token_len']
        
        # Find replacement candidates with same token length
        candidates = [
            n['text'] for n in numbers_list
            if n['text'] != original_num 
            and n['token_len'] == token_len
            and n['text'] not in replacement_map.values()
        ]
        
        if not candidates:
            # Generate a different number with matching token length
            try:
                if '.' in original_num:
                    num_val = float(original_num)
                    if num_val != 0:
                        # Try different multipliers to find one with matching token length
                        multipliers = [0.5, 1.5, 2.0, 0.8, 1.2, 0.6, 1.4, 0.7, 1.3]
                        replacement_num = None
                        for mult in multipliers:
                            candidate = str(num_val * mult)
                            candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                            if len(candidate_ids) == token_len:
                                replacement_num = candidate
                                break
                        if replacement_num is None:
                            continue  # Skip if can't find matching length
                    else:
                        continue  # Skip zero
                else:
                    num_val = int(original_num)
                    if num_val != 0:
                        multipliers = [0.5, 1.5, 2.0, 0.8, 1.2, 0.6, 1.4]
                        replacement_num = None
                        for mult in multipliers:
                            candidate = str(int(num_val * mult))
                            candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                            if len(candidate_ids) == token_len:
                                replacement_num = candidate
                                break
                        if replacement_num is None:
                            continue  # Skip if can't find matching length
                    else:
                        continue  # Skip zero
            except:
                continue  # Skip if parsing fails
        else:
            replacement_num = random.choice(candidates)
        
        replacement_map[original_num] = replacement_num
        original_tokens.append(original_num)
    
    if not replacement_map:
        return None, None, None
    
    # Build buggy answer by replacing numbers
    buggy_answer = answer
    error_positions = []
    
    # Replace numbers in text
    for original_num, replacement_num in replacement_map.items():
        if propagate_mode:
            # Mode 2: Replace all occurrences of this number
            pattern = r'\b' + re.escape(original_num) + r'\b'
            buggy_answer = re.sub(pattern, replacement_num, buggy_answer)
        else:
            # Mode 1: Replace only first occurrence of this number
            pattern = r'\b' + re.escape(original_num) + r'\b'
            buggy_answer = re.sub(pattern, replacement_num, buggy_answer, count=1)
    
    # For <<...>> #### format: recompute so chain is consistent but keep our injected values (replacement_map)
    if "<<" in answer and ">>" in answer and "####" in answer:
        buggy_answer = _recompute_reasoning_chain(buggy_answer, replacement_map)
    
    # Tokenize buggy answer; error_positions = indices (in buggy answer token sequence) where buggy differs from original
    buggy_ids = tokenizer(buggy_answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    n_orig = len(answer_ids)
    n_buggy = len(buggy_ids)
    error_positions = []
    for i in range(min(n_orig, n_buggy)):
        if answer_ids[i] != buggy_ids[i]:
            error_positions.append(i)
    for i in range(n_orig, n_buggy):
        error_positions.append(i)
    # Align: only indices in [0, n_buggy), sorted and deduplicated (for downstream conf_var indexing)
    error_positions = sorted(set(p for p in error_positions if 0 <= p < n_buggy))
    
    # Final consistency check
    if not check_token_consistency(buggy_ids, tokenizer):
        return None, None, None
    
    return buggy_answer, error_positions, original_tokens


def symbol_bug(answer: str, n_replace: int, tokenizer):
    """
    Inject errors by swapping operators (e.g. + <-> -, * <-> /) in one or more steps
    of the reasoning chain, then recompute the chain so the result (and ####) becomes wrong.
    
    Args:
        answer: GSM8K answer text (format: "... <<expr=result>> ... #### number")
        n_replace: Number of steps in which to inject one symbol error (default 1)
        tokenizer: Tokenizer for consistency check and error_positions
    
    Returns:
        tuple: (buggy_answer, error_positions, original_tokens) or (None, None, None)
    """
    n_replace = max(1, int(n_replace))
    if "<<" not in answer or ">>" not in answer or "####" not in answer:
        return None, None, None

    text = answer.strip()
    final_num = ""
    if "####" in text:
        idx = text.rfind("####")
        rest = text[idx + 4 :].strip()
        final_num = rest.split()[0] if rest else ""
        text = text[: idx].strip()

    steps = re.findall(r"<<([^>]+)>>", text)
    steps_with_results = []
    for s in steps:
        if "=" not in s:
            continue
        expr, res = s.split("=", 1)
        expr, res = expr.strip(), res.strip()
        if _safe_eval_expr(expr) is not None:
            steps_with_results.append((expr, res))

    if len(steps_with_results) == 0:
        return None, None, None

    # Collect steps that have at least one operator we can swap
    candidate_indices = []
    for i, (expr, _) in enumerate(steps_with_results):
        positions = _find_operator_positions(expr)
        if positions:
            candidate_indices.append(i)

    if len(candidate_indices) < n_replace:
        n_replace = len(candidate_indices)
    if n_replace == 0:
        return None, None, None

    selected_step_indices = random.sample(candidate_indices, min(n_replace, len(candidate_indices)))
    original_tokens = []

    # Apply one symbol error per selected step; reuse _recompute_reasoning_chain with step_overrides
    current_answer = answer
    for step_idx in sorted(selected_step_indices):
        # Re-parse current answer to get steps (after previous iteration's propagation)
        text_cur = current_answer.strip()
        if "####" in text_cur:
            text_cur = text_cur[: text_cur.rfind("####")].strip()
        steps_cur = re.findall(r"<<([^>]+)>>", text_cur)
        steps_cur_list = []
        for s in steps_cur:
            if "=" not in s:
                continue
            expr, res = s.split("=", 1)
            expr, res = expr.strip(), res.strip()
            if _safe_eval_expr(expr) is not None:
                steps_cur_list.append((expr, res))
        if step_idx >= len(steps_cur_list):
            continue
        expr, old_res = steps_cur_list[step_idx]
        positions = _find_operator_positions(expr)
        if not positions:
            continue
        op_idx = random.randint(0, len(positions) - 1)
        new_expr, new_result = _swap_one_operator(expr, op_idx)
        if new_expr is None or new_result == old_res:
            continue
        original_tokens.append(expr)
        current_answer = _recompute_reasoning_chain(
            current_answer, replacement_map=None, step_overrides={step_idx: (new_expr, new_result)}
        )

    buggy_answer = current_answer

    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    buggy_ids = tokenizer(buggy_answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    n_orig, n_buggy = len(answer_ids), len(buggy_ids)
    error_positions = []
    for i in range(min(n_orig, n_buggy)):
        if answer_ids[i] != buggy_ids[i]:
            error_positions.append(i)
    for i in range(n_orig, n_buggy):
        error_positions.append(i)
    error_positions = sorted(set(p for p in error_positions if 0 <= p < n_buggy))

    if not check_token_consistency(buggy_ids, tokenizer):
        return None, None, None

    return buggy_answer, error_positions, original_tokens


def gsm8k_answer_to_reasoning_format(answer: str) -> str:
    """
    Convert GSM8K answer (with natural language) to training format: <<expr=result>> ... #### number only.
    Strips all prose so output is consistent with training data.
    """
    if not answer or not answer.strip():
        return answer
    text = answer.strip()
    # Extract final number after ####
    final_num = ""
    if "####" in text:
        idx = text.rfind("####")
        rest = text[idx + 4 :].strip()
        final_num = rest.split()[0] if rest else ""
        text = text[:idx].strip()
    # 1) Extract all <<expr=result>> in order (strip prose)
    existing = re.findall(r"<<([^>]+)>>", text)
    if existing:
        steps = [f"<<{s}>>" for s in existing if "=" in s]
        if steps:
            return " ".join(steps) + (" #### " + final_num if final_num else "")
    # 2) No <<>>: parse arithmetic from prose and build <<a op b=result>>
    steps = []
    # Two-number ops: 3+4=7, 48/2=24, 7*15=105 (optional spaces)
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)", text
    ):
        a, op, b, res = m.group(1), m.group(2), m.group(3), m.group(4)
        steps.append(f"<<{a}{op}{b}={res}>>")
    if steps:
        suffix = " #### " + final_num if final_num else " #### " + steps[-1].split("=")[-1].rstrip(">>")
        return " ".join(steps) + suffix
    if final_num:
        return "#### " + final_num
    return answer


def save_data(ds: str, type: str, orig_data:str, out_path: str, data_num = None, n_replace = 1, tokenizer=None, deduplicate=False, input_file=None, propagate_numbers=False):
    """
    Main process for adding number errors to GSM8K answers and save data to .jsonl file.
    data_num controls how many buggy variants to generate per original sample.
    """
    if orig_data != "gsm8k":
        raise ValueError(f"This version only supports GSM8K dataset, got: {orig_data}")
    
    if type not in ("number", "symbol"):
        raise ValueError(f"Unsupported error type: {type}. Use 'number' or 'symbol'.")
    
    variants_per_sample = 1 if data_num is None else max(1, int(data_num))
    all_items = []

    # Load samples from dataset (use "test" for openai/gsm8k, "train" for json data_files)
    if input_file:
        raise ValueError("Input file not supported for GSM8K in this version")
    else:
        samples = ds["test"] if "test" in ds else ds["train"]

    for idx, sample in enumerate(samples):
        question = sample["question"]
        raw_answer = sample["answer"]  # May contain prose + <<...>> and ####
        # Convert to training format (<<expr=result>> ... #### number only, no prose)
        answer = gsm8k_answer_to_reasoning_format(raw_answer)
        
        # Generate task_id if not present
        task_id = sample.get("task_id", f"gsm8k_{idx}")
        
        # Build prompt/completion to match sft/gsm8k_data.py: question + "||", completion = <<...>> #### number (no "Answer: ")
        question_part = question.strip() + "||"
        variants_generated = 0
        while variants_generated < variants_per_sample:
            if type == "number":
                buggy_answer_raw, error_positions, error_original_tokens = number_bug(
                    answer, n_replace, tokenizer, propagate_mode=propagate_numbers
                )
            else:
                assert type == "symbol"
                buggy_answer_raw, error_positions, error_original_tokens = symbol_bug(
                    answer, n_replace, tokenizer
                )

            if buggy_answer_raw is None:
                break

            buggy_answer = buggy_answer_raw  # completion format: <<...>> #### number, no "Answer: "
            canonical_answer = answer  # same format for error_positions alignment
            canon_ids = tokenizer(
                canonical_answer, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist()
            buggy_ids = tokenizer(
                buggy_answer, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist()
            n_c, n_b = len(canon_ids), len(buggy_ids)
            err_pos = [i for i in range(min(n_c, n_b)) if canon_ids[i] != buggy_ids[i]]
            err_pos.extend(range(n_c, n_b))
            error_positions = sorted(set(p for p in err_pos if 0 <= p < n_b))

            item = {
                "task_id": task_id,
                "question": question,
                "prompt": question_part,
                "answer": answer,  # Canonical answer (for eval/display)
                "canonical_answer": canonical_answer,
                "error_positions": error_positions,
                "error_original_tokens": error_original_tokens,
                "question_part": question_part,
                "buggy_answer": buggy_answer,
            }
            all_items.append(item)
            variants_generated += 1

    # Write all items to file
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Total samples generated: {len(all_items)}")
    print(f"Output file: {out_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k"],
        help="Dataset (only GSM8K supported)"
    )
    parser.add_argument(
        "--error_type",
        type=str,
        required=True,
        choices=["number", "symbol"],
        help="Error type: 'number' (replace numbers) or 'symbol' (swap +/-, */ in one or more steps; result and #### change)"
    )
    parser.add_argument(
        "--propagate_numbers", action="store_true",
        help="For 'number' error type only: propagate number errors through reasoning chain (replace all occurrences and recalculate answer)"
    )
    parser.add_argument(
        "--data_num", type=int,
        default=1,
        help="Number of buggy variants to generate per original sample"
    )
    parser.add_argument(
        "--n_replace", type=str,
        required=False,
        default=1,
        help="Number of numbers to replace"
    )
    parser.add_argument(
        "--data_path", type=str,
        required=False,
        default="mathcorrection",
        help="Path for saving data (default: mathcorrection)"
    )
    parser.add_argument(
        "--model_name", type=str,
        required=True,
        help="The model whose tokenizer will be used"
    )
    parser.add_argument(
        "--seed", type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip generation if output file already exists"
    )
    parser.add_argument(
        "--gsm8k_test_file",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to GSM8K test jsonl (same as evaluate_gsm8k.py). Default used for alignment."
    )
    parser.add_argument(
        "--use_hf_gsm8k",
        action="store_true",
        help="Use openai/gsm8k from HuggingFace instead of local data/gsm8k/test.jsonl"
    )

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    model_name_clean = os.path.basename(args.model_name)
    variants_per_sample = 1 if args.data_num is None else max(1, int(args.data_num))
    out_path = os.path.join(
        args.data_path,
        f"{args.dataset}/{model_name_clean}_{args.error_type}_{variants_per_sample}_wrong_{args.n_replace}.jsonl"
    )
    if args.propagate_numbers:
        out_path = os.path.join(
            args.data_path,
            f"{args.dataset}/{model_name_clean}_{args.error_type}_propagate_{variants_per_sample}_wrong_{args.n_replace}.jsonl"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Check if output file exists and skip if requested
    if args.skip_existing and os.path.exists(out_path):
        print(f"Output file {out_path} already exists, skipping generation (--skip_existing enabled)")
        return
    
    # For HF .pth paths (e.g. yiheng0824/smdm/cdlm_model.pth), model_name is not a valid
    # HuggingFace model id; use TinyLlama tokenizer (same as SMDM/MDM in utils.py)
    _mn = args.model_name.lower()
    if args.model_name.endswith(".pth") or "smdm" in _mn or "cdlm_model" in _mn or "mdm" in _mn:
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = 32000
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    if "open-dcoder" in args.model_name.lower():
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # Load GSM8K: default = data/gsm8k/test.jsonl (same as evaluate_gsm8k.py); --use_hf_gsm8k = openai/gsm8k
    if args.use_hf_gsm8k:
        print(f"Loading {args.dataset} from openai/gsm8k...")
        ds = load_dataset("openai/gsm8k", "main")
    else:
        test_file = args.gsm8k_test_file
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"GSM8K test file not found: {test_file} (same path as evaluate_gsm8k.py)")
        print(f"Loading GSM8K from local test file (aligned with evaluate_gsm8k.py): {test_file}")
        ds = load_dataset("json", data_files=test_file)
    
    save_data(ds, args.error_type, args.dataset, out_path, variants_per_sample, args.n_replace, tokenizer, False, None, args.propagate_numbers)


if __name__ == "__main__":
    main()
