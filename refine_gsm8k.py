"""
Refine LLaDA on GSM8K math problems.
"""

import os
import sys
import json
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset
from llada_sample import llada_sample
from utils import load_model_and_tokenizer, build_output_paths
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torch.distributed as dist

class RefineDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_sample_history(batch_history, sample_idx):
    """Extract history for a single sample from batch history."""
    fix_mask = batch_history['prompt']['fix_mask'][sample_idx]

    sample_history = {
        'prompt': {
            'fix_mask': fix_mask,
            'tokens': batch_history['prompt']['tokens'][sample_idx],
        },
        'steps': []
    }

    # Calculate offset for this sample in the flattened variable tensors
    batch_size = batch_history['prompt']['fix_mask'].shape[0]
    num_vars_per_sample = [
        (~batch_history['prompt']['fix_mask'][i]).sum().item()
        for i in range(batch_size)
    ]
    start_idx = sum(num_vars_per_sample[:sample_idx])
    end_idx = start_idx + num_vars_per_sample[sample_idx]

    # Extract data for each step
    for step in batch_history['steps']:
        remask_pos = step['remask_positions'][sample_idx]
        variable_mask = ~fix_mask
        conf_var = step['conf_variable'][start_idx:end_idx]
        prob_var = step['prob_variable'][start_idx:end_idx]
        sample_step = {
            'x0_variable': step['x0_variable'][start_idx:end_idx],
            'conf_variable': step['conf_variable'][start_idx:end_idx],
            'prob_variable': step.get('prob_variable')[start_idx:end_idx],
            'remask_positions': remask_pos,
            'remask_variable_positions': remask_pos[variable_mask],
            'avg_error_conf': float(conf_var.mean().item()),
            'avg_error_prob': float(prob_var.mean().item()),
        }

        sample_history['steps'].append(sample_step)

    return sample_history


def build_synthetic_history_two_stage(
    question_part,
    buggy_answer,
    refined_completion,
    error_positions,
    error_original_tokens,
    tokenizer,
    pad_id,
    stage2_sample_history=None,
    refined_reasoning_str=None,
):
    """
    Build a history dict that matches single_stage structure so visualization works.
    Variable part = full answer (buggy then refined).
    If stage2_sample_history and refined_reasoning_str are provided, build one step per
    stage-2 denoising step (so the slider shows multiple steps); otherwise one step buggy -> refined.
    """
    prompt_tokens = tokenizer(
        question_part, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    buggy_tokens = tokenizer(
        buggy_answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    refined_tokens = tokenizer(
        refined_completion, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    n_var = len(buggy_tokens)  # original content length (before padding)
    n_ref = len(refined_tokens)
    steps = []

    if stage2_sample_history and refined_reasoning_str and stage2_sample_history.get("steps"):
        reasoning_tokens = tokenizer(
            refined_reasoning_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        n_reasoning = len(reasoning_tokens)
        max_num_len = 0
        for step in stage2_sample_history["steps"]:
            nt = step["x0_variable"]
            n_num = nt.shape[0] if isinstance(nt, torch.Tensor) else len(nt)
            max_num_len = max(max_num_len, n_num)
        n_max = max(n_var, n_ref, n_reasoning + max_num_len)

        for step in stage2_sample_history["steps"]:
            num_tokens = step["x0_variable"]
            if not isinstance(num_tokens, torch.Tensor):
                num_tokens = torch.tensor(num_tokens, dtype=torch.long)
            n_num = num_tokens.shape[0]
            var_full = torch.cat([reasoning_tokens, num_tokens])
            if var_full.shape[0] < n_max:
                var_full = torch.cat([
                    var_full,
                    torch.full((n_max - var_full.shape[0],), pad_id, dtype=torch.long),
                ])
            elif var_full.shape[0] > n_max:
                var_full = var_full[:n_max]
            remask_pos = torch.zeros(len(prompt_tokens) + n_max, dtype=torch.bool)
            variable_mask = torch.cat([
                torch.ones(len(prompt_tokens), dtype=torch.bool),
                torch.zeros(n_max, dtype=torch.bool),
            ]) == 0
            # Use real confidence from stage2 for number part; reasoning part has no conf so use 1.0
            conf_num = step.get("conf_variable")
            if isinstance(conf_num, torch.Tensor) and conf_num.numel() >= n_num:
                conf_num = conf_num[:n_num].float()
            elif isinstance(conf_num, torch.Tensor):
                conf_num = torch.ones(n_num, dtype=torch.float32)
            else:
                conf_num = torch.ones(n_num, dtype=torch.float32)
            prob_num = step.get("prob_variable")
            if isinstance(prob_num, torch.Tensor) and prob_num.numel() >= n_num:
                prob_num = prob_num[:n_num].float()
            else:
                prob_num = torch.ones(n_num, dtype=torch.float32)
            conf_full = torch.cat([
                torch.ones(n_reasoning, dtype=torch.float32),
                conf_num,
            ])
            prob_full = torch.cat([
                torch.ones(n_reasoning, dtype=torch.float32),
                prob_num,
            ])
            if conf_full.shape[0] < n_max:
                conf_full = torch.cat([
                    conf_full,
                    torch.ones(n_max - conf_full.shape[0], dtype=torch.float32),
                ])
            else:
                conf_full = conf_full[:n_max]
            if prob_full.shape[0] < n_max:
                prob_full = torch.cat([
                    prob_full,
                    torch.ones(n_max - prob_full.shape[0], dtype=torch.float32),
                ])
            else:
                prob_full = prob_full[:n_max]
            steps.append({
                "x0_variable": var_full,
                "conf_variable": conf_full,
                "prob_variable": prob_full,
                "remask_positions": remask_pos,
                "remask_variable_positions": remask_pos[len(prompt_tokens):],
                "avg_error_conf": float(step.get("avg_error_conf", 0.0)),
                "avg_error_prob": float(step.get("avg_error_prob", 0.0)),
            })
    if not steps:
        n_max = max(n_var, n_ref)
        if n_var < n_max:
            buggy_tokens = torch.cat([
                buggy_tokens,
                torch.full((n_max - n_var,), pad_id, dtype=torch.long),
            ])
        if n_ref < n_max:
            refined_tokens = torch.cat([
                refined_tokens,
                torch.full((n_max - n_ref,), pad_id, dtype=torch.long),
            ])
        remask_positions = torch.zeros(len(prompt_tokens) + n_max, dtype=torch.bool)
        steps = [{
            "x0_variable": refined_tokens,
            "conf_variable": torch.ones_like(refined_tokens, dtype=torch.float32),
            "prob_variable": torch.ones_like(refined_tokens, dtype=torch.float32),
            "remask_positions": remask_positions,
            "remask_variable_positions": remask_positions[len(prompt_tokens):],
            "avg_error_conf": 0.0,
            "avg_error_prob": 0.0,
        }]

    # initial_tokens: prompt + buggy (padded to n_max)
    if n_var < n_max:
        buggy_padded = torch.cat([
            tokenizer(buggy_answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0],
            torch.full((n_max - n_var,), pad_id, dtype=torch.long),
        ])
    else:
        buggy_padded = buggy_tokens[:n_max] if len(buggy_tokens) > n_max else buggy_tokens
        if len(buggy_padded) < n_max:
            buggy_padded = torch.cat([
                buggy_padded,
                torch.full((n_max - len(buggy_padded),), pad_id, dtype=torch.long),
            ])
    initial_tokens = torch.cat([prompt_tokens, buggy_padded])
    fix_mask = torch.cat([
        torch.ones(len(prompt_tokens), dtype=torch.bool),
        torch.zeros(n_max, dtype=torch.bool),
    ])
    variable_mask = ~fix_mask
    for s in steps:
        v = s["x0_variable"]
        if v.shape[0] != n_max:
            if v.shape[0] < n_max:
                s["x0_variable"] = torch.cat([v, torch.full((n_max - v.shape[0],), pad_id, dtype=torch.long)])
            else:
                s["x0_variable"] = v[:n_max]
        s["remask_positions"] = torch.zeros_like(initial_tokens, dtype=torch.bool)
        s["remask_variable_positions"] = s["remask_positions"][variable_mask]

    error_positions = [int(p) for p in error_positions if 0 <= p < n_var]
    error_original_tokens = list(error_original_tokens)[: len(error_positions)]

    return {
        "prompt": {"fix_mask": fix_mask, "tokens": initial_tokens},
        "steps": steps,
        "error_positions": error_positions,
        "error_original_tokens": error_original_tokens,
        "variable_content_length": n_var,
    }


def parse_tokens_from_path(file_path):
    """Parse tokens value from file path like .../tokens128/..."""
    path_str = str(file_path)
    match = re.search(r'/tokens(\d+)/', path_str)
    if match:
        return int(match.group(1))
    # Fallback: try to extract from max_new_tokens if available
    return None


def load_input_samples(results_file):
    """Load failed samples from evaluated results (only items with test_passed=False)."""
    failed_samples = []
    total = 0

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            if item['test_passed']:
                continue

            failed_samples.append({
                'task_id': item.get('task_id', ''),
                'question': item.get('question', ''),
                'prompt': item.get('prompt', ''),  # Full prompt with question
                'answer': item.get('answer', '') or item.get('canonical_answer', ''),
                'error_message': item.get('error_message', ''),
                'error_positions': item.get('error_positions', []),
                'question_part': item.get('question_part', item.get('prompt', '')),  # Question part (fixed context)
                'buggy_answer': item.get('buggy_answer', ''),  # Answer part with errors (variable)
                'error_original_tokens': item.get('error_original_tokens', []),
            })

    return failed_samples, total


def strip_answer_prefix(text: str) -> str:
    """Remove a leading 'Answer:' prefix if present."""
    if not text:
        return ""
    return re.sub(r"^\s*Answer\s*:\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def normalize_question_part(question_part: str) -> str:
    """Normalize question part to canonical '<question>||' format."""
    q = (question_part or "").strip()
    if "||" in q:
        q = q.split("||", 1)[0].strip()
    return q + "||"


def extract_corrected_solution(text: str) -> str:
    """Extract corrected-solution segment from eval_diff decoded text."""
    if not text:
        return ""
    marker = "Corrected solution:"
    s = text.rsplit(marker, 1)[-1] if marker in text else text
    s = s.strip()

    # Aggressively remove echoed prompt fragments (both line-wise and inline).
    s = re.sub(
        r"Question:.*?(?=(Corrected solution:|<<|####|$))",
        " ",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(
        r"Buggy solution:.*?(?=(Corrected solution:|<<|####|$))",
        " ",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(
        r"Please correct.*?(?=(Corrected solution:|<<|####|$))",
        " ",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"Corrected solution:", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    # If prompt style separator appears, keep the tail as answer region
    if "||" in s:
        s = s.rsplit("||", 1)[-1].strip()

    # Prefer the math-solution region if present
    if "<<" in s:
        s = s[s.find("<<"):].strip()
    elif "####" in s:
        s = s[s.find("####"):].strip()
    else:
        # If no math-answer marker exists, treat as no valid answer region.
        return ""
    return s


@torch.no_grad()
def diff_sample_eval_style(
    model,
    prompt_ids,
    steps,
    temperature,
    cfg_scale,
    context_length,
    mask_id,
    device,
    eps=1e-5,
    return_history=False,
):
    """Evaluate-style diffusion sampling (same core logic as eval/gen_model_answer.py diff_sample alg='greddy')."""
    batch_size = prompt_ids.shape[0]
    x = torch.full((batch_size, context_length), mask_id, dtype=torch.long).to(device)
    prompt_len = min(prompt_ids.shape[1], context_length)
    x[:, :prompt_len] = prompt_ids[:, :prompt_len]

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    step_history = []
    for i in range(steps):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[:, :prompt_len] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits_cat = model(x_cat)
                logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
                logits = logits[mask_index]
                un_logits = un_logits[mask_index]
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x)[mask_index]

        # alg == 'greddy'
        x0 = torch.argmax(logits, dim=-1)
        logits64 = logits.to(torch.float64)
        probs = F.softmax(logits64, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)

        t = timesteps[i]
        s = timesteps[i + 1]
        num_mask_token = int(mask_index.sum().item())
        number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
        if number_transfer_tokens <= 0:
            continue

        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
        x0_fill = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_id
        x0_fill[transfer_index] = x0[transfer_index].clone()
        x[mask_index] = x0_fill
        if return_history:
            step_history.append(x.clone().detach().cpu())

    if return_history:
        return x, step_history
    return x


def run_eval_style_two_stage_generation(
    model,
    tokenizer,
    questions,
    steps,
    temperature,
    cfg1,
    cfg2,
    context_length,
    mask_id,
    device,
    return_history=False,
):
    """Same two-pass generation pattern as evaluate_gsm8k.get_diff_sample()."""
    question_ids = tokenizer(
        questions, padding="longest", truncation=True, return_tensors="pt"
    )["input_ids"].to(device)

    prefix_ids = diff_sample_eval_style(
        model=model,
        prompt_ids=question_ids,
        steps=steps,
        temperature=temperature,
        cfg_scale=cfg1,
        context_length=context_length,
        mask_id=mask_id,
        device=device,
    )
    prefix_text = tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)

    prefix_ids = tokenizer(
        prefix_text, padding="longest", truncation=True, return_tensors="pt"
    )["input_ids"].to(device)
    answer_out = diff_sample_eval_style(
        model=model,
        prompt_ids=prefix_ids,
        steps=steps,
        temperature=temperature,
        cfg_scale=cfg2,
        context_length=context_length,
        mask_id=mask_id,
        device=device,
        return_history=return_history,
    )
    if return_history:
        answer_ids, step_history = answer_out
        return tokenizer.batch_decode(answer_ids, skip_special_tokens=True), step_history
    return tokenizer.batch_decode(answer_out, skip_special_tokens=True)


def split_buggy_answer(buggy_answer: str):
    """
    Split buggy_answer into reasoning part (before ####) and number part (#### + rest).
    Matches evaluate_gsm8k two-stage: stage1 = reasoning, stage2 = #### number.
    Returns (reasoning_part, number_part). If no '####', number_part is ''.
    """
    buggy_answer = strip_answer_prefix(buggy_answer)
    if not buggy_answer or "####" not in buggy_answer:
        return buggy_answer.strip(), ""
    idx = buggy_answer.find("####")
    reasoning_part = buggy_answer[:idx].strip()
    number_part = buggy_answer[idx:].strip()  # "#### 123" or "####123"
    if not number_part.startswith("####"):
        number_part = "#### " + number_part
    return reasoning_part, number_part


def normalize_final_number_part(s: str) -> str:
    """
    Normalize stage2 decoded output to clean "#### <number>" (no extra spaces/symbols).
    Takes the first token after "####" or the first number-like token.
    """
    if not s or not s.strip():
        return ""
    s = strip_answer_prefix(s).strip()
    if "####" in s:
        rest = s[s.find("####") + 4:].strip()
        first = rest.split()[0] if rest else ""
    else:
        first = s.split()[0] if s.split() else ""
    first = first.strip(".,;:!? \t\n\r")
    if not first:
        return ""
    return "#### " + first


def build_refine_prompt_remove_all(question_part, add_correction_instruction=False):
    """
    Build refine prompt (fixed context only). Format matches sft/gsm8k_data.py:
    prompt = question + "||", completion = <<...>> #### number (no "Answer: ").

    If add_correction_instruction is True, prepend a short instruction so the model knows this is a
    correction task and must output <<...>> #### <final_number> with the number consistent with the steps.
    """
    if not add_correction_instruction:
        return question_part
    instruction = (
        "Correct the solution below. Use steps like <<expr=result>> and end with #### followed by the final number. "
    )
    return instruction + question_part
    
def collate_fn_refine(
    batch, tokenizer, mask_id, pad_id, refine_setting='remove_all',
    add_correction_instruction=False
):
    """
    Collate function for refining GSM8K answers.
    
    Context structure:
        [question (context part)] + [answer (completion part)]
        |<----- Fixed part (fix_mask=True) ----->|  |<-- Variable (False) -->|
    
    Args:
        batch: List of failed samples, where:
            'question_part' is the question (fixed part)
            'buggy_answer' is the failed generated answer (variable part)
        tokenizer: Tokenizer
        mask_id: Mask token ID
        pad_id: Padding token ID
    """
    task_ids = [item['task_id'] for item in batch]

    # Only support remove_all and remove_all_without_initialization
    if refine_setting not in ('remove_all', 'remove_all_without_initialization'):
        raise ValueError(
            f"Invalid refine_setting: {refine_setting}. "
            "Only 'remove_all' and 'remove_all_without_initialization' "
            "are supported."
        )
    
    # Build refine prompts and tokenize
    refine_contexts = []
    buggy_answers = []
    question_parts = []
    error_positions_list = []
    error_original_tokens_list = []
    refine_context_tokens_list = []
    buggy_answer_tokens_list = []
    
    for item in batch:
        question_part = normalize_question_part(item.get('question_part', item.get('prompt', '')))
        buggy_answer = strip_answer_prefix(item.get('buggy_answer', ''))
        canonical_answer = strip_answer_prefix(item.get('answer') or item.get('canonical_answer', ''))

        # Build refine prompt (question_part = question + "||"; optional correction instruction)
        refine_prompt = build_refine_prompt_remove_all(
            question_part, add_correction_instruction=add_correction_instruction
        )
        refine_contexts.append(refine_prompt)
        buggy_answers.append(buggy_answer)
        question_parts.append(question_part)
        error_original_tokens_list.append(item.get('error_original_tokens', []))
        
        # Tokenize without special tokens to avoid BOS/EOS in the middle
        context_tokens = tokenizer(
            refine_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        # For answer tokens: use mask_id if remove_all_without_initialization,
        # otherwise use actual buggy_answer tokens
        if refine_setting == 'remove_all_without_initialization':
            # First tokenize to get length, then replace with mask_id
            answer_tokens_len = len(tokenizer(
                buggy_answer, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]) if buggy_answer else 0
            answer_tokens = torch.full(
                (answer_tokens_len,), mask_id, dtype=torch.long
            )
            # No canonical to compare; keep stored error_positions and clamp later
            error_positions_list.append(item.get('error_positions', []))
        else:
            answer_tokens = tokenizer(
                buggy_answer, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0] if buggy_answer else torch.tensor([], dtype=torch.long)
            # Recompute error_positions from canonical vs buggy with this tokenizer so they align
            canon_ids = tokenizer(
                canonical_answer, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist() if canonical_answer else []
            buggy_ids = answer_tokens.tolist()
            n_c, n_b = len(canon_ids), len(buggy_ids)
            err_pos = [i for i in range(min(n_c, n_b)) if canon_ids[i] != buggy_ids[i]]
            err_pos.extend(range(n_c, n_b))
            error_positions_list.append(sorted(set(p for p in err_pos if 0 <= p < n_b)))
        
        refine_context_tokens_list.append(context_tokens)
        buggy_answer_tokens_list.append(answer_tokens)

    remask_only_variable_indices = [item.get('remask_only_positions') for item in batch]
    if all(r is None for r in remask_only_variable_indices):
        remask_only_variable_indices = None
    else:
        # Use [] for any sample that didn't specify (allow all variable positions for that sample)
        remask_only_variable_indices = [r if r is not None else [] for r in remask_only_variable_indices]
    
    # Collate sequences and masks
    input_ids_list = []
    fix_mask_list = []
    attention_mask_list = []
    
    for context_tokens, answer_tokens in zip(
        refine_context_tokens_list, buggy_answer_tokens_list
    ):
        full_seq = torch.cat([context_tokens, answer_tokens])
        fix_mask = torch.cat([
            torch.ones(len(context_tokens), dtype=torch.bool),  # Fixed context
            torch.zeros(len(answer_tokens), dtype=torch.bool),    # Variable answer
        ])
        attn_mask = torch.ones(len(full_seq), dtype=torch.bool)
        
        input_ids_list.append(full_seq)
        fix_mask_list.append(fix_mask)
        attention_mask_list.append(attn_mask)
    
    max_len = max(seq.shape[0] for seq in input_ids_list)
    
    for i in range(len(input_ids_list)):
        current_len = input_ids_list[i].shape[0]
        if current_len < max_len:
            pad_len = max_len - current_len
            input_ids_list[i] = torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                input_ids_list[i]
            ])
            fix_mask_list[i] = torch.cat([
                torch.ones((pad_len,), dtype=torch.bool),
                fix_mask_list[i]
            ])
            attention_mask_list[i] = torch.cat([
                torch.zeros((pad_len,), dtype=torch.bool),
                attention_mask_list[i]
            ])
    
    return {
        'input_ids': torch.stack(input_ids_list).long(),
        'fix_mask': torch.stack(fix_mask_list),
        'attention_mask': torch.stack(attention_mask_list),
        'task_ids': task_ids,
        'refine_contexts': refine_contexts,
        'buggy_answers': buggy_answers,
        'question_parts': question_parts,
        'error_positions': error_positions_list,
        'error_original_tokens': error_original_tokens_list,
        'remask_only_variable_indices': remask_only_variable_indices,
    }

def refine_main(
    initial_results_file,
    model_name,
    batch_size,
    refined_steps,
    algorithm,
    temperature,
    refine_setting,
    confidence_threshold=None,
    output_prefix="correction",
    mad_k=2.5,
    skip_existing=False,
    add_correction_instruction=False,
    refine_mode="two_stage",
    sampler_backend="llada",
    eval_cfg1=0.1,
    eval_cfg2=0.1,
    eval_context_length=256,
    eval_question_prefix=True,
    eval_use_buggy_context=True,
):
    """Refine failed GSM8K samples. refine_mode: 'single_stage' (one denoise) or 'two_stage' (reasoning then #### number, like evaluate_gsm8k)."""
    # Load model and tokenizer
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    
    # Build output paths first to check if output file exists
    paths = build_output_paths(
        initial_results_file=initial_results_file,
        refined_steps=refined_steps,
        refine_setting=refine_setting,
        algorithm=algorithm,
        local_rank=local_rank,
        world_size=world_size,
        output_prefix=output_prefix,
        confidence_threshold=confidence_threshold,
        mad_k=mad_k,
        temperature=temperature
    )
    output_file = paths['output_file']
    history_dir = paths['history_dir']
    history_file = paths['history_file']
    
    # Check if output file exists and skip if requested
    if skip_existing and os.path.exists(output_file):
        print(
            f"[Rank {local_rank}] Output file {output_file} already exists, "
            f"skipping refinement (--skip_existing enabled)"
        )
        return
    
    model, tokenizer, pad_id, mask_id = load_model_and_tokenizer(model_name, device=device, local_rank=local_rank)
    
    # Load failed samples
    print(f"[Rank {local_rank}] Loading failed samples from {initial_results_file}...")
    failed_samples, total_evaluated = load_input_samples(initial_results_file)
    print(
        f"[Rank {local_rank}] Loaded {total_evaluated} evaluated samples, "
        f"{len(failed_samples)} failed (test_passed=False) → will refine those"
    )
    
    if len(failed_samples) == 0:
        raise ValueError("No failed samples to refine!")
    
    # Distribute data across GPUs
    per_gpu = len(failed_samples) // world_size
    start_idx = local_rank * per_gpu
    if local_rank < world_size - 1:
        end_idx = start_idx + per_gpu
    else:
        end_idx = len(failed_samples)
    local_dataset = failed_samples[start_idx:end_idx]

    print(f"[Rank {local_rank}] Processing {len(local_dataset)} samples")

    if sampler_backend == "eval_diff":
        print(
            f"[Rank {local_rank}] Using eval_diff backend "
            f"(cfg1={eval_cfg1}, cfg2={eval_cfg2}, context_length={eval_context_length})"
        )
        results = []
        all_histories = []
        for s in tqdm(
            range(0, len(local_dataset), batch_size),
            desc=f"[Rank {local_rank}] Eval-style refining"
        ):
            batch_items = local_dataset[s:s + batch_size]
            task_ids = [item.get('task_id', '') for item in batch_items]
            buggy_answers = [strip_answer_prefix(item.get('buggy_answer', '')) for item in batch_items]

            questions = []
            for i, item in enumerate(batch_items):
                q = (item.get('question') or "").strip()
                if not q:
                    qp = normalize_question_part(item.get('question_part', item.get('prompt', '')))
                    q = qp.split("||", 1)[0].strip()
                if eval_question_prefix and not q.startswith("Question:"):
                    q = "Question: " + q
                if eval_use_buggy_context:
                    b = buggy_answers[i].strip()
                    prompt = (
                        f"{q}\n"
                        f"Buggy solution: {b}\n"
                        "Please correct the buggy solution. "
                        "Output corrected steps and end with #### <final_answer>.\n"
                        "Corrected solution:"
                    )
                else:
                    prompt = q
                questions.append(prompt)

            generated, step_history = run_eval_style_two_stage_generation(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                steps=refined_steps,
                temperature=temperature,
                cfg1=eval_cfg1,
                cfg2=eval_cfg2,
                context_length=eval_context_length,
                mask_id=mask_id,
                device=device,
                return_history=True,
            )

            for i, text in enumerate(generated):
                refined_completion = extract_corrected_solution(text)
                if not refined_completion:
                    refined_completion = text.strip()

                question_part = normalize_question_part(
                    batch_items[i].get('question_part', batch_items[i].get('prompt', ''))
                )
                buggy_tokens = tokenizer(
                    buggy_answers[i], return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]

                per_step_solution_tokens = []
                for step_seq in step_history:
                    step_text = tokenizer.decode(step_seq[i].tolist(), skip_special_tokens=True)
                    sol = extract_corrected_solution(step_text)
                    st = tokenizer(sol, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    per_step_solution_tokens.append(st)

                # Keep variable length fixed to initial buggy answer length.
                n_var = len(buggy_tokens)
                if n_var == 0:
                    n_var = 1

                buggy_padded = buggy_tokens
                if len(buggy_padded) < n_var:
                    buggy_padded = torch.cat([
                        buggy_padded,
                        torch.full((n_var - len(buggy_padded),), pad_id, dtype=torch.long),
                    ])
                else:
                    buggy_padded = buggy_padded[:n_var]

                # Keep question/prompt in fixed region; variable region is answer-only.
                fixed_context_text = question_part + "Corrected solution: "
                fixed_context_tokens = tokenizer(
                    fixed_context_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                fixed_prefix_len = len(fixed_context_tokens)
                prompt_tokens = torch.cat([fixed_context_tokens, buggy_padded])
                prompt_fix_mask = torch.cat([
                    torch.ones(fixed_prefix_len, dtype=torch.bool),
                    torch.zeros(n_var, dtype=torch.bool),
                ])

                step_dicts = []
                for st in per_step_solution_tokens:
                    st_padded = st
                    if len(st_padded) < n_var:
                        st_padded = torch.cat([
                            st_padded,
                            torch.full((n_var - len(st_padded),), pad_id, dtype=torch.long),
                        ])
                    else:
                        st_padded = st_padded[:n_var]
                    remask_positions = torch.zeros(len(prompt_tokens), dtype=torch.bool)
                    step_dicts.append({
                        'x0_variable': st_padded,
                        'conf_variable': torch.ones(n_var, dtype=torch.float32),
                        'prob_variable': torch.ones(n_var, dtype=torch.float32),
                        'remask_positions': remask_positions,
                        'remask_variable_positions': remask_positions[fixed_prefix_len:],
                        'avg_error_conf': 0.0,
                        'avg_error_prob': 0.0,
                    })
                results.append({
                    'task_id': task_ids[i],
                    'original_buggy_answer': buggy_answers[i],
                    'refined_completion': refined_completion,
                    'completion': question_part + refined_completion,
                    'refine_context': questions[i],
                    'full_text': question_part + refined_completion,
                    'actual_steps': refined_steps,
                    'algorithm': f"eval_diff:greddy_cfg({eval_cfg1},{eval_cfg2})",
                    'steps': refined_steps,
                    'refine_mode': 'eval_style_two_stage',
                })
                all_histories.append({
                    'task_id': task_ids[i],
                    'backend': 'eval_diff',
                    'prompt': {
                        'tokens': prompt_tokens,
                        'fix_mask': prompt_fix_mask,
                    },
                    'steps': step_dicts,
                    'error_positions': batch_items[i].get('error_positions', []),
                    'error_original_tokens': [],
                    'variable_content_length': int(len(buggy_tokens)),
                })

        output_path = f"{output_file}.rank{local_rank}"
        print(f"[Rank {local_rank}] Saving results to {output_path}")
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        print(f"[Rank {local_rank}] Saving {len(all_histories)} histories to {history_file}")
        torch.save(all_histories, history_file)
        print(f"[Rank {local_rank}] Done!")

        if world_size > 1:
            sync_file = f"{output_file}.sync.rank{local_rank}"
            with open(sync_file, 'w') as f:
                f.write('done')

            if local_rank == 0:
                import time
                print("Waiting for all ranks to finish...")
                while True:
                    all_done = all(
                        os.path.exists(f"{output_file}.sync.rank{r}")
                        for r in range(world_size)
                    )
                    if all_done:
                        break
                    time.sleep(1)
                print("All ranks finished!")

        if local_rank == 0:
            print("Merging results from all ranks...")
            all_results = []
            for rank in range(world_size):
                rank_file = f"{output_file}.rank{rank}"
                if os.path.exists(rank_file):
                    with open(rank_file, 'r') as f:
                        for line in f:
                            all_results.append(json.loads(line))
                    os.remove(rank_file)

            with open(output_file, 'w') as f:
                for result in all_results:
                    f.write(json.dumps(result) + '\n')

            print(f"Saved {len(all_results)} refined results to {output_file}")
            print(f"Histories saved to {history_dir}/")

            for rank in range(world_size):
                sync_file = f"{output_file}.sync.rank{rank}"
                if os.path.exists(sync_file):
                    os.remove(sync_file)
        return

    # Create dataloader
    refine_dataset = RefineDataset(local_dataset)

    def collate_wrapper(batch):
        return collate_fn_refine(
            batch, tokenizer, mask_id, pad_id, refine_setting,
            add_correction_instruction=add_correction_instruction
        )

    dataloader = DataLoader(
        refine_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper
    )

    # Generate
    results = []
    all_histories = []

    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"[Rank {local_rank}] Refining"
    )):
        batch_size_actual = len(batch['task_ids'])
        task_ids = batch['task_ids']
        question_parts = batch['question_parts']
        buggy_answers = batch['buggy_answers']
        refine_contexts = batch['refine_contexts']
        error_positions_list = batch['error_positions']
        error_original_tokens_list = batch['error_original_tokens']

        if refine_mode == "two_stage":
            # Stage 1: refine reasoning (before ####); Stage 2: refine #### number (like evaluate_gsm8k)
            reasoning_parts = []
            number_parts = []
            for buggy_answer in buggy_answers:
                r, n = split_buggy_answer(buggy_answer)
                reasoning_parts.append(r)
                number_parts.append(n)

            # Build items for stage 1: prompt = question_part, variable = reasoning_part
            items_s1 = [
                {
                    'task_id': task_ids[i],
                    'question_part': question_parts[i],
                    'buggy_answer': reasoning_parts[i],
                    'answer': reasoning_parts[i],
                    'canonical_answer': reasoning_parts[i],
                    'error_positions': [],
                    'error_original_tokens': [],
                }
                for i in range(batch_size_actual)
            ]
            batch_s1 = collate_fn_refine(
                items_s1, tokenizer, mask_id, pad_id, refine_setting,
                add_correction_instruction=add_correction_instruction
            )
            input_ids_s1 = batch_s1['input_ids'].to(device).long()
            fix_mask_s1 = batch_s1['fix_mask'].to(device)
            attention_mask_s1 = batch_s1['attention_mask'].to(device)
            output_s1 = llada_sample(
                model=model,
                input_ids=input_ids_s1,
                fix_mask=fix_mask_s1,
                mask_id=mask_id,
                attention_mask=attention_mask_s1,
                steps=refined_steps,
                algorithm=algorithm,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                return_history=False,
                model_name=model_name,
                mad_k=mad_k,
            )
            sequences_s1 = output_s1['sequences']
            refined_reasoning_list = []
            for i in range(batch_size_actual):
                seq = sequences_s1[i].detach().cpu()
                fix_cpu = batch_s1['fix_mask'][i].bool()
                attn_cpu = batch_s1['attention_mask'][i].bool()
                ans_mask = (~fix_cpu) & attn_cpu
                ans_ids = seq[ans_mask]
                refined_reasoning_list.append(
                    tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True) if ans_ids.numel() > 0 else ""
                )

            # Stage 2: prompt = question_part + refined_reasoning (question||reasoning), variable = #### number
            # Only remask tokens after "####" (so we keep "####" fixed, remask space+number) to avoid "<<"
            # Use "####" not "#### " for prefix: "#### 123" may tokenize as ["####"," 123"] so "#### " prefix would give remask_only=[] and step2 would never correct
            refined_reasoning_list = [strip_answer_prefix(s) for s in refined_reasoning_list]
            num_prefix_tokens = len(tokenizer("####", return_tensors="pt", add_special_tokens=False)["input_ids"][0])
            items_s2 = []
            for i in range(batch_size_actual):
                num_part = number_parts[i] if number_parts[i] else " #### "
                num_ids = tokenizer(num_part, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                n_num = len(num_ids)
                remask_only = list(range(num_prefix_tokens, n_num)) if n_num > num_prefix_tokens else []
                items_s2.append({
                    'task_id': task_ids[i],
                    'question_part': question_parts[i].rstrip() + refined_reasoning_list[i],
                    'buggy_answer': num_part,
                    'answer': num_part,
                    'canonical_answer': num_part,
                    'error_positions': [],
                    'error_original_tokens': [],
                    'remask_only_positions': remask_only,
                })
            batch_s2 = collate_fn_refine(
                items_s2, tokenizer, mask_id, pad_id, refine_setting,
                add_correction_instruction=False
            )
            input_ids_s2 = batch_s2['input_ids'].to(device).long()
            fix_mask_s2 = batch_s2['fix_mask'].to(device)
            attention_mask_s2 = batch_s2['attention_mask'].to(device)
            output_s2 = llada_sample(
                model=model,
                input_ids=input_ids_s2,
                fix_mask=fix_mask_s2,
                mask_id=mask_id,
                attention_mask=attention_mask_s2,
                steps=refined_steps,
                algorithm=algorithm,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                return_history=True,
                model_name=model_name,
                mad_k=mad_k,
                remask_only_variable_indices=batch_s2.get('remask_only_variable_indices'),
            )
            sequences_s2 = output_s2['sequences']
            refined_number_list = []
            for i in range(batch_size_actual):
                seq = sequences_s2[i].detach().cpu()
                fix_cpu = batch_s2['fix_mask'][i].bool()
                attn_cpu = batch_s2['attention_mask'][i].bool()
                ans_mask = (~fix_cpu) & attn_cpu
                ans_ids = seq[ans_mask]
                raw = tokenizer.decode(ans_ids.tolist(), skip_special_tokens=True) if ans_ids.numel() > 0 else ""
                refined_number_list.append(normalize_final_number_part(raw))

            # Combine: full completion = reasoning + number (clean "#### <number>", no extra spaces/symbols)
            for i in range(batch_size_actual):
                r = refined_reasoning_list[i]
                n = refined_number_list[i]
                if n and not n.startswith("####"):
                    n = " #### " + n.strip()
                refined_completion = (r + " " + n).strip() if n else r
                refined_completion = strip_answer_prefix(refined_completion)

                result = {
                    'task_id': task_ids[i],
                    'original_buggy_answer': buggy_answers[i],
                    'refined_completion': refined_completion,
                    'completion': question_parts[i] + refined_completion,
                    'refine_context': refine_contexts[i],
                    'full_text': question_parts[i] + refined_completion,
                    'actual_steps': output_s2['actual_steps'][i].item(),
                    'algorithm': algorithm,
                    'steps': refined_steps,
                    'refine_mode': 'two_stage',
                }
                results.append(result)

            # Synthetic history: include stage2 steps so visualization shows multiple steps
            for i in range(batch_size_actual):
                r = refined_reasoning_list[i]
                n = refined_number_list[i]
                if n and not n.startswith("####"):
                    n = " #### " + n.strip()
                refined_completion_i = (r + " " + n).strip() if n else r
                refined_completion_i = strip_answer_prefix(refined_completion_i)
                stage2_sample = extract_sample_history(output_s2['history'], i) if output_s2.get('history') else None
                sample_history = build_synthetic_history_two_stage(
                    question_parts[i],
                    buggy_answers[i],
                    refined_completion_i,
                    error_positions_list[i],
                    error_original_tokens_list[i],
                    tokenizer,
                    pad_id,
                    stage2_sample_history=stage2_sample,
                    refined_reasoning_str=r,
                )
                sample_history['task_id'] = task_ids[i]
                all_histories.append(sample_history)

        else:
            # Single-stage: one denoise over full answer (original behavior)
            input_ids = batch['input_ids'].to(device).long()
            fix_mask = batch['fix_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = llada_sample(
                model=model,
                input_ids=input_ids,
                fix_mask=fix_mask,
                mask_id=mask_id,
                attention_mask=attention_mask,
                steps=refined_steps,
                algorithm=algorithm,
                temperature=temperature,
                confidence_threshold=confidence_threshold,
                return_history=True,
                model_name=model_name,
                mad_k=mad_k,
            )
            sequences = output['sequences']
            generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            for i in range(batch_size_actual):
                sample_history = extract_sample_history(output['history'], i)
                sample_history['task_id'] = task_ids[i]
                sample_history['error_positions'] = error_positions_list[i]
                sample_history['error_original_tokens'] = error_original_tokens_list[i]
                for step in sample_history['steps']:
                    conf_var = step['conf_variable'].cpu().numpy()
                    prob_var = step['prob_variable'].cpu().numpy()
                    n_var = len(conf_var)
                    err_pos = error_positions_list[i]
                    valid_positions = [p for p in err_pos if 0 <= p < n_var]
                    valid_confs = [conf_var[p] for p in valid_positions]
                    valid_probs = [prob_var[p] for p in valid_positions]
                    step['avg_error_conf'] = float(sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0
                    step['avg_error_prob'] = float(sum(valid_probs) / len(valid_probs)) if valid_probs else 0.0
                all_histories.append(sample_history)

            for i, (task_id, refine_ctx, buggy_answer, question_part, text) in enumerate(zip(
                task_ids, refine_contexts, buggy_answers, question_parts, generated_texts
            )):
                seq_tensor = sequences[i].detach().cpu()
                fix_mask_cpu = batch['fix_mask'][i].bool()
                attn_mask_cpu = batch['attention_mask'][i].bool()
                answer_mask = (~fix_mask_cpu) & attn_mask_cpu
                answer_token_ids = seq_tensor[answer_mask]
                refined_completion = (
                    tokenizer.decode(answer_token_ids.tolist(), skip_special_tokens=True)
                    if answer_token_ids.numel() > 0 else ""
                )
                refined_completion = strip_answer_prefix(refined_completion)
                result = {
                    'task_id': task_id,
                    'original_buggy_answer': buggy_answer,
                    'refined_completion': refined_completion,
                    'completion': question_part + refined_completion,
                    'refine_context': refine_ctx,
                    'full_text': question_part + refined_completion,
                    'actual_steps': output['actual_steps'][i].item(),
                    'algorithm': algorithm,
                    'steps': refined_steps,
                }
                results.append(result)

    # Save results (jsonl)
    output_path = f"{output_file}.rank{local_rank}"
    print(f"[Rank {local_rank}] Saving results to {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save histories (.pt)
    # History directory and file paths already defined above
    # (reuse the paths from the existence check)

    print(f"[Rank {local_rank}] Saving {len(all_histories)} histories to "
          f"{history_file}")
    torch.save(all_histories, history_file)

    print(f"[Rank {local_rank}] Done!")

    # Wait for all ranks (simple file-based synchronization)
    if world_size > 1:
        # Create sync file for this rank
        sync_file = f"{output_file}.sync.rank{local_rank}"
        with open(sync_file, 'w') as f:
            f.write('done')
        
        # Wait for all ranks to finish
        if local_rank == 0:
            import time
            print("Waiting for all ranks to finish...")
            while True:
                all_done = all(
                    os.path.exists(f"{output_file}.sync.rank{r}")
                    for r in range(world_size)
                )
                if all_done:
                    break
                time.sleep(1)
            print("All ranks finished!")

    # Merge results on rank 0
    if local_rank == 0:
        print("Merging results from all ranks...")
        all_results = []
        for rank in range(world_size):
            rank_file = f"{output_file}.rank{rank}"
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as f:
                    for line in f:
                        all_results.append(json.loads(line))
                os.remove(rank_file)

        # Save merged results
        with open(output_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        print(f"Saved {len(all_results)} refined results to {output_file}")
        print(f"Histories saved to {history_dir}/")
        
        # Clean up sync files
        for rank in range(world_size):
            sync_file = f"{output_file}.sync.rank{rank}"
            if os.path.exists(sync_file):
                os.remove(sync_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_results_file", type=str,
        default="eval_results/gsm8k_evaluated.jsonl",
        help="Path to evaluated results with failed samples"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="GSAI-ML/LLaDA-8B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--refined_steps", type=int, default=2,
        help="Number of refinement steps"
    )
    parser.add_argument(
        "--algorithm", type=str,
        default='self_conf-remask:vanilla'
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--refine_setting", type=str, default="remove_all",
        choices=["remove_all", "remove_all_without_initialization"],
        help=(
            "Refine setting: 'remove_all' (only function_head, "
            "body initialized with original code), or "
            "'remove_all_without_initialization' (only question_part, "
            "answer initialized with mask_id)"
        )
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Optional confidence threshold (e.g., 0.9) for remasking strategy"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="correction",
        help="Prefix for output directories (default: 'correction'). Use 'AR_correction' for AR experiments."
    )
    parser.add_argument(
        "--mad_k",
        type=float,
        default=2.5,
        help="Threshold multiplier for MAD algorithm (default: 2.5, typical values: 2.0, 2.5, 3.0). Only used when algorithm is 'self_conf-remask:vanilla_MAD'"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip refinement if output file already exists"
    )
    parser.add_argument(
        "--add_correction_instruction",
        action="store_true",
        help=(
            "Prepend a short instruction so the model knows this is a correction task "
            "and must output <<...>> #### <final_number>. May help fix final-answer errors."
        )
    )
    parser.add_argument(
        "--refine_mode",
        type=str,
        choices=["single_stage", "two_stage"],
        default="two_stage",
        help=(
            "single_stage: one denoise over full answer. "
            "two_stage: stage1 refine reasoning (before ####), stage2 refine #### number (like evaluate_gsm8k). Default: two_stage"
        )
    )
    parser.add_argument(
        "--sampler_backend",
        type=str,
        choices=["llada", "eval_diff"],
        default="llada",
        help="Sampling backend. 'llada' keeps current remask refinement; 'eval_diff' uses evaluate_gsm8k-style two-pass diffusion generation."
    )
    parser.add_argument(
        "--eval_cfg1",
        type=float,
        default=0.1,
        help="CFG scale for evaluate-style pass 1 (question -> prefix). Only for --sampler_backend eval_diff."
    )
    parser.add_argument(
        "--eval_cfg2",
        type=float,
        default=0.1,
        help="CFG scale for evaluate-style pass 2 (prefix -> answer). Only for --sampler_backend eval_diff."
    )
    parser.add_argument(
        "--eval_context_length",
        type=int,
        default=256,
        help="Context length for evaluate-style generation. Only for --sampler_backend eval_diff."
    )
    parser.add_argument(
        "--no_eval_question_prefix",
        action="store_true",
        help="Disable automatic 'Question: ' prefix for eval_diff backend."
    )
    parser.add_argument(
        "--no_eval_buggy_context",
        action="store_true",
        help="Disable adding buggy_answer context in eval_diff backend (not recommended for correction testing)."
    )

    args = parser.parse_args()

    refine_main(
        initial_results_file=args.initial_results_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        refined_steps=args.refined_steps,
        algorithm=args.algorithm,
        temperature=args.temperature,
        refine_setting=args.refine_setting,
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
        mad_k=args.mad_k,
        skip_existing=args.skip_existing,
        add_correction_instruction=args.add_correction_instruction,
        refine_mode=args.refine_mode,
        sampler_backend=args.sampler_backend,
        eval_cfg1=args.eval_cfg1,
        eval_cfg2=args.eval_cfg2,
        eval_context_length=args.eval_context_length,
        eval_question_prefix=not args.no_eval_question_prefix,
        eval_use_buggy_context=not args.no_eval_buggy_context,
    )
