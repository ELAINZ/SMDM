#!/usr/bin/env python3
"""
Visualize differences between original cleaned_code and refined tokens.

Usage:
    python visualize_refine_diff.py <results_jsonl> <history_dir> [--output output.html] [--sample-ids 0,1,2] [--model MODEL]
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer
import difflib


def load_results(results_jsonl: Path) -> Dict[str, Dict]:
    """Load refined results and create task_id -> result mapping."""
    results_map = {}
    with open(results_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            task_id = item['task_id']
            results_map[task_id] = item
    return results_map


def load_histories(history_dir: Path) -> List[Dict[str, Any]]:
    """Load and merge histories from all .pt files."""
    pt_files = sorted(history_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {history_dir}")
    
    all_histories = []
    for pt_file in pt_files:
        histories = torch.load(pt_file, map_location='cpu')
        all_histories.extend(histories)
    
    return all_histories


def tokenize_code(tokenizer, code: str) -> List[int]:
    """Tokenize code string."""
    tokens = tokenizer(code, return_tensors="pt")["input_ids"][0]
    return tokens.tolist()


def decode_tokens(tokenizer, token_ids) -> List[str]:
    """Decode token IDs to strings."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [tokenizer.decode([tid]) for tid in token_ids]


def compute_token_diff(original_tokens: List[str], refined_tokens: List[str]) -> Dict[str, Any]:
    """Compute detailed diff between original and refined tokens."""
    # Compute statistics
    matcher = difflib.SequenceMatcher(None, original_tokens, refined_tokens)
    opcodes = list(matcher.get_opcodes())
    
    unchanged = 0
    insertions = 0
    deletions = 0
    replacements = 0
    
    for op, i1, i2, j1, j2 in opcodes:
        if op == 'equal':
            unchanged += (i2 - i1)
        elif op == 'insert':
            insertions += (j2 - j1)
        elif op == 'delete':
            deletions += (i2 - i1)
        elif op == 'replace':
            replacements += max((i2 - i1), (j2 - j1))
    
    total_original = len(original_tokens)
    total_refined = len(refined_tokens)
    
    return {
        'opcodes': opcodes,
        'statistics': {
            'unchanged': unchanged,
            'insertions': insertions,
            'deletions': deletions,
            'replacements': replacements,
            'total_original': total_original,
            'total_refined': total_refined,
            'retention_rate': unchanged / total_original if total_original > 0 else 0,
            'change_rate': (insertions + deletions + replacements) / total_original if total_original > 0 else 0,
        }
    }


def compute_step_changes(original_token_strs: List[str], history: Dict, tokenizer) -> List[Dict]:
    """Compute changes at each step relative to original."""
    initial_tokens = history['prompt']['tokens']
    fix_mask = history['prompt']['fix_mask']
    
    # Get original variable tokens (extract variable part from original code)
    # We need to extract only the variable part, which corresponds to the code body
    # For now, we'll compare the full sequences
    original_variable_strs = original_token_strs  # Full original
    
    step_changes = []
    
    # Initial state (before any refinement)
    initial_variable_strs = decode_tokens(tokenizer, initial_tokens[~fix_mask])
    initial_diff = compute_token_diff(original_variable_strs, initial_variable_strs)
    step_changes.append({
        'step': -1,
        'step_name': 'Initial',
        'diff': initial_diff,
    })
    
    # Each refine step
    for step_idx, step in enumerate(history['steps']):
        refined_variable_tokens = step['x0_variable']
        refined_variable_strs = decode_tokens(tokenizer, refined_variable_tokens)
        
        # Compare variable parts only
        step_diff = compute_token_diff(original_variable_strs, refined_variable_strs)
        
        step_changes.append({
            'step': step_idx,
            'step_name': f'Step {step_idx + 1}',
            'diff': step_diff,
        })
    
    return step_changes


def load_evaluated_results(evaluated_jsonl: Path) -> Dict[str, Dict]:
    """Load evaluated results and create task_id -> evaluated_result mapping."""
    evaluated_map = {}
    if not evaluated_jsonl or not evaluated_jsonl.exists():
        return evaluated_map
    
    with open(evaluated_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            task_id = item.get('task_id')
            if task_id:
                evaluated_map[task_id] = item
    return evaluated_map


def generate_diff_html(
    results_map: Dict[str, Dict],
    histories: List[Dict],
    tokenizer,
    output_path: Path,
    sample_ids: List[int] = None,
    task_ids: List[str] = None,
    evaluated_map: Dict[str, Dict] = None
):
    """Generate HTML visualization with diff comparison.
    
    Args:
        results_map: Mapping from task_id to result dict
        histories: List of history dicts
        tokenizer: Tokenizer for decoding
        output_path: Output HTML file path
        sample_ids: Optional list of sample indices (0-based)
        task_ids: Optional list of task_ids to visualize (takes precedence over sample_ids)
        evaluated_map: Optional mapping from task_id to evaluated result (for showing test_passed status)
    """
    
    # If task_ids provided, use them; otherwise use sample_ids
    if task_ids is not None:
        # Create a mapping from task_id to history index
        task_id_to_history_idx = {}
        for idx, history in enumerate(histories):
            hist_task_id = history.get('task_id')
            if hist_task_id:
                task_id_to_history_idx[hist_task_id] = idx
        
        # Filter histories by task_ids
        selected_indices = []
        for task_id in task_ids:
            if task_id in task_id_to_history_idx:
                selected_indices.append(task_id_to_history_idx[task_id])
            elif task_id in results_map:
                print(f"Warning: Task {task_id} found in results but not in histories")
        selected_indices.sort()
    elif sample_ids is not None:
        selected_indices = sample_ids
    else:
        selected_indices = list(range(len(histories)))
    
    # Prepare data
    samples_data = []
    for sample_id in selected_indices:
        if sample_id >= len(histories):
            continue
            
        history = histories[sample_id]
        task_id = history.get('task_id')
        
        if not task_id or task_id not in results_map:
            continue
        
        result = results_map[task_id]
        # GSM8K: original_buggy_answer; code tasks: original_cleaned_code
        is_gsm8k = 'original_buggy_answer' in result
        original_text = result.get('original_buggy_answer') or result.get('original_cleaned_code')
        if original_text is None:
            continue
        original_code = original_text
        
        # Get test_passed status from evaluated results if available
        test_passed = None
        if evaluated_map and task_id in evaluated_map:
            eval_result = evaluated_map[task_id]
            test_passed_val = eval_result.get('test_passed', False)
            # Handle both bool and string
            test_passed = (
                test_passed_val is True or 
                test_passed_val == 'true' or 
                test_passed_val == 'True'
            )
        
        # Get tokens from history
        fix_mask = history['prompt']['fix_mask']
        initial_tokens = history['prompt']['tokens']
        
        # Extract full prompt (fixed part - this includes padding and refine_context)
        # fix_mask=True corresponds to: padding tokens + refine_context tokens
        full_prompt_tokens = initial_tokens[fix_mask]
        full_prompt_strs = decode_tokens(tokenizer, full_prompt_tokens)
        
        # Extract real function_head from refine_context
        # refine_context format: ... + "Fixed code:\n" + function_head + "\n"
        refine_context = result.get('refine_context', '')
        real_function_head_strs = None
        if refine_context and 'Fixed code:\n' in refine_context:
            # Extract function_head after "Fixed code:\n"
            function_head_text = refine_context.split('Fixed code:\n', 1)[-1]
            # Remove trailing newline
            function_head_text = function_head_text.rstrip('\n')
            if function_head_text:
                real_function_head_tokens = tokenize_code(tokenizer, function_head_text)
                real_function_head_strs = decode_tokens(tokenizer, real_function_head_tokens)
        
        # Extract variable part from history (for step display and full sequence)
        initial_variable_tokens = initial_tokens[~fix_mask]
        initial_variable_strs = decode_tokens(tokenizer, initial_variable_tokens)
        if history['steps']:
            final_step = history['steps'][-1]
            final_variable_tokens = final_step['x0_variable']
        else:
            final_variable_tokens = initial_variable_tokens
        final_variable_strs_from_history = decode_tokens(tokenizer, final_variable_tokens)

        # For GSM8K: compute diff from result strings so we compare actual content
        # (history may have padding / different token alignment and show "all changed")
        refined_text = result.get('refined_completion', '')
        if is_gsm8k and refined_text:
            orig_ids = tokenize_code(tokenizer, original_code)
            ref_ids = tokenize_code(tokenizer, refined_text)
            original_variable_strs_for_diff = decode_tokens(tokenizer, orig_ids)
            final_variable_strs_for_diff = decode_tokens(tokenizer, ref_ids)
            final_diff = compute_token_diff(original_variable_strs_for_diff, final_variable_strs_for_diff)
            final_variable_strs = final_variable_strs_for_diff
            original_variable_strs_display = original_variable_strs_for_diff
        else:
            final_diff = compute_token_diff(initial_variable_strs, final_variable_strs_from_history)
            final_variable_strs = final_variable_strs_from_history
            original_variable_strs_display = initial_variable_strs

        original_tokens = tokenize_code(tokenizer, original_code)
        original_token_strs = decode_tokens(tokenizer, original_tokens)
        step_changes = compute_step_changes(initial_variable_strs, history, tokenizer)
        
        # Also store full tokens for side-by-side view
        if history['steps']:
            final_full_tokens = initial_tokens.clone()
            final_full_tokens[~fix_mask] = final_step['x0_variable']
        else:
            final_full_tokens = initial_tokens
        final_full_token_strs = decode_tokens(tokenizer, final_full_tokens)
        
        samples_data.append({
            'id': sample_id,
            'task_id': task_id,
            'test_passed': test_passed,  # None if not available, True/False if available
            'is_gsm8k': is_gsm8k,
            'original_tokens': original_token_strs,  # Full original text for reference
            'full_prompt_tokens': full_prompt_strs,  # Complete refine_context (fixed part)
            'real_function_head_tokens': real_function_head_strs,  # Code: function_head; GSM8K: None
            'original_variable_tokens': original_variable_strs_display,  # For diff: result-based when GSM8K
            'final_tokens': final_full_token_strs,
            'final_variable_tokens': final_variable_strs,
            'final_diff': final_diff,
            'step_changes': step_changes,
            'num_steps': len(history['steps']),
            'fix_mask': fix_mask.tolist(),
        })
    
    # Generate HTML
    html_content = generate_html_content(samples_data, tokenizer)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Diff visualization saved to: {output_path}")


def generate_html_content(samples_data: List[Dict], tokenizer) -> str:
    """Generate HTML content with diff visualization."""
    
    samples_json = json.dumps(samples_data, ensure_ascii=False)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Refine Diff Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: "Courier New", monospace; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        .controls {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .diff-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .token-display {{ padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 5px; max-height: 600px; overflow-y: auto; }}
        .token {{ display: inline-block; margin: 2px; padding: 4px 6px; border-radius: 3px; font-size: 13px; }}
        .token-unchanged {{ background: #f0f0f0; color: #333; }}
        .token-added {{ background: #c8e6c9; color: #2e7d32; }}
        .token-removed {{ background: #ffcdd2; color: #c62828; text-decoration: line-through; }}
        .token-modified {{ background: #fff9c4; color: #f57f17; }}
        .function-head {{ 
            display: block; 
            padding: 10px 15px; 
            margin: 10px 0; 
            background: #e3f2fd; 
            border: 2px solid #2196f3; 
            border-radius: 5px; 
            font-family: "Courier New", monospace;
            width: 100%;
            box-sizing: border-box;
        }}
        .function-head-label {{
            font-size: 11px; 
            color: #1976d2; 
            font-weight: bold; 
            margin-bottom: 5px;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }}
        .stat-card {{ padding: 10px; background: white; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #333; }}
        .step-changes {{ margin: 20px 0; }}
        .step-change-item {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        select, input {{ margin: 5px; padding: 5px 10px; border-radius: 3px; border: 1px solid #ddd; }}
        button {{ padding: 8px 15px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Refine Diff Visualization</h1>
        
        <div class="controls">
            <label>Sample: <select id="sampleSelect" onchange="updateVisualization()"></select></label>
            <button onclick="toggleViewMode()">Switch View Mode</button>
        </div>
        
        <div id="statsContainer" class="stats-grid"></div>
        
        <div id="diffView" class="diff-container"></div>
        
        <div id="stepChangesView" class="step-changes"></div>
        
        <div id="changePlot" style="width: 100%; height: 400px; margin-top: 20px;"></div>
    </div>
    
    <script>
        const data = {samples_json};
        let viewMode = 'side-by-side'; // 'side-by-side' or 'unified'
        
        function initialize() {{
            const select = document.getElementById("sampleSelect");
            data.forEach((sample, idx) => {{
                const option = document.createElement("option");
                option.value = idx;
                // Add test_passed status if available
                let statusLabel = "";
                if (sample.test_passed !== null && sample.test_passed !== undefined) {{
                    statusLabel = sample.test_passed ? " ✅" : " ❌";
                }}
                option.text = `Sample #${{sample.id}} (Task: ${{sample.task_id}}, Steps: ${{sample.num_steps}}${{statusLabel}})`;
                select.appendChild(option);
            }});
            updateVisualization();
        }}
        
        function toggleViewMode() {{
            viewMode = viewMode === 'side-by-side' ? 'unified' : 'side-by-side';
            updateVisualization();
        }}
        
        function updateVisualization() {{
            const sampleIdx = parseInt(document.getElementById("sampleSelect").value);
            const sample = data[sampleIdx];
            
            updateStats(sample);
            updateDiffView(sample);
            updateStepChanges(sample);
            updateChangePlot(sample);
        }}
        
        function updateStats(sample) {{
            const stats = sample.final_diff.statistics;
            const html = `
                <div class="stat-card">
                    <div class="stat-label">Retention Rate</div>
                    <div class="stat-value">${{(stats.retention_rate * 100).toFixed(1)}}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Change Rate</div>
                    <div class="stat-value">${{(stats.change_rate * 100).toFixed(1)}}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Unchanged</div>
                    <div class="stat-value">${{stats.unchanged}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Insertions</div>
                    <div class="stat-value">${{stats.insertions}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Deletions</div>
                    <div class="stat-value">${{stats.deletions}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Replacements</div>
                    <div class="stat-value">${{stats.replacements}}</div>
                </div>
            `;
            document.getElementById("statsContainer").innerHTML = html;
        }}
        
        function renderFixedBox(tokens, label, bgColor, borderColor) {{
            if (!tokens || !Array.isArray(tokens) || tokens.length === 0) {{
                return '';
            }}
            let html = '<div class="function-head" style="width: 100%; background: ' + bgColor + '; border-color: ' + borderColor + ';">';
            html += '<div class="function-head-label">' + label + '</div>';
            html += '<div style="word-wrap: break-word;">';
            tokens.forEach(token => {{
                const tokenStr = typeof token === 'string' ? token : String(token);
                html += '<span class="token" style="background: ' + bgColor + '; border: 1px solid ' + borderColor + '; margin: 2px;">' + escapeHtml(tokenStr) + '</span>';
            }});
            html += '</div></div>';
            return html;
        }}
        
        function updateDiffView(sample) {{
            const container = document.getElementById("diffView");
            
            // Render full prompt and real function head
            const fullPromptHtml = renderFixedBox(
                sample.full_prompt_tokens,
                '🔒 Full Prompt (Fixed - Unchanged)',
                '#e3f2fd',
                '#2196f3'
            );
            const realFunctionHeadHtml = sample.real_function_head_tokens ? renderFixedBox(
                sample.real_function_head_tokens,
                '📌 Real Function Head (Fixed - Unchanged)',
                '#c8e6c9',
                '#4caf50'
            ) : '';
            
            const varLabel = (sample.is_gsm8k ? 'Answer' : 'Code');
            const initialTitle = sample.is_gsm8k ? 'Initial Answer (Refine Start)' : 'Initial Code (Refine Start)';
            const finalTitle = sample.is_gsm8k ? 'Final Answer (Refine End)' : 'Final Code (Refine End)';
            if (viewMode === 'side-by-side') {{
                container.style.gridTemplateColumns = '1fr 1fr';
                container.innerHTML = `
                    <div class="token-display">
                        <h3>${{initialTitle}}</h3>
                        ${{fullPromptHtml}}
                        ${{realFunctionHeadHtml}}
                        <div style="margin-top: 10px;">
                            <strong>Variable ${{varLabel}}:</strong>
                            <div>${{renderTokens(sample.original_variable_tokens, sample.final_diff, 'original')}}</div>
                        </div>
                    </div>
                    <div class="token-display">
                        <h3>${{finalTitle}}</h3>
                        ${{fullPromptHtml}}
                        ${{realFunctionHeadHtml}}
                        <div style="margin-top: 10px;">
                            <strong>Variable ${{varLabel}}:</strong>
                            <div>${{renderTokens(sample.final_variable_tokens, sample.final_diff, 'refined')}}</div>
                        </div>
                    </div>
                `;
            }} else {{
                container.style.gridTemplateColumns = '1fr';
                container.innerHTML = `
                    <div class="token-display">
                        <h3>Unified Diff View</h3>
                        ${{fullPromptHtml}}
                        ${{realFunctionHeadHtml}}
                        <div style="margin-top: 10px;">
                            <strong>Variable ${{varLabel}}:</strong>
                            <div>${{renderUnifiedDiff(sample)}}</div>
                        </div>
                    </div>
                `;
            }}
        }}
        
        function renderTokens(tokens, diff, mode) {{
            const opcodes = diff.opcodes;
            let html = '';
            
            for (const [op, i1, i2, j1, j2] of opcodes) {{
                if (mode === 'original') {{
                    if (op === 'equal' || op === 'delete' || op === 'replace') {{
                        for (let i = i1; i < i2; i++) {{
                            const className = op === 'equal' ? 'token-unchanged' : (op === 'delete' ? 'token-removed' : 'token-modified');
                            html += '<span class="token ' + className + '">' + escapeHtml(tokens[i]) + '</span>';
                        }}
                    }}
                }} else {{
                    if (op === 'equal' || op === 'insert' || op === 'replace') {{
                        for (let j = j1; j < j2; j++) {{
                            const className = op === 'equal' ? 'token-unchanged' : (op === 'insert' ? 'token-added' : 'token-modified');
                            html += '<span class="token ' + className + '">' + escapeHtml(tokens[j]) + '</span>';
                        }}
                    }}
                }}
            }}
            
            return html.replace(/\\n/g, '<br>');
        }}
        
        function renderUnifiedDiff(sample) {{
            let html = '';
            const opcodes = sample.final_diff.opcodes;
            
            for (const [op, i1, i2, j1, j2] of opcodes) {{
                if (op === 'equal') {{
                    for (let i = i1; i < i2; i++) {{
                        html += '<span class="token token-unchanged">' + escapeHtml(sample.original_variable_tokens[i]) + '</span>';
                    }}
                }} else if (op === 'delete') {{
                    for (let i = i1; i < i2; i++) {{
                        html += '<span class="token token-removed">' + escapeHtml(sample.original_variable_tokens[i]) + '</span>';
                    }}
                }} else if (op === 'insert') {{
                    for (let j = j1; j < j2; j++) {{
                        html += '<span class="token token-added">' + escapeHtml(sample.final_variable_tokens[j]) + '</span>';
                    }}
                }} else if (op === 'replace') {{
                    for (let i = i1; i < i2; i++) {{
                        html += '<span class="token token-removed">' + escapeHtml(sample.original_variable_tokens[i]) + '</span>';
                    }}
                    for (let j = j1; j < j2; j++) {{
                        html += '<span class="token token-added">' + escapeHtml(sample.final_variable_tokens[j]) + '</span>';
                    }}
                }}
            }}
            
            return html.replace(/\\n/g, '<br>');
        }}
        
        function updateStepChanges(sample) {{
            const container = document.getElementById("stepChangesView");
            let html = '<h3>Step-by-Step Changes</h3>';
            
            sample.step_changes.forEach(stepChange => {{
                const stats = stepChange.diff.statistics;
                html += `
                    <div class="step-change-item">
                        <h4>${{stepChange.step_name}}</h4>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-label">Retention</div>
                                <div class="stat-value">${{(stats.retention_rate * 100).toFixed(1)}}%</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Changes</div>
                                <div class="stat-value">${{stats.insertions + stats.deletions + stats.replacements}}</div>
                            </div>
                        </div>
                    </div>
                `;
            }});
            
            container.innerHTML = html;
        }}
        
        function updateChangePlot(sample) {{
            const retentionRates = sample.step_changes.map(sc => sc.diff.statistics.retention_rate * 100);
            const changeRates = sample.step_changes.map(sc => sc.diff.statistics.change_rate * 100);
            const stepNames = sample.step_changes.map(sc => sc.step_name);
            
            const trace1 = {{
                x: stepNames,
                y: retentionRates,
                name: 'Retention Rate (%)',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: 'green'}}
            }};
            
            const trace2 = {{
                x: stepNames,
                y: changeRates,
                name: 'Change Rate (%)',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: 'orange'}}
            }};
            
            const layout = {{
                title: 'Change Statistics Over Steps',
                xaxis: {{title: 'Step'}},
                yaxis: {{title: 'Rate (%)', range: [0, 100]}},
                hovermode: 'closest'
            }};
            
            Plotly.newPlot("changePlot", [trace1, trace2], layout);
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        window.onload = initialize;
    </script>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Visualize refine diff')
    parser.add_argument('results_jsonl', type=str, help='Path to refined results JSONL file')
    parser.add_argument('history_dir', type=str, help='Path to history directory')
    parser.add_argument('--output', type=str, default='refine_diff_visualization.html',
                        help='Output HTML file path')
    parser.add_argument('--sample-ids', type=str, default=None,
                        help='Comma-separated sample IDs (0-based indices) to visualize')
    parser.add_argument('--task-ids', type=str, default=None,
                        help='Comma-separated task IDs to visualize (takes precedence over --sample-ids)')
    parser.add_argument('--evaluated-jsonl', type=str, default=None,
                        help='Path to evaluated JSONL file (optional, for showing test_passed status)')
    parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model name for tokenizer')
    
    args = parser.parse_args()
    
    # Parse sample IDs
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
    
    # Parse task IDs
    task_ids = None
    if args.task_ids:
        task_ids = [x.strip() for x in args.task_ids.split(',')]
    
    # Load data
    print("Loading results...")
    results_map = load_results(Path(args.results_jsonl))
    
    print("Loading histories...")
    histories = load_histories(Path(args.history_dir))
    
    # Load evaluated results if provided
    evaluated_map = None
    if args.evaluated_jsonl:
        evaluated_jsonl_path = Path(args.evaluated_jsonl)
        if evaluated_jsonl_path.exists():
            print(f"Loading evaluated results from {evaluated_jsonl_path}...")
            evaluated_map = load_evaluated_results(evaluated_jsonl_path)
            print(f"Loaded {len(evaluated_map)} evaluated results")
        else:
            print(f"Warning: Evaluated JSONL file not found: {evaluated_jsonl_path}")
    
    print(f"Loading tokenizer: {args.model}")
    if args.model.endswith(".pth") or "smdm" in args.model.lower() or "cdlm" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            padding_side="right", use_fast=True, trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = 32000
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Generate visualization
    generate_diff_html(
        results_map,
        histories,
        tokenizer,
        Path(args.output),
        sample_ids=sample_ids,
        task_ids=task_ids,
        evaluated_map=evaluated_map
    )


if __name__ == '__main__':
    main()

