#!/usr/bin/env python3
"""
Interactive HTML Visualization for Remask Trajectories

Usage:
    python visualize_remask.py <history_dir> [--output output.html] [--sample-ids 0,1,2]

Example:
    python visualize_remask.py evals_results/histories/humaneval/fredzzp_open-dcoder-0.5B/random_remask/linear
"""

import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer
import numpy as np


def load_histories(history_dir: Path) -> List[Dict[str, Any]]:
    """Load and merge histories from all process files."""
    pt_files = sorted(history_dir.glob("*_process*.pt"))
    if not pt_files:
        # Try single-process file
        pt_files = sorted(history_dir.glob("*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {history_dir}")
    
    print(f"Found {len(pt_files)} history files")
    
    # Load and merge all histories
    all_histories = []
    for pt_file in pt_files:
        print(f"Loading {pt_file.name}...")
        histories = torch.load(pt_file, map_location='cpu', weights_only=False)
        all_histories.extend(histories)
    
    print(f"Total {len(all_histories)} samples loaded")
    return all_histories


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs to strings."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [tokenizer.decode([tid]) for tid in token_ids]


def create_step_visualization(sample_history: Dict, step_idx: int, tokenizer) -> Dict:
    """Create visualization data for a single step."""
    prompt = sample_history['prompt']
    fix_mask = prompt['fix_mask']
    initial_tokens = prompt['tokens']
    
    # Get step data
    if step_idx == -1:  # Initial state
        tokens = initial_tokens
        conf = torch.ones_like(tokens, dtype=torch.float32)
        remask_pos = torch.zeros_like(tokens, dtype=torch.bool)
    else:
        step = sample_history['steps'][step_idx]
        
        # Reconstruct full sequence
        tokens = initial_tokens.clone()
        tokens[~fix_mask] = step['x0_variable']
        
        # Reconstruct confidence (fixed positions have inf confidence)
        conf = torch.full_like(tokens, float('inf'), dtype=torch.float32)
        conf[~fix_mask] = step['conf_variable']
        
        remask_pos = step['remask_positions']
    
    # Decode tokens
    token_strs = decode_tokens(tokenizer, tokens)
    
    return {
        'tokens': tokens.tolist(),
        'token_strs': token_strs,
        'confidence': conf.tolist(),
        'remask_positions': remask_pos.tolist(),
        'fix_mask': fix_mask.tolist(),
    }


def create_interactive_html(
    histories: List[Dict],
    tokenizer,
    output_path: Path,
    sample_ids: List[int] = None,
    task_ids: List[str] = None,
    history_dir: Path = None,
    task_id_to_passed: Dict[str, bool] = None,
):
    """Create interactive HTML visualization for all samples.
    
    Args:
        histories: List of history dicts
        tokenizer: Tokenizer for decoding
        output_path: Output HTML file path
        sample_ids: Optional list of sample indices (0-based)
        task_ids: Optional list of task_ids to visualize (takes precedence over sample_ids)
        history_dir: Optional history directory for metadata
        task_id_to_passed: Optional dict task_id -> test_passed (from refined evaluated jsonl)
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
            else:
                print(f"Warning: Task {task_id} not found in histories")
        selected_indices.sort()
        sample_ids = selected_indices
    elif sample_ids is None:
        sample_ids = list(range(len(histories)))
    else:
        sample_ids = [i for i in sample_ids if i < len(histories)]
    
    print(f"\nGenerating visualization for {len(sample_ids)} samples...")
    
    # Extract metadata from path
    metadata_note = ""
    if history_dir:
        parts = history_dir.parts
        if len(parts) >= 5:
            task = parts[-5]
            model = parts[-4]
            algorithm = parts[-3]
            scheduler = parts[-2]
            steps = parts[-1]
            metadata_note = f"Task: {task} | Model: {model} | Algorithm: {algorithm} | Scheduler: {scheduler} | Steps: {steps}"
        metadata_note += f"<br>Source: {history_dir}"
    
    # Create HTML structure
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <title>Remask Trajectory Visualization</title>',
        '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        '    <style>',
        '        body { font-family: "Courier New", monospace; margin: 20px; background: #f5f5f5; }',
        '        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }',
        '        .controls { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }',
        '        .token-display { margin: 20px 0; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 5px; }',
        '        .token { display: inline-block; margin: 2px; padding: 5px 8px; border-radius: 3px; font-size: 14px; cursor: pointer; position: relative; }',
        '        .token-fixed { background: #e3f2fd; border: 1px solid #90caf9; }',
        '        .token-variable { background: #fff3e0; border: 1px solid #ffb74d; }',
        '        .token-remask { background: #ffcdd2; border: 2px solid #ef5350; font-weight: bold; }',
        '        .token-error { background: #fff9c4; border: 2px solid #ff6f00; box-shadow: 0 0 5px rgba(255, 152, 0, 0.5); }',
        '        .token-corrected { background: #e8f5e9; border: 2px solid #4CAF50; box-shadow: 0 0 5px rgba(76, 175, 80, 0.4); }',
        '        .token-tooltip { position: absolute; display: none; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 5px; z-index: 1000; min-width: 300px; font-size: 12px; line-height: 1.6; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }',
        '        .token-tooltip.show { display: block; }',
        '        .tooltip-section { margin: 5px 0; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.2); }',
        '        .tooltip-section:last-child { border-bottom: none; }',
        '        .tooltip-label { color: #aaa; font-size: 11px; }',
        '        .tooltip-value { color: #fff; font-weight: bold; }',
        '        .tooltip-arrow { color: #4CAF50; }',
        '        .info-box { margin: 10px 0; padding: 10px; background: #f1f3f4; border-radius: 3px; }',
        '        select, input { margin: 5px; padding: 5px 10px; border-radius: 3px; border: 1px solid #ddd; }',
        '        button { padding: 8px 15px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }',
        '        button:hover { background: #45a049; }',
        '        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0; }',
        '        .stat-card { padding: 10px; background: white; border: 1px solid #ddd; border-radius: 5px; }',
        '        .stat-label { font-size: 12px; color: #666; }',
        '        .stat-value { font-size: 18px; font-weight: bold; color: #333; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        '        <h1>🎯 Remask Trajectory Visualization</h1>',
        f'        <div class="info-box" style="background: #e8f5e9; border-left: 4px solid #4CAF50;">{metadata_note}</div>' if metadata_note else '',
        '        ',
        '        <div class="controls">',
        '            <label>Sample: <select id="sampleSelect" onchange="updateVisualization()"></select></label>',
        '            <label>Step: <input type="range" id="stepSlider" min="-1" value="-1" oninput="updateVisualization()" style="width: 300px;"></label>',
        '            <span id="stepLabel">Step: Initial</span>',
        '            <button onclick="playAnimation()">▶ Play</button>',
        '            <button onclick="stopAnimation()">⏸ Pause</button>',
        '        </div>',
        '        <div id="sampleStatusBadge" class="info-box" style="margin-bottom: 10px; display: none;"></div>',
        '        ',
        '        <div class="stats" id="statsContainer"></div>',
        '        ',
        '        <div class="token-display" id="tokenDisplay"></div>',
        '        ',
        '        <div id="confPlot" style="width: 100%; height: 400px;"></div>',
        '        ',
        '        <div id="errorStatsPlot" style="width: 100%; height: 400px;"></div>',
        '        <div id="confARRefPlot" style="width: 100%; height: 400px; display: none;"></div>',
        '    </div>',
        '',
        '    <script>',
        '        // Data embedded in JavaScript',
        '        const data = ' + generate_js_data(histories, tokenizer, sample_ids) + ';',
        '        const taskIdToPassed = ' + json.dumps(task_id_to_passed or {}) + ';',
        '        ',
        '        let animationInterval = null;',
        '        ',
        '        function initializeSampleSelect() {',
        '            const select = document.getElementById("sampleSelect");',
        '            data.samples.forEach((sample, idx) => {',
        '                const option = document.createElement("option");',
        '                option.value = idx;',
        '                const taskInfo = sample.task_id ? `Task: ${sample.task_id}, ` : "";',
        '                const passedStr = (taskIdToPassed && sample.task_id && taskIdToPassed.hasOwnProperty(sample.task_id)) ? (taskIdToPassed[sample.task_id] ? " ✅ Passed" : " ❌ Failed") : "";',
        '                option.text = `Sample #${sample.id} (${taskInfo}Steps: ${sample.num_steps})${passedStr}`;',
        '                select.appendChild(option);',
        '            });',
        '            ',
        '            const slider = document.getElementById("stepSlider");',
        '            slider.max = data.samples[0].num_steps - 1;',
        '        }',
        '        ',
        '        function updateVisualization() {',
        '            const sampleIdx = parseInt(document.getElementById("sampleSelect").value);',
        '            const stepIdx = parseInt(document.getElementById("stepSlider").value);',
        '            const sample = data.samples[sampleIdx];',
        '            ',
        '            // Update sample status badge (Corrected & Passed / Still Failed)',
        '            const badgeEl = document.getElementById("sampleStatusBadge");',
        '            if (taskIdToPassed && Object.keys(taskIdToPassed).length > 0 && sample.task_id && taskIdToPassed.hasOwnProperty(sample.task_id)) {',
        '                badgeEl.style.display = "block";',
        '                const passed = taskIdToPassed[sample.task_id];',
        '                if (passed) {',
        '                    badgeEl.style.background = "#e8f5e9"; badgeEl.style.borderLeft = "4px solid #4CAF50";',
        '                    badgeEl.innerHTML = "✅ <strong>This sample: Corrected & Passed</strong> (refined answer matches ground truth)";',
        '                } else {',
        '                    badgeEl.style.background = "#ffebee"; badgeEl.style.borderLeft = "4px solid #f44336";',
        '                    badgeEl.innerHTML = "❌ <strong>This sample: Still Failed</strong> (refined answer does not match ground truth)";',
        '                }',
        '            } else {',
        '                badgeEl.style.display = "none";',
        '            }',
        '            ',
        '            // Update slider range',
        '            const slider = document.getElementById("stepSlider");',
        '            slider.max = sample.num_steps - 1;',
        '            ',
        '            // Update step label',
        '            const stepLabel = stepIdx === -1 ? "Initial" : `${stepIdx + 1}/${sample.num_steps}`;',
        '            document.getElementById("stepLabel").textContent = `Step: ${stepLabel}`;',
        '            ',
        '            // Get step data',
        '            const stepData = stepIdx === -1 ? sample.initial : sample.steps[stepIdx];',
        '            ',
        '            // Update token display (pass stepIdx and sample for corrected-token and error highlighting)',
        '            updateTokenDisplay(stepData, stepIdx, sample);',
        '            ',
        '            // Update statistics',
        '            updateStats(stepData, stepIdx);',
        '            ',
        '            // Update confidence plot',
        '            updateConfidencePlot(stepData);',
        '            ',
        '            // Update error token plot',
        '            updateErrorStatsPlot(sample);',
        '            // Update AR ref confidence plot if available',
        '            if (stepData.confidence_AR_ref) {',
        '                updateARRefConfidencePlot(stepData);',
        '            }',
        '        }',
        '        ',
        'function updateErrorStatsPlot(sample) {',
        '    const steps = sample.steps;',
        '    const stepIndices = steps.map((_, i) => i + 1);',
        '',
        '    const avgConfErrors = steps.map(step => step.avg_error_conf ?? null);',
        '',
        # '    const avgProbErrors = steps.map(step => step.avg_prob_error ?? null);',
        # '',
        '    const traces = [',
        '        {',
        '            x: stepIndices,',
        '            y: avgConfErrors,',
        '            mode: \'lines+markers\',',
        '            name: \'Avg Confidence\'',
        '        },',
        '    ];',
        '',
        '    const layout = {',
        '        title: \'Avg Confidence over Steps\',',
        '        xaxis: { title: \'Step\' },',
        '        yaxis: { title: \'Average Value\', range: [0, 1] },',
        '        hovermode: \'closest\'',
        '    };',
        '',
        '    Plotly.newPlot(\'errorStatsPlot\', traces, layout);',
        '};',
        '        function updateTokenDisplay(stepData, stepIdx, sample) {',
        '            const container = document.getElementById("tokenDisplay");',
        '            let html = "<div>";',
        '            ',
        '            // Calculate offset for error positions (they are relative to variable part)',
        '            const numFixed = stepData.fix_mask.filter(x => x).length;',
        '            const isLastStep = (stepIdx >= 0 && sample.steps && stepIdx === sample.num_steps - 1);',
        '            ',
        '            stepData.tokens.forEach((token, idx) => {',
        '                const isFixed = stepData.fix_mask[idx];',
        '                const isRemask = stepData.remask_positions[idx];',
        '                const conf = stepData.confidence[idx];',
        '                // Error positions are relative to the variable (body) part',
        '                const bodyIdx = idx - numFixed;  // Index within the body',
        '                const contentLen = (sample.variable_content_length != null && sample.variable_content_length !== undefined) ? sample.variable_content_length : Infinity;',
        '                const isError = !isFixed && bodyIdx < contentLen && sample.error_positions && sample.error_positions.includes(bodyIdx);',
        '                const originalToken = isError && sample.error_original_tokens ? sample.error_original_tokens[sample.error_positions.indexOf(bodyIdx)] : null;',
        '                const initialToken = (sample.initial && sample.initial.tokens && idx < sample.initial.tokens.length) ? sample.initial.tokens[idx] : null;',
        '                const isCorrected = isLastStep && isError && initialToken !== null && token !== initialToken;',
        '                ',
        '                let className = isFixed ? "token-fixed" : "token-variable";',
        '                if (isRemask) className = "token-remask";',
        '                if (isError && !isFixed) { if (isCorrected) className += " token-corrected"; else className += " token-error"; }',
        '                ',
        '                // Build detailed tooltip for variable tokens',
                '                let tooltipHTML = "";',
                '                if (!isFixed) {',
        '                    const confStr = conf === Infinity ? "∞" : conf.toFixed(4);',
        '                    const hasARRef = stepData.confidence_AR_ref !== undefined;',
        '                    const confARRefStr = hasARRef ? (stepData.confidence_AR_ref[idx] === Infinity ? "∞" : stepData.confidence_AR_ref[idx].toFixed(4)) : null;',
        '                    ',
        '                    tooltipHTML = `<div class="token-tooltip" id="tooltip-${idx}">',
        '                        <div class="tooltip-section">',
        '                            <div class="tooltip-label">📍 Position:</div>',
        '                            <div class="tooltip-value">${idx}</div>',
        '                        </div>`;',
        '                    ',
        '                    // At last step show corrected as primary; otherwise show error (buggy)',
        '                    if (isCorrected && initialToken !== null) {',
        '                        tooltipHTML += `<div class="tooltip-section" style="background: rgba(76, 175, 80, 0.2); padding: 8px; border-radius: 3px;">',
        '                            <div class="tooltip-label" style="color: #2e7d32;">✅ CORRECTED:</div>',
        '                            <div class="tooltip-value">"${initialToken}" → "${token}"</div>',
        '                        </div>`;',
        '                    } else if (isError && (originalToken !== undefined && originalToken !== null)) {',
        '                        tooltipHTML += `<div class="tooltip-section" style="background: rgba(255, 152, 0, 0.2); padding: 8px; border-radius: 3px;">',
        '                            <div class="tooltip-label" style="color: #ff6f00;">⚠️ ERROR TOKEN (buggy):</div>',
        '                            <div class="tooltip-value" style="color: #ff9800;">Current: "${token}"</div>',
        '                            <div class="tooltip-value" style="color: #4CAF50;">Expected (correct): "${originalToken}"</div>',
        '                        </div>`;',
        '                    }',
        '                    ',
        '                    tooltipHTML += `<div class="tooltip-section">',
        '                            <div class="tooltip-label">🎯 Current Step ${stepIdx + 1}:</div>',
        '                            <div class="tooltip-value">Token: "${token}" (ID: ${stepData.token_ids[idx]})</div>',
        '                            <div class="tooltip-value">Confidence: ${confStr}</div>`;',
        '                    if (hasARRef) {',
        '                        tooltipHTML += `<div class="tooltip-value">AR Ref Conf: ${confARRefStr}</div>`;',
        '                    }',
        '                    tooltipHTML += `<div class="tooltip-value">${isRemask ? "🔴 Will be remasked" : "✅ Kept"}</div>',
        '                        </div>`;',
        '                    ',
        '                    // Add next step info if available',
        '                    if (stepIdx < sample.num_steps - 1) {',
        '                        const nextStep = sample.steps[stepIdx + 1];',
        '                        const nextToken = nextStep.tokens[idx];',
        '                        const nextTokenId = nextStep.token_ids[idx];',
        '                        const nextConf = nextStep.confidence[idx];',
        '                        const nextConfStr = nextConf === Infinity ? "∞" : nextConf.toFixed(4);',
        '                        const nextHasARRef = nextStep.confidence_AR_ref !== undefined;',
        '                        const nextConfARRefStr = nextHasARRef ? (nextStep.confidence_AR_ref[idx] === Infinity ? "∞" : nextStep.confidence_AR_ref[idx].toFixed(4)) : null;',
        '                        const changed = nextToken !== token;',
        '                        ',
        '                        tooltipHTML += `<div class="tooltip-section">',
        '                            <div class="tooltip-label">➡️ Next Step ${stepIdx + 2}:</div>',
        '                            <div class="tooltip-value">Token: "${nextToken}" (ID: ${nextTokenId})</div>',
        '                            <div class="tooltip-value">Confidence: ${nextConfStr}</div>`;',
        '                        if (nextHasARRef) {',
        '                            tooltipHTML += `<div class="tooltip-value">AR Ref Conf: ${nextConfARRefStr}</div>`;',
        '                        }',
        '                        if (changed) {',
        '                            tooltipHTML += `<div class="tooltip-value" style="color: #ff9800;">🔄 Changed from "${token}"</div>`;',
        '                        } else {',
        '                            tooltipHTML += `<div class="tooltip-value" style="color: #4CAF50;">✓ Unchanged</div>`;',
        '                        }',
        '                        tooltipHTML += `</div>`;',
        '                    }',
        '                    ',
        '                    tooltipHTML += `</div>`;',
        '                } else {',
        '                    tooltipHTML = `<div class="token-tooltip" id="tooltip-${idx}">',
        '                        <div class="tooltip-section">',
        '                            <div class="tooltip-label">📍 Position:</div>',
        '                            <div class="tooltip-value">${idx}</div>',
        '                        </div>',
        '                        <div class="tooltip-section">',
        '                            <div class="tooltip-value">🔒 Fixed (Prompt)</div>',
        '                            <div class="tooltip-value">Token: "${token}" (ID: ${stepData.token_ids[idx]})</div>',
        '                        </div>',
        '                    </div>`;',
        '                }',
        '                ',
        '                html += `<span class="token ${className}" ',
        '                    onmouseenter="showTooltip(${idx}, event)" ',
        '                    onmouseleave="hideTooltip(${idx})">${token}${tooltipHTML}</span>`;',
        '            });',
        '            ',
        '            html += "</div>";',
        '            ',
        '            // Add legend',
        '            html += `<div class="info-box" style="margin-top: 10px;">',
        '                <span class="token token-fixed">🔒 Fixed (Prompt)</span>',
        '                <span class="token token-variable">📝 Variable (Generated)</span>',
        '                <span class="token token-remask">🔴 Remasked</span>',
        '                <span class="token token-error">⚠️ Error Token (buggy)</span>',
        '                <span class="token token-corrected">✅ Corrected</span>',
        '                <span style="margin-left: 20px; color: #666;">💡 Hover over tokens for details</span>',
        '            </div>`;',
        '            ',
        '            container.innerHTML = html;',
        '        }',
        '        ',
        '        function updateStats(stepData, stepIdx) {',
        '            const numTokens = stepData.tokens.length;',
        '            const numFixed = stepData.fix_mask.filter(x => x).length;',
        '            const numVariable = numTokens - numFixed;',
        '            const numRemasked = stepData.remask_positions.filter(x => x).length;',
        '            const remaskRatio = numVariable > 0 ? (numRemasked / numVariable * 100).toFixed(1) : 0;',
        '            ',
        '            // Calculate average confidence (excluding inf)',
        '            const validConf = stepData.confidence.filter(c => c !== Infinity);',
        '            const avgConf = validConf.length > 0 ? (validConf.reduce((a, b) => a + b, 0) / validConf.length).toFixed(3) : "N/A";',
        '            const minConf = validConf.length > 0 ? Math.min(...validConf).toFixed(3) : "N/A";',
        '            const avgConfError = stepData.avg_error_conf;',
        # '            const avgProbError = stepData.avg_error_prob;',
        '            // Calculate AR reference confidence stats if available',
        '            const hasARRef = stepData.confidence_AR_ref !== undefined;',
        '            let avgConfAR = "N/A";',
        '            let minConfAR = "N/A";',
        '            if (hasARRef) {',
        '                const validConfAR = stepData.confidence_AR_ref.filter(c => c !== Infinity);',
        '                avgConfAR = validConfAR.length > 0 ? (validConfAR.reduce((a, b) => a + b, 0) / validConfAR.length).toFixed(3) : "N/A";',
        '                minConfAR = validConfAR.length > 0 ? Math.min(...validConfAR).toFixed(3) : "N/A";',
        '            }',
        '            ',
        '            let html = `',
        '                <div class="stat-card">',
        '                    <div class="stat-label">Total Tokens</div>',
        '                    <div class="stat-value">${numTokens}</div>',
        '                </div>',
        '                <div class="stat-card">',
        '                    <div class="stat-label">Fixed / Variable</div>',
        '                    <div class="stat-value">${numFixed} / ${numVariable}</div>',
        '                </div>',
        '                <div class="stat-card">',
        '                    <div class="stat-label">Remasked Tokens</div>',
        '                    <div class="stat-value">${numRemasked} (${remaskRatio}%)</div>',
        '                </div>',
        '                <div class="stat-card">',
        '                    <div class="stat-label">Avg Confidence</div>',
        '                    <div class="stat-value">${avgConf}</div>',
        '                </div>',
        '                <div class="stat-card">',
        '                    <div class="stat-label">Min Confidence</div>',
        '                    <div class="stat-value">${minConf}</div>',
        '                </div>',
        '                <div class="stat-card" style="background: #fce4ec;">',
        '                    <div class="stat-label">Avg Error Conf</div>',
        '                    <div class="stat-value">${avgConfError}</div>',
        '                </div>`;',
        # '                <div class="stat-card" style="background: #fce4ec;">',
        # '                    <div class="stat-label">Avg Error Prob</div>',
        # '                    <div class="stat-value">${avgProbError}</div>',
        # '                </div>`;',
        '            ',
        '            // Add AR reference confidence stats if available',
        '            if (hasARRef) {',
        '                html += `',
        '                <div class="stat-card" style="background: #e8f5e9;">',
        '                    <div class="stat-label">Avg AR Ref Conf</div>',
        '                    <div class="stat-value">${avgConfAR}</div>',
        '                </div>',
        '                <div class="stat-card" style="background: #e8f5e9;">',
        '                    <div class="stat-label">Min AR Ref Conf</div>',
        '                    <div class="stat-value">${minConfAR}</div>',
        '                </div>`;',
        '            }',
        '            ',
        '            document.getElementById("statsContainer").innerHTML = html;',
        '        }',
        '        ',
        '        function updateConfidencePlot(stepData) {',
        '            const positions = stepData.tokens.map((_, idx) => idx);',
        '            const conf = stepData.confidence.map(c => c === Infinity ? null : c);',
        '            const colors = stepData.remask_positions.map(r => r ? "red" : (stepData.fix_mask[positions.indexOf(positions.find((_, i) => i === positions.indexOf(_)))] ? "blue" : "orange"));',
        '            ',
        '            const trace = {',
        '                x: positions,',
        '                y: conf,',
        '                mode: "markers+lines",',
        '                marker: {',
        '                    size: 8,',
        '                    color: stepData.remask_positions.map((r, i) => r ? 1 : 0),',
        '                    colorscale: [[0, "lightblue"], [1, "red"]],',
        '                    showscale: false',
        '                },',
        '                line: { color: "lightgray", width: 1 },',
        '                text: stepData.tokens,',
        '                hovertemplate: "<b>%{text}</b><br>Position: %{x}<br>Confidence: %{y:.3f}<extra></extra>"',
        '            };',
        '            ',
        '            const layout = {',
        '                title: "Token Confidence Distribution",',
        '                xaxis: { title: "Token Position" },',
        '                yaxis: { title: "Confidence", range: [0, 1] },',
        '                hovermode: "closest"',
        '            };',
        '            ',
        '            Plotly.newPlot("confPlot", [trace], layout);',
        '        }',
        '        ',
        '        function updateARRefConfidencePlot(stepData) {',
        '            const arRefPlot = document.getElementById("confARRefPlot");',
        '            if (!stepData.confidence_AR_ref) {',
        '                arRefPlot.style.display = "none";',
        '                return;',
        '            }',
        '            ',
        '            arRefPlot.style.display = "block";',
        '            ',
        '            const positions = stepData.tokens.map((_, idx) => idx);',
        '            const confARRef = stepData.confidence_AR_ref.map(c => c === Infinity ? null : c);',
        '            ',
        '            const trace = {',
        '                x: positions,',
        '                y: confARRef,',
        '                mode: "markers+lines",',
        '                marker: {',
        '                    size: 8,',
        '                    color: stepData.remask_positions.map((r, i) => r ? 1 : 0),',
        '                    colorscale: [[0, "lightgreen"], [1, "red"]],',
        '                    showscale: false',
        '                },',
        '                line: { color: "lightgray", width: 1 },',
        '                text: stepData.tokens,',
        '                hovertemplate: "<b>%{text}</b><br>Position: %{x}<br>AR Ref Conf: %{y:.3f}<extra></extra>"',
        '            };',
        '            ',
        '            const layout = {',
        '                title: "AR Reference Confidence Distribution",',
        '                xaxis: { title: "Token Position" },',
        '                yaxis: { title: "AR Reference Confidence", range: [0, 1] },',
        '                hovermode: "closest"',
        '            };',
        '            ',
        '            Plotly.newPlot("confARRefPlot", [trace], layout);',
        '        }',
        '        ',
        '        function playAnimation() {',
        '            if (animationInterval) return;',
        '            ',
        '            const slider = document.getElementById("stepSlider");',
        '            animationInterval = setInterval(() => {',
        '                let currentStep = parseInt(slider.value);',
        '                currentStep++;',
        '                if (currentStep > parseInt(slider.max)) {',
        '                    currentStep = -1;',
        '                }',
        '                slider.value = currentStep;',
        '                updateVisualization();',
        '            }, 1000);',
        '        }',
        '        ',
        '        function stopAnimation() {',
        '            if (animationInterval) {',
        '                clearInterval(animationInterval);',
        '                animationInterval = null;',
        '            }',
        '        }',
        '        ',
        '        function showTooltip(idx, event) {',
        '            const tooltip = document.getElementById("tooltip-" + idx);',
        '            if (tooltip) {',
        '                tooltip.classList.add("show");',
        '                ',
        '                // Position tooltip near the mouse',
        '                const rect = event.target.getBoundingClientRect();',
        '                tooltip.style.left = "0px";',
        '                tooltip.style.top = (rect.height + 5) + "px";',
        '            }',
        '        }',
        '        ',
        '        function hideTooltip(idx) {',
        '            const tooltip = document.getElementById("tooltip-" + idx);',
        '            if (tooltip) {',
        '                tooltip.classList.remove("show");',
        '            }',
        '        }',
        '        ',
        '        // Initialize',
        '        window.onload = function() {',
        '            initializeSampleSelect();',
        '            updateVisualization();',
        '            ',
        '            // Check if AR ref conf exists and show/hide plot accordingly',
        '            const firstSample = data.samples[0];',
        '            const hasARRef = firstSample.steps.length > 0 && firstSample.steps[0].confidence_AR_ref !== undefined;',
        '            if (!hasARRef) {',
        '                document.getElementById("confARRefPlot").style.display = "none";',
        '            }',
        '        };',
        '    </script>',
        '</body>',
        '</html>',
    ]
    
    html_content = '\n'.join(html_parts)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"📊 Open in browser to explore {len(sample_ids)} samples")


def generate_js_data(histories: List[Dict], tokenizer, sample_ids: List[int]) -> str:
    """Generate JavaScript data structure."""
    import json
    
    samples_data = []
    
    for sample_id in sample_ids:
        sample = histories[sample_id]
        
        # Process initial state
        initial_tokens = sample['prompt']['tokens']
        fix_mask = sample['prompt']['fix_mask']
        initial_data = {
            'tokens': decode_tokens(tokenizer, initial_tokens),
            'token_ids': initial_tokens.tolist(),
            'confidence': [float('inf')] * len(initial_tokens),
            'remask_positions': [False] * len(initial_tokens),
            'fix_mask': fix_mask.tolist(),
        }
        
        # Process each step
        steps_data = []
        for step in sample['steps']:
            # Reconstruct full sequence
            tokens = initial_tokens.clone()
            tokens[~fix_mask] = step['x0_variable']
            
            conf = torch.full_like(tokens, float('inf'), dtype=torch.float32)
            # Convert to float32 to match conf dtype (in case conf_variable is bfloat16)
            conf[~fix_mask] = step['conf_variable'].float()
            
            step_data = {
                'tokens': decode_tokens(tokenizer, tokens),
                'token_ids': tokens.tolist(),
                'confidence': [float(c) if c != float('inf') else float('inf') for c in conf.tolist()],
                'remask_positions': step['remask_positions'].tolist(),
                'fix_mask': fix_mask.tolist(),
                'avg_error_conf': step['avg_error_conf'],
                'avg_error_prob': step['avg_error_prob'],
            }

            # Add AR reference confidence if available
            if 'conf_AR_ref_variable' in step:
                conf_AR_ref = torch.full_like(tokens, float('inf'), dtype=torch.float32)
                # Convert to float32 to match conf_AR_ref dtype (in case conf_AR_ref_variable is bfloat16)
                conf_AR_ref[~fix_mask] = step['conf_AR_ref_variable'].float()
                step_data['confidence_AR_ref'] = [float(c) if c != float('inf') else float('inf') for c in conf_AR_ref.tolist()]
            
            steps_data.append(step_data)
        
        # Get task_id and error information from history if available
        task_id = sample.get('task_id', None)
        raw_error_positions = sample.get('error_positions', [])
        raw_error_original_tokens = sample.get('error_original_tokens', [])
        variable_content_length = sample.get('variable_content_length', None)
        # Ensure list of int; clamp to variable length so pad positions are never marked error
        fix_list = fix_mask.tolist() if hasattr(fix_mask, 'tolist') else fix_mask
        num_variable = len(fix_list) - sum(1 for x in fix_list if x)
        content_len = int(variable_content_length) if variable_content_length is not None else num_variable
        raw_orig = list(raw_error_original_tokens)
        kept = []
        for pi, p in enumerate(raw_error_positions):
            p = int(p) if p is not None else -1
            if 0 <= p < num_variable and p < content_len:
                orig = raw_orig[pi] if pi < len(raw_orig) else None
                kept.append((p, orig))
        error_positions = [x[0] for x in kept]
        error_original_tokens = [x[1] for x in kept]
        samples_data.append({
            'id': sample_id,
            'task_id': task_id,
            'num_steps': len(steps_data),
            'initial': initial_data,
            'steps': steps_data,
            'error_positions': error_positions,
            'error_original_tokens': error_original_tokens,
            'variable_content_length': variable_content_length,
        })
    
    return json.dumps({'samples': samples_data}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize remask trajectories')
    parser.add_argument('history_dir', type=str, help='Path to history directory')
    parser.add_argument('--output', type=str, default='remask_visualization.html',
                        help='Output HTML file path')
    parser.add_argument('--sample-ids', type=str, default=None,
                        help='Comma-separated sample IDs (0-based indices) to visualize (default: all)')
    parser.add_argument('--task-ids', type=str, default=None,
                        help='Comma-separated task IDs to visualize (takes precedence over --sample-ids)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Coder-0.5B-Instruct',
                        help='Model name for tokenizer')
    parser.add_argument('--skip_if_exist', action='store_true',
                        help='Skip visualization if output file already exists')
    parser.add_argument('--evaluated-jsonl', type=str, default=None,
                        help='Path to refined evaluated JSONL (to show Passed/Failed per sample)')
    
    args = parser.parse_args()
    
    # Check if output file exists and skip if requested
    output_path = Path(args.output)
    if args.skip_if_exist and output_path.exists():
        print(f"Output file {output_path} already exists, skipping visualization (--skip_if_exist enabled)")
        return
    
    # Parse sample IDs
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
    
    # Parse task IDs
    task_ids = None
    if args.task_ids:
        task_ids = [x.strip() for x in args.task_ids.split(',')]
    
    # Load task_id -> test_passed from evaluated jsonl if provided
    task_id_to_passed = {}
    if args.evaluated_jsonl:
        eval_path = Path(args.evaluated_jsonl)
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    tid = item.get('task_id')
                    if tid is not None:
                        task_id_to_passed[tid] = item.get('test_passed', False)
            print(f"Loaded test_passed for {len(task_id_to_passed)} tasks from {eval_path}")
        else:
            print(f"Warning: evaluated jsonl not found: {eval_path}")
    
    # Load histories
    history_dir = Path(args.history_dir)
    histories = load_histories(history_dir)
    
    # Load tokenizer (use TinyLlama for .pth/smdm/cdlm diffusion models)
    print(f"\nLoading tokenizer: {args.model}")
    if args.model.endswith(".pth") or "smdm" in args.model.lower() or "cdlm" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            padding_side="right", use_fast=True, trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = 32000
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create visualization
    create_interactive_html(
        histories, tokenizer, output_path,
        sample_ids=sample_ids, task_ids=task_ids, history_dir=history_dir,
        task_id_to_passed=task_id_to_passed if task_id_to_passed else None,
    )


if __name__ == '__main__':
    main()