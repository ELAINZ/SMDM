#!/usr/bin/env python3
"""
Filter passed and failed refined samples and visualize them.

This script:
1. Filters samples where test_passed=True/False from refined_evaluated.jsonl
2. Extracts their task_ids (5 passed + 5 failed, or max available)
3. Calls visualize_refine_diff.py or visualize_remask.py to visualize these samples

Usage:
    python visualize_passed_refined.py <refined_evaluated_jsonl> <history_dir> [options]
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path


def filter_task_ids_by_status(evaluated_jsonl: Path, max_count: int = 5) -> tuple[list[str], list[str]]:
    """Filter samples by test_passed status and return task_ids.
    
    Returns:
        tuple: (passed_task_ids, failed_task_ids)
    """
    passed_task_ids = []
    failed_task_ids = []
    
    with open(evaluated_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            test_passed = item.get('test_passed', False)
            task_id = item.get('task_id')
            
            if not task_id:
                continue
                
            # Check if test_passed is True (handle both bool and string)
            if test_passed is True or test_passed == 'true' or test_passed == 'True':
                if len(passed_task_ids) < max_count:
                    passed_task_ids.append(task_id)
            else:
                # Failed (False, false, or not present)
                if len(failed_task_ids) < max_count:
                    failed_task_ids.append(task_id)
    
    return passed_task_ids, failed_task_ids


def main():
    parser = argparse.ArgumentParser(
        description='Filter passed refined samples and visualize them'
    )
    parser.add_argument(
        'evaluated_jsonl',
        type=str,
        help='Path to refined_evaluated.jsonl file'
    )
    parser.add_argument(
        'history_dir',
        type=str,
        help='Path to history directory'
    )
    parser.add_argument(
        '--results-jsonl',
        type=str,
        default=None,
        help='Path to refined results JSONL file (default: infer from evaluated_jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HTML file path (default: auto-generate)'
    )
    parser.add_argument(
        '--output_remask',
        type=str,
        default=None,
        help='Output HTML file path (default: auto-generate)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='GSAI-ML/LLaDA-8B-Instruct',
        help='Model name for tokenizer'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['diff', 'remask', 'both'],
        default='diff',
        help='Visualization mode: "diff" for refine diff, "remask" for remask trajectories, "both" for both (default: diff)'
    )
    parser.add_argument(
        '--max-count',
        type=int,
        default=5,
        help='Maximum number of samples to include for each status (passed/failed) (default: 5)'
    )
    
    args = parser.parse_args()
    
    evaluated_jsonl_path = Path(args.evaluated_jsonl)
    history_dir = Path(args.history_dir)
    
    # Determine results_jsonl path
    if args.results_jsonl:
        results_jsonl_path = Path(args.results_jsonl)
    else:
        # Infer from evaluated_jsonl: remove _evaluated suffix
        stem = evaluated_jsonl_path.stem
        if stem.endswith('_evaluated'):
            stem = stem[:-10]  # Remove '_evaluated'
        results_jsonl_path = evaluated_jsonl_path.parent / f"{stem}.jsonl"
    
    # Check files exist
    if not evaluated_jsonl_path.exists():
        print(f"Error: {evaluated_jsonl_path} does not exist")
        sys.exit(1)
    
    if not results_jsonl_path.exists():
        print(f"Error: {results_jsonl_path} does not exist")
        print(f"  (Inferred from: {evaluated_jsonl_path})")
        sys.exit(1)
    
    if not history_dir.exists():
        print(f"Error: {history_dir} does not exist")
        sys.exit(1)
    
    # Filter passed and failed samples
    print(f"Filtering samples from {evaluated_jsonl_path}...")
    passed_task_ids, failed_task_ids = filter_task_ids_by_status(
        evaluated_jsonl_path, max_count=args.max_count
    )
    
    if not passed_task_ids and not failed_task_ids:
        print("No samples found!")
        sys.exit(0)
    
    # Combine passed and failed task_ids
    all_task_ids = passed_task_ids + failed_task_ids
    
    print(f"Found {len(passed_task_ids)} passed samples (showing {len(passed_task_ids)})")
    if passed_task_ids:
        print(f"  Passed: {', '.join(passed_task_ids[:5])}{'...' if len(passed_task_ids) > 5 else ''}")
    print(f"Found {len(failed_task_ids)} failed samples (showing {len(failed_task_ids)})")
    if failed_task_ids:
        print(f"  Failed: {', '.join(failed_task_ids[:5])}{'...' if len(failed_task_ids) > 5 else ''}")
    
    task_ids_str = ','.join(all_task_ids)
    output_dir = evaluated_jsonl_path.parent
    
    # Determine which visualizations to generate
    generate_diff = args.mode in ['diff', 'both']
    generate_remask = args.mode in ['remask', 'both']
    
    exit_code = 0
    
    # Generate refine diff visualization
    if generate_diff:
        # Determine output path
        if args.output:
            # If only diff mode and output specified, use it
            output_path = Path(args.output)
        else:
            # Auto-generate output path (include both passed and failed)
            output_path = output_dir / "refine_diff_passed.html"
        
        # Build command for visualize_refine_diff.py
        # Pass evaluated_jsonl to show refine success/failure status
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_refine_diff.py"),
            str(results_jsonl_path),
            str(history_dir),
            '--output', str(output_path),
            '--task-ids', task_ids_str,
            '--model', args.model,
            '--evaluated-jsonl', str(evaluated_jsonl_path),
        ]
        
        print(f"\n{'='*60}")
        print("Running refine diff visualization...")
        print(f"  Results: {results_jsonl_path}")
        print(f"  History: {history_dir}")
        print(f"  Output: {output_path}")
        print(f"  Evaluated: {evaluated_jsonl_path}")
        print(f"  Task IDs: {len(all_task_ids)} samples ({len(passed_task_ids)} passed, {len(failed_task_ids)} failed)\n")
        
        # Run visualization
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✅ Refine diff visualization saved to: {output_path}")
        else:
            print(f"❌ Refine diff visualization failed with exit code {result.returncode}")
            exit_code = result.returncode
    
    # Generate remask visualization
    if generate_remask:
        # Determine output path
        if args.output:
            # If only remask mode and output specified, use it
            output_path = Path(args.output_remask)
        else:
            # Auto-generate output path (include both passed and failed)
            output_path = output_dir / "remask_passed.html"
        
        # Build command for visualize_remask.py (pass evaluated jsonl so HTML shows Passed/Failed)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_remask.py"),
            str(history_dir),
            '--output', str(output_path),
            '--task-ids', task_ids_str,
            '--model', args.model,
            '--evaluated-jsonl', str(evaluated_jsonl_path),
        ]
        
        print(f"\n{'='*60}")
        print("Running remask visualization...")
        print(f"  History: {history_dir}")
        print(f"  Output: {output_path}")
        print(f"  Task IDs: {len(all_task_ids)} samples ({len(passed_task_ids)} passed, {len(failed_task_ids)} failed)\n")
        
        # Run visualization
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✅ Remask visualization saved to: {output_path}")
        else:
            print(f"❌ Remask visualization failed with exit code {result.returncode}")
            if exit_code == 0:
                exit_code = result.returncode
    
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()

