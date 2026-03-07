#!/usr/bin/env python3
"""
Analyze Top-K hit rate for error token prediction.
Calculates whether the lowest-confidence K tokens contain error tokens.
"""

import torch
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def load_histories(history_dir: Path) -> List[Dict[str, Any]]:
    """Load all history files from a directory."""
    pt_files = sorted(history_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {history_dir}")
    
    all_histories = []
    for pt_file in pt_files:
        print(f"  Loading {pt_file.name}...")
        histories = torch.load(pt_file, map_location='cpu', weights_only=False)
        all_histories.extend(histories)
    
    print(f"  Total {len(all_histories)} samples loaded")
    return all_histories


def calculate_topk_hit_rate(histories: List[Dict], max_k: int = 10) -> Dict:
    """
    Calculate Top-K hit rate, false positive rate, and precision for K=1 to max_k.
    
    Returns:
        dict with results for each K
    """
    results = []
    total_samples = len(histories)
    
    for k in range(1, max_k + 1):
        hit_count = 0
        total_false_positives = 0
        total_true_positives = 0
        total_topk_tokens = 0
        
        for sample in histories:
            # Get step 1 confidence (variable tokens only)
            if len(sample['steps']) == 0:
                continue
            
            step1 = sample['steps'][0]
            conf_variable = step1['conf_variable'].cpu().numpy()
            
            # Get error positions (relative to body/variable part)
            error_positions = sample.get('error_positions', [])
            if isinstance(error_positions, str):
                error_positions = json.loads(error_positions) if error_positions else []
            error_positions_set = set(error_positions)
            
            if len(error_positions) == 0:
                continue
            
            # Get indices of K lowest confidence tokens
            if len(conf_variable) < k:
                topk_indices = np.arange(len(conf_variable))
            else:
                topk_indices = np.argsort(conf_variable)[:k]
            
            # Check if any error token is in top-K
            hit = any(pos in topk_indices for pos in error_positions)
            if hit:
                hit_count += 1
            
            # Calculate false positives and true positives for this sample
            for idx in topk_indices:
                if idx in error_positions_set:
                    total_true_positives += 1
                else:
                    total_false_positives += 1
            
            total_topk_tokens += len(topk_indices)
        
        hit_rate = hit_count / total_samples if total_samples > 0 else 0.0
        false_positive_rate = total_false_positives / total_topk_tokens if total_topk_tokens > 0 else 0.0
        precision = total_true_positives / total_topk_tokens if total_topk_tokens > 0 else 0.0
        
        results.append({
            'top_k': k,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'false_positive_count': total_false_positives,
            'false_positive_rate': false_positive_rate,
            'precision': precision
        })
    
    return {
        'total_samples': total_samples,
        'results': results
    }


def save_results(results: Dict, output_dir: Path, model_name: str):
    """
    Save results to JSON and CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['model'] = model_name
    results['setting'] = model_name
    
    # Save JSON
    json_path = output_dir / "topk_hit_rate.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON to {json_path}")
    
    # Save CSV
    csv_path = output_dir / "topk_hit_rate.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['top_k', 'hit_count', 'hit_rate', 'false_positive_count', 'false_positive_rate', 'precision']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results['results'])
    print(f"  Saved CSV to {csv_path}")


def analyze_topk_hit_rate(history_dir: Path, output_dir: Path, max_k: int = 10):
    """
    Main analysis function.
    """
    model_name = history_dir.name
    print(f"\nAnalyzing Top-K hit rate for {model_name}...")
    
    # Load histories
    histories = load_histories(history_dir)
    
    # Calculate hit rates
    results = calculate_topk_hit_rate(histories, max_k)
    
    # Save results
    save_results(results, output_dir, model_name)
    
    print(f"  ✓ Analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Top-K hit rate")
    parser.add_argument("history_dir", type=str, help="Path to history directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum K value")
    
    args = parser.parse_args()
    
    analyze_topk_hit_rate(
        Path(args.history_dir),
        Path(args.output_dir),
        args.max_k
    )
