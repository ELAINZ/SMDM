#!/usr/bin/env python3
"""
Analyze outlier detection metrics for error tokens.
Quantifies how much error tokens are outliers compared to clean tokens.
"""

import torch
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from scipy import stats


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


def calculate_percentile_rank(error_confs: np.ndarray, clean_confs: np.ndarray) -> Dict:
    """
    Calculate percentile rank of error tokens in clean token distribution.
    """
    all_error_ranks = []
    sample_mean_ranks = []
    sample_min_ranks = []
    
    for error_conf in error_confs:
        rank = stats.percentileofscore(clean_confs, error_conf, kind='rank')
        all_error_ranks.append(rank)
    
    # For sample-level stats, we need to group by sample
    # But since we're aggregating, we just use all error tokens
    if len(all_error_ranks) > 0:
        sample_mean_ranks.append(np.mean(all_error_ranks))
        sample_min_ranks.append(np.min(all_error_ranks))
    
    all_error_ranks = np.array(all_error_ranks)
    
    return {
        'mean': float(np.mean(all_error_ranks)) if len(all_error_ranks) > 0 else 0.0,
        'std': float(np.std(all_error_ranks)) if len(all_error_ranks) > 0 else 0.0,
        'median': float(np.median(all_error_ranks)) if len(all_error_ranks) > 0 else 0.0,
        'p25': float(np.percentile(all_error_ranks, 25)) if len(all_error_ranks) > 0 else 0.0,
        'p75': float(np.percentile(all_error_ranks, 75)) if len(all_error_ranks) > 0 else 0.0,
        'p5_ratio': float(np.mean(all_error_ranks < 5)) if len(all_error_ranks) > 0 else 0.0,
        'p10_ratio': float(np.mean(all_error_ranks < 10)) if len(all_error_ranks) > 0 else 0.0,
        'p25_ratio': float(np.mean(all_error_ranks < 25)) if len(all_error_ranks) > 0 else 0.0,
        'sample_mean_mean': float(np.mean(sample_mean_ranks)) if len(sample_mean_ranks) > 0 else 0.0,
        'sample_min_mean': float(np.mean(sample_min_ranks)) if len(sample_min_ranks) > 0 else 0.0,
    }


def calculate_z_score(error_confs: np.ndarray, clean_confs: np.ndarray) -> Dict:
    """
    Calculate z-scores of error tokens relative to clean token distribution.
    """
    clean_mean = np.mean(clean_confs)
    clean_std = np.std(clean_confs)
    
    if clean_std == 0:
        z_scores = np.zeros_like(error_confs)
    else:
        z_scores = (error_confs - clean_mean) / clean_std
    
    return {
        'mean': float(np.mean(z_scores)),
        'std': float(np.std(z_scores)),
        'median': float(np.median(z_scores)),
        'p25': float(np.percentile(z_scores, 25)),
        'p75': float(np.percentile(z_scores, 75)),
        'sample_mean_mean': float(np.mean(z_scores)),
        'sample_min_mean': float(np.min(z_scores)) if len(z_scores) > 0 else 0.0,
    }


def calculate_quantile_gap(error_confs: np.ndarray, clean_confs: np.ndarray) -> Dict:
    """
    Calculate gap between error tokens and clean token quantiles.
    """
    q1 = np.percentile(clean_confs, 25)
    median = np.percentile(clean_confs, 50)
    q3 = np.percentile(clean_confs, 75)
    
    gaps_to_q1 = np.maximum(0, q1 - error_confs)
    gaps_to_median = median - error_confs
    gaps_to_q3 = q3 - error_confs
    
    return {
        'gap_to_q1': {
            'mean': float(np.mean(gaps_to_q1)),
            'std': float(np.std(gaps_to_q1)),
            'sample_mean_mean': float(np.mean(gaps_to_q1)),
        },
        'gap_to_median': {
            'mean': float(np.mean(gaps_to_median)),
            'std': float(np.std(gaps_to_median)),
            'sample_mean_mean': float(np.mean(gaps_to_median)),
        },
        'gap_to_q3': {
            'mean': float(np.mean(gaps_to_q3)),
            'std': float(np.std(gaps_to_q3)),
            'sample_mean_mean': float(np.mean(gaps_to_q3)),
        }
    }


def calculate_relative_distance(error_confs: np.ndarray, clean_confs: np.ndarray) -> Dict:
    """
    Calculate relative distance of error tokens in clean token range.
    """
    min_clean = np.min(clean_confs)
    max_clean = np.max(clean_confs)
    
    if max_clean == min_clean:
        relative_positions = np.zeros_like(error_confs)
    else:
        relative_positions = (error_confs - min_clean) / (max_clean - min_clean)
    
    return {
        'mean': float(np.mean(relative_positions)),
        'std': float(np.std(relative_positions)),
        'median': float(np.median(relative_positions)),
        'sample_mean_mean': float(np.mean(relative_positions)),
        'sample_min_mean': float(np.min(relative_positions)) if len(relative_positions) > 0 else 0.0,
    }


def calculate_iqr_outlier(histories: List[Dict]) -> Dict:
    """
    Calculate IQR-based outlier detection metrics.
    """
    n_outlier_tokens_list = []
    outlier_error_ratio_list = []
    error_outlier_ratio_list = []
    samples_with_outliers = 0
    
    for sample in histories:
        if len(sample['steps']) == 0:
            continue
        
        step1 = sample['steps'][0]
        conf_variable = step1['conf_variable'].cpu().numpy()
        
        error_positions = sample.get('error_positions', [])
        if isinstance(error_positions, str):
            error_positions = json.loads(error_positions) if error_positions else []
        error_positions = set(error_positions)
        
        if len(error_positions) == 0:
            continue
        
        # Calculate IQR for clean tokens only
        clean_confs = [conf_variable[i] for i in range(len(conf_variable)) if i not in error_positions]
        if len(clean_confs) == 0:
            continue
        
        q1 = np.percentile(clean_confs, 25)
        q3 = np.percentile(clean_confs, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        
        # Detect outliers in all variable tokens
        outlier_positions = [i for i, conf in enumerate(conf_variable) if conf < lower_bound]
        n_outlier_tokens = len(outlier_positions)
        
        # Count how many outliers are error tokens
        n_outlier_error_tokens = sum(1 for pos in outlier_positions if pos in error_positions)
        
        # Calculate ratios
        if n_outlier_tokens > 0:
            outlier_error_ratio = n_outlier_error_tokens / n_outlier_tokens
            samples_with_outliers += 1
        else:
            outlier_error_ratio = 0.0
        
        error_outlier_ratio = n_outlier_error_tokens / len(error_positions)
        
        n_outlier_tokens_list.append(n_outlier_tokens)
        outlier_error_ratio_list.append(outlier_error_ratio)
        error_outlier_ratio_list.append(error_outlier_ratio)
    
    total_samples = len(n_outlier_tokens_list)
    
    return {
        'n_outlier_tokens_mean': float(np.mean(n_outlier_tokens_list)) if total_samples > 0 else 0.0,
        'n_outlier_tokens_std': float(np.std(n_outlier_tokens_list)) if total_samples > 0 else 0.0,
        'n_outlier_tokens_median': float(np.median(n_outlier_tokens_list)) if total_samples > 0 else 0.0,
        'outlier_error_ratio_mean': float(np.mean(outlier_error_ratio_list)) if total_samples > 0 else 0.0,
        'outlier_error_ratio_std': float(np.std(outlier_error_ratio_list)) if total_samples > 0 else 0.0,
        'error_outlier_ratio_mean': float(np.mean(error_outlier_ratio_list)) if total_samples > 0 else 0.0,
        'error_outlier_ratio_std': float(np.std(error_outlier_ratio_list)) if total_samples > 0 else 0.0,
        'samples_with_outliers': samples_with_outliers,
        'samples_with_outliers_ratio': float(samples_with_outliers / total_samples) if total_samples > 0 else 0.0,
    }


def calculate_rank_gap(histories: List[Dict]) -> Dict:
    """
    Calculate rank-based metrics for error tokens.
    """
    all_ranks = []
    all_gaps = []
    sample_mean_ranks = []
    sample_min_ranks = []
    samples_in_top_3 = 0
    samples_in_top_5 = 0
    samples_in_top_10 = 0
    
    for sample in histories:
        if len(sample['steps']) == 0:
            continue
        
        step1 = sample['steps'][0]
        conf_variable = step1['conf_variable'].cpu().numpy()
        
        error_positions = sample.get('error_positions', [])
        if isinstance(error_positions, str):
            error_positions = json.loads(error_positions) if error_positions else []
        error_positions = set(error_positions)
        
        if len(error_positions) == 0:
            continue
        
        # Sort all tokens by confidence (ascending)
        sorted_indices = np.argsort(conf_variable)
        
        # Find ranks of error tokens (1-indexed)
        error_ranks = []
        for pos in error_positions:
            rank = np.where(sorted_indices == pos)[0][0] + 1
            error_ranks.append(rank)
            all_ranks.append(rank)
            all_gaps.append(rank - 1)
        
        # Sample-level stats
        sample_mean_ranks.append(np.mean(error_ranks))
        sample_min_ranks.append(np.min(error_ranks))
        
        # Check if in top-K
        min_rank = np.min(error_ranks)
        if min_rank <= 3:
            samples_in_top_3 += 1
        if min_rank <= 5:
            samples_in_top_5 += 1
        if min_rank <= 10:
            samples_in_top_10 += 1
    
    total_samples = len(sample_mean_ranks)
    
    return {
        'mean_rank': float(np.mean(all_ranks)) if len(all_ranks) > 0 else 0.0,
        'mean_gap': float(np.mean(all_gaps)) if len(all_gaps) > 0 else 0.0,
        'median_rank': float(np.median(all_ranks)) if len(all_ranks) > 0 else 0.0,
        'min_rank_mean': float(np.mean(sample_min_ranks)) if len(sample_min_ranks) > 0 else 0.0,
        'samples_in_top_3': samples_in_top_3,
        'samples_in_top_3_ratio': float(samples_in_top_3 / total_samples) if total_samples > 0 else 0.0,
        'samples_in_top_5': samples_in_top_5,
        'samples_in_top_5_ratio': float(samples_in_top_5 / total_samples) if total_samples > 0 else 0.0,
        'samples_in_top_10': samples_in_top_10,
        'samples_in_top_10_ratio': float(samples_in_top_10 / total_samples) if total_samples > 0 else 0.0,
    }


def calculate_over_conservative_ratio(histories: List[Dict]) -> Dict:
    """
    Calculate over-conservative ratio: proportion of clean tokens with confidence
    lower than the maximum confidence of error tokens in each sample.
    
    This metric identifies if the model assigns low confidence to many clean tokens.
    """
    sample_ratios = []
    sample_over_conservative_counts = []
    
    for sample in histories:
        if len(sample['steps']) == 0:
            continue
        
        step1 = sample['steps'][0]
        conf_variable = step1['conf_variable'].cpu().numpy()
        
        error_positions = sample.get('error_positions', [])
        if isinstance(error_positions, str):
            error_positions = json.loads(error_positions) if error_positions else []
        error_positions_set = set(error_positions)
        
        if len(error_positions) == 0:
            continue
        
        # Get error token confidences
        error_confs = [conf_variable[i] for i in error_positions]
        max_error_conf = np.max(error_confs)
        
        # Count clean tokens with confidence lower than max_error_conf
        over_conservative_count = 0
        total_clean_tokens = 0
        
        for i, conf in enumerate(conf_variable):
            if i not in error_positions_set:
                total_clean_tokens += 1
                if conf < max_error_conf:
                    over_conservative_count += 1
        
        if total_clean_tokens > 0:
            ratio = over_conservative_count / total_clean_tokens
            sample_ratios.append(ratio)
            sample_over_conservative_counts.append(over_conservative_count)
    
    return {
        'mean_ratio': float(np.mean(sample_ratios)) if len(sample_ratios) > 0 else 0.0,
        'std_ratio': float(np.std(sample_ratios)) if len(sample_ratios) > 0 else 0.0,
        'median_ratio': float(np.median(sample_ratios)) if len(sample_ratios) > 0 else 0.0,
        'mean_count': float(np.mean(sample_over_conservative_counts)) if len(sample_over_conservative_counts) > 0 else 0.0,
        'std_count': float(np.std(sample_over_conservative_counts)) if len(sample_over_conservative_counts) > 0 else 0.0,
    }


def extract_all_confidences(histories: List[Dict]) -> tuple:
    """
    Extract all error and clean confidences across all samples.
    """
    error_confs = []
    clean_confs = []
    total_error_tokens = 0
    
    for sample in histories:
        if len(sample['steps']) == 0:
            continue
        
        step1 = sample['steps'][0]
        conf_variable = step1['conf_variable'].cpu().numpy()
        
        error_positions = sample.get('error_positions', [])
        if isinstance(error_positions, str):
            error_positions = json.loads(error_positions) if error_positions else []
        error_positions = set(error_positions)
        
        total_error_tokens += len(error_positions)
        
        for i, conf in enumerate(conf_variable):
            if i in error_positions:
                error_confs.append(conf)
            else:
                clean_confs.append(conf)
    
    return np.array(error_confs), np.array(clean_confs), total_error_tokens


def save_results(results: Dict, output_dir: Path):
    """
    Save results to JSON and CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "outlier_detection_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON to {json_path}")
    
    # Save CSV (flattened)
    csv_path = output_dir / "outlier_detection_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        
        # Flatten nested dict
        def flatten_dict(d, prefix=''):
            rows = []
            for k, v in d.items():
                if isinstance(v, dict):
                    rows.extend(flatten_dict(v, prefix=f"{prefix}{k}."))
                else:
                    rows.append([f"{prefix}{k}", v])
            return rows
        
        writer.writerows(flatten_dict(results))
    
    print(f"  Saved CSV to {csv_path}")


def analyze_outlier_detection(history_dir: Path, output_dir: Path):
    """
    Main analysis function.
    """
    model_name = history_dir.name
    print(f"\nAnalyzing outlier detection for {model_name}...")
    
    # Load histories
    histories = load_histories(history_dir)
    
    # Extract all confidences
    error_confs, clean_confs, total_error_tokens = extract_all_confidences(histories)
    
    print(f"  Total error tokens: {total_error_tokens}")
    print(f"  Total clean tokens: {len(clean_confs)}")
    
    # Calculate mean confidences
    mean_error_conf = float(np.mean(error_confs)) if len(error_confs) > 0 else None
    mean_clean_conf = float(np.mean(clean_confs)) if len(clean_confs) > 0 else None
    
    # Calculate all metrics
    results = {
        'model': model_name,
        'setting': model_name,
        'total_samples': len(histories),
        'total_error_tokens': int(total_error_tokens),
        'mean_error_conf': mean_error_conf,
        'mean_clean_conf': mean_clean_conf,
        'percentile_rank': calculate_percentile_rank(error_confs, clean_confs),
        'z_score': calculate_z_score(error_confs, clean_confs),
        'quantile_gap': calculate_quantile_gap(error_confs, clean_confs),
        'relative_distance': calculate_relative_distance(error_confs, clean_confs),
        'iqr_outlier': calculate_iqr_outlier(histories),
        'rank_gap': calculate_rank_gap(histories),
        'over_conservative': calculate_over_conservative_ratio(histories),
    }
    
    # Save results
    save_results(results, output_dir)
    
    print(f"  ✓ Analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze outlier detection metrics")
    parser.add_argument("history_dir", type=str, help="Path to history directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    
    args = parser.parse_args()
    
    analyze_outlier_detection(
        Path(args.history_dir),
        Path(args.output_dir)
    )
