#!/usr/bin/env python3
"""
Analyze confidence distribution for error tokens vs clean tokens.
Generates a side-by-side distribution plot.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import json


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


def extract_step1_confidences(histories: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract step 1 confidences for error tokens and clean tokens.
    
    Returns:
        dict with keys 'error_confs' and 'clean_confs'
    """
    error_confs = []
    clean_confs = []
    
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
        
        # Split into error and clean
        for i, conf in enumerate(conf_variable):
            if i in error_positions:
                error_confs.append(float(conf))
            else:
                clean_confs.append(float(conf))
    
    return {
        'error_confs': error_confs,
        'clean_confs': clean_confs
    }


def plot_distribution(
    error_confs: List[float],
    clean_confs: List[float],
    output_path: Path,
    model_name: str
):
    """
    Create side-by-side distribution plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error tokens (left)
    ax_error = axes[0]
    ax_error.hist(error_confs, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax_error.set_xlabel('Confidence', fontsize=12)
    ax_error.set_ylabel('Density', fontsize=12)
    ax_error.set_title('Error Token Confidence Distribution', fontsize=14, fontweight='bold')
    ax_error.grid(alpha=0.3)
    
    # Statistics for error tokens
    error_mean = np.mean(error_confs)
    error_var = np.var(error_confs)
    error_count = len(error_confs)
    ax_error.text(
        0.95, 0.95,
        f'Count: {error_count}\nMean: {error_mean:.4f}\nVar: {error_var:.4f}',
        transform=ax_error.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Clean tokens (right)
    ax_clean = axes[1]
    ax_clean.hist(clean_confs, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax_clean.set_xlabel('Confidence', fontsize=12)
    ax_clean.set_ylabel('Density', fontsize=12)
    ax_clean.set_title('Clean Token Confidence Distribution', fontsize=14, fontweight='bold')
    ax_clean.grid(alpha=0.3)
    
    # Statistics for clean tokens
    clean_mean = np.mean(clean_confs)
    clean_var = np.var(clean_confs)
    clean_count = len(clean_confs)
    ax_clean.text(
        0.95, 0.95,
        f'Count: {clean_count}\nMean: {clean_mean:.4f}\nVar: {clean_var:.4f}',
        transform=ax_clean.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    # Overall title
    fig.suptitle(f'Confidence Distribution - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved distribution plot to {output_path}")


def analyze_confidence_distribution(history_dir: Path, output_dir: Path):
    """
    Main analysis function.
    """
    model_name = history_dir.name
    print(f"\nAnalyzing confidence distribution for {model_name}...")
    
    # Load histories
    histories = load_histories(history_dir)
    
    # Extract step 1 confidences
    confs = extract_step1_confidences(histories)
    
    # Plot
    output_path = output_dir / "confidence_distribution.png"
    plot_distribution(
        confs['error_confs'],
        confs['clean_confs'],
        output_path,
        model_name
    )
    
    print(f"  ✓ Analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze confidence distribution")
    parser.add_argument("history_dir", type=str, help="Path to history directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    
    args = parser.parse_args()
    
    analyze_confidence_distribution(
        Path(args.history_dir),
        Path(args.output_dir)
    )
