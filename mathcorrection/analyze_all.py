#!/usr/bin/env python3
"""
Main script to run all analyses on correction history.
Traverses the correction_history directory and runs all three analysis scripts.
"""

import sys
from pathlib import Path
from tqdm import tqdm
from analyze_confidence_distribution import analyze_confidence_distribution
from analyze_topk_hit_rate import analyze_topk_hit_rate
from analyze_outlier_detection import analyze_outlier_detection


def find_history_dirs(base_dir: Path) -> list:
    """
    Find all leaf directories containing .pt files.
    These are the model/setting directories we need to analyze.
    """
    history_dirs = []
    
    for path in base_dir.rglob("*.pt"):
        history_dir = path.parent
        if history_dir not in history_dirs:
            history_dirs.append(history_dir)
    
    return sorted(history_dirs)


def map_history_to_analysis_dir(history_dir: Path, base_history_dir: Path, base_analysis_dir: Path) -> Path:
    """
    Map history directory to corresponding analysis directory.
    
    Example:
        history_dir: correction_history/refined_steps2/remove_all/.../Dream-v0-Base-7B_unary_10_wrong_1
        base_history_dir: correction_history
        base_analysis_dir: correction_analysis/confidence_analysis
        
        output: correction_analysis/confidence_analysis/correction_history/refined_steps2/remove_all/.../Dream-v0-Base-7B_unary_10_wrong_1
    """
    relative_path = history_dir.relative_to(base_history_dir)
    analysis_dir = base_analysis_dir / base_history_dir.name / relative_path
    return analysis_dir


def check_analysis_complete(analysis_dir: Path) -> bool:
    """
    Check if all analysis output files exist.
    
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        "confidence_distribution.png",
        "topk_hit_rate.json",
        "topk_hit_rate.csv",
        "outlier_detection_summary.json",
        "outlier_detection_summary.csv"
    ]
    
    for filename in required_files:
        file_path = analysis_dir / filename
        if not file_path.exists():
            return False
    
    return True


def analyze_all(base_history_dir: Path, base_analysis_dir: Path, skip_if_exist: bool = False):
    """
    Run all analyses on all model/setting directories.
    
    Args:
        base_history_dir: Base directory containing correction history
        base_analysis_dir: Base directory for analysis output
        skip_if_exist: If True, skip directories that already have all analysis outputs
    """
    print("=" * 80)
    print(f"Starting analysis")
    print(f"Input: {base_history_dir}")
    print(f"Output: {base_analysis_dir}")
    if skip_if_exist:
        print(f"Mode: Skip if output files already exist")
    print("=" * 80)
    
    # Find all history directories
    history_dirs = find_history_dirs(base_history_dir)
    print(f"\nFound {len(history_dirs)} model/setting directories to analyze:")
    for history_dir in history_dirs:
        print(f"  - {history_dir}")
    
    # Analyze each directory
    skipped_count = 0
    processed_count = 0
    
    # Use tqdm for progress bar
    with tqdm(total=len(history_dirs), desc="Processing", unit="dir") as pbar:
        for i, history_dir in enumerate(history_dirs, 1):
            pbar.set_description(f"Processing {history_dir.name}")
            
            # Map to analysis directory
            analysis_dir = map_history_to_analysis_dir(history_dir, base_history_dir, base_analysis_dir)
            
            # Check if analysis already exists
            if skip_if_exist and check_analysis_complete(analysis_dir):
                skipped_count += 1
                pbar.set_postfix({"skipped": skipped_count, "processed": processed_count})
                pbar.update(1)
                continue
            
            # Run all three analyses
            try:
                analyze_confidence_distribution(history_dir, analysis_dir)
                analyze_topk_hit_rate(history_dir, analysis_dir)
                analyze_outlier_detection(history_dir, analysis_dir)
                processed_count += 1
            except Exception as e:
                tqdm.write(f"  ✗ Error processing {history_dir.name}: {e}")
            
            pbar.set_postfix({"skipped": skipped_count, "processed": processed_count})
            pbar.update(1)
    
    print(f"\n{'=' * 80}")
    print(f"All analyses complete!")
    print(f"  Processed: {processed_count}")
    if skip_if_exist:
        print(f"  Skipped: {skipped_count}")
    print(f"Results saved to: {base_analysis_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse
    
    # Get the project root directory (dlm_agents)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # Go up from codecorrection/ to dlm_agents/
    
    parser = argparse.ArgumentParser(description="Run all analyses on correction history")
    parser.add_argument(
        "--history_dir",
        type=str,
        default=str(project_root / "correction_history"),
        help="Base directory containing correction history"
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=str(project_root / "correction_analysis" / "confidence_analysis"),
        help="Base directory for analysis output"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix for directories (e.g., 'AR_correction'). If set, will use <prefix>_history and <prefix>_analysis"
    )
    parser.add_argument(
        "--skip_if_exist",
        action="store_true",
        help="Skip directories that already have all analysis output files"
    )
    
    args = parser.parse_args()
    
    # Handle prefix
    if args.prefix:
        base_history_dir = project_root / f"{args.prefix}_history"
        base_analysis_dir = project_root / f"{args.prefix}_analysis"
    else:
        base_history_dir = Path(args.history_dir)
        base_analysis_dir = Path(args.analysis_dir)
    
    if not base_history_dir.exists():
        print(f"Error: History directory does not exist: {base_history_dir}")
        sys.exit(1)
    
    analyze_all(base_history_dir, base_analysis_dir, skip_if_exist=args.skip_if_exist)
