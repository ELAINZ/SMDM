"""
Analyze refined results and plot success counts across different refine_settings.
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


def parse_path_info(file_path):
    """
    Parse refined_steps, refine_setting, algorithm, model_name, and initial_steps from file path.
    
    Supported path formats:
    1) humaneval: .../refined_steps{steps}/{setting}/{algorithm}/humaneval/{model}/steps{initial_steps}/.../...evaluated.jsonl
    2) mathcorrection/gsm8k: .../refined_steps{steps}/{setting}/{algorithm}/mathcorrection/gsm8k/{model_folder}/...evaluated.jsonl
       (initial_steps set to None when not in path)
    """
    path_str = str(file_path)
    
    # Extract refined_steps
    steps_match = re.search(r'/refined_steps(\d+)/', path_str)
    if not steps_match:
        return None
    refined_steps = int(steps_match.group(1))
    
    # Extract refine_setting (vanilla, remove_error_feedback, remove_all, ...)
    setting_match = re.search(r'/refined_steps\d+/([^/]+)/', path_str)
    if not setting_match:
        return None
    refine_setting = setting_match.group(1)
    
    # Format 1: humaneval
    algorithm_match = re.search(r'/refined_steps\d+/[^/]+/([^/]+)/humaneval/', path_str)
    if algorithm_match:
        algorithm = algorithm_match.group(1)
        model_match = re.search(r'/humaneval/([^/]+)/', path_str)
        if not model_match:
            return None
        model_name = model_match.group(1)
        initial_steps_match = re.search(r'/humaneval/[^/]+/steps(\d+)/', path_str)
        if not initial_steps_match:
            return None
        initial_steps = int(initial_steps_match.group(1))
        return {
            'refined_steps': refined_steps,
            'refine_setting': refine_setting,
            'algorithm': algorithm,
            'model_name': model_name,
            'initial_steps': initial_steps,
        }
    
    # Format 2: mathcorrection/gsm8k (or other dataset under mathcorrection)
    math_match = re.search(r'/refined_steps\d+/[^/]+/([^/]+)/mathcorrection/[^/]+/([^/]+)/', path_str)
    if math_match:
        algorithm = math_match.group(1)
        model_name = math_match.group(2)  # e.g. cdlm_model.pth_number_propagate_1_wrong_1_evaluated
        # No steps in path; use None so (model_name, algorithm, None) differentiates models
        return {
            'refined_steps': refined_steps,
            'refine_setting': refine_setting,
            'algorithm': algorithm,
            'model_name': model_name,
            'initial_steps': None,
        }
    
    return None


def count_successes(evaluated_jsonl_path):
    """Count successful (test_passed=True) samples in evaluated jsonl file."""
    success_count = 0
    total_count = 0
    
    try:
        with open(evaluated_jsonl_path, 'r') as f:
            for line in f:
                total_count += 1
                item = json.loads(line)
                test_passed = item.get('test_passed', False)
                # Handle both bool and string representations
                if test_passed is True or test_passed == 'true' or test_passed == 'True':
                    success_count += 1
    except Exception as e:
        print(f"Error reading {evaluated_jsonl_path}: {e}")
        return None, None
    
    return success_count, total_count


def collect_results(results_dir, model_name=None, algorithm=None, initial_steps=None):
    """
    Collect success counts for all refine_setting and refined_steps combinations.
    
    Returns:
        dict: {
            (model_name, algorithm, initial_steps): {
                refine_setting: {
                    refined_steps: (success_count, total_count)
                }
            }
        }
    """
    results_dir = Path(results_dir)
    
    # Find all evaluated jsonl files
    evaluated_files = list(results_dir.glob("**/*_evaluated.jsonl"))
    
    # Organize by (model_name, algorithm, initial_steps) -> refine_setting -> refined_steps
    data = defaultdict(lambda: defaultdict(dict))
    
    for file_path in evaluated_files:
        path_info = parse_path_info(file_path)
        if path_info is None:
            continue
        
        # Filter by model_name if specified
        if model_name and path_info['model_name'] != model_name:
            continue
        
        # Filter by algorithm if specified
        if algorithm and path_info['algorithm'] != algorithm:
            continue
        
        # Filter by initial_steps if specified
        if initial_steps is not None and path_info['initial_steps'] != initial_steps:
            continue
        
        refined_steps = path_info['refined_steps']
        refine_setting = path_info['refine_setting']
        model = path_info['model_name']
        algo = path_info['algorithm']
        init_steps = path_info['initial_steps']
        
        success_count, total_count = count_successes(file_path)
        if success_count is not None:
            key = (model, algo, init_steps)
            data[key][refine_setting][refined_steps] = (success_count, total_count)
    
    return data


def plot_results(data, output_file=None, title=None):
    """Plot success counts vs refined_steps for different refine_settings."""
    if not data:
        print("No data to plot!")
        return
    
    # Prepare data for plotting
    settings = sorted(data.keys())
    all_steps = set()
    for setting_data in data.values():
        all_steps.update(setting_data.keys())
    all_steps = sorted(all_steps)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot each refine_setting with different styles
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']  # Different line styles
    
    for i, setting in enumerate(settings):
        success_counts = []
        steps_list = []
        
        for steps in all_steps:
            if steps in data[setting]:
                success_count, total_count = data[setting][steps]
                success_counts.append(success_count)
                steps_list.append(steps)
        
        if steps_list:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            
            # Plot with different styles to avoid overlap issues
            plt.plot(
                steps_list, success_counts,
                marker=marker,
                label=setting,
                linewidth=3.5,  # Thicker lines
                markersize=10,
                markeredgewidth=1.5,
                markeredgecolor='white',  # White edge to make markers stand out
                color=color,
                linestyle=linestyle,
                alpha=0.5,  # Fully opaque for better visibility
                zorder=10-i  # Different z-order for layering
            )
    
    plt.xlabel('Refined Steps', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Successful Refinements', fontsize=13, fontweight='bold')
    
    if title:
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
    else:
        plt.title(
            'Refinement Success Count by Setting and Steps',
            fontsize=15, fontweight='bold', pad=15
        )
    
    # Improve legend with better formatting
    plt.legend(
        fontsize=11,
        loc='best',
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    plt.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def print_summary(data, output_file=None):
    """Print summary statistics and optionally save to CSV file."""
    # Prepare data for console output
    lines = []
    
    lines.append("\n" + "=" * 80)
    lines.append("Summary Statistics")
    lines.append("=" * 80)
    
    all_steps = set()
    for setting_data in data.values():
        all_steps.update(setting_data.keys())
    all_steps = sorted(all_steps)
    
    # Print header
    settings = sorted(data.keys())
    header = f"{'Steps':<8}"
    for setting in settings:
        header += f"  {setting:<25}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Print data rows
    for steps in all_steps:
        row = f"{steps:<8}"
        for setting in settings:
            if steps in data[setting]:
                success, total = data[setting][steps]
                row += f"  {success:>3}/{total:<3} ({success/total*100:>5.1f}%)"
            else:
                row += f"  {'-':<25}"
        lines.append(row)
    
    lines.append("=" * 80)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Save to CSV file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV file
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header row
            csv_header = ['Steps'] + [setting for setting in settings]
            writer.writerow(csv_header)
            
            # Write data rows
            for steps in all_steps:
                row = [steps]
                for setting in settings:
                    if steps in data[setting]:
                        success, total = data[setting][steps]
                        percentage = success / total * 100
                        row.append(f"{success}/{total} ({percentage:.1f}%)")
                    else:
                        row.append("-")
                writer.writerow(row)
        
        print(f"\nSummary table saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze refined results and plot success counts"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/zhang2968/SMDM/correction_results",
        help="Directory containing refined results"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Filter by model name (e.g., 'LLaDA-8B-Base'). If not specified, analyzes all models."
    )
    parser.add_argument(
        "--algorithm", type=str, default=None,
        help="Filter by algorithm (e.g., 'self_conf-remask_vanilla'). If not specified, analyzes all algorithms."
    )
    parser.add_argument(
        "--initial_steps", type=int, default=None,
        help="Filter by initial steps (e.g., 128). If not specified, analyzes all initial steps."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots (default: 'refined_analysis_plots' with initial_steps subdirectories)"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Single output file path (overrides output_dir if specified)"
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Plot title (only used with --output_file)"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning results directory: {args.results_dir}")
    print(f"Model filter: {args.model_name or 'All models'}")
    print(f"Algorithm filter: {args.algorithm or 'All algorithms'}")
    print(f"Initial steps filter: {args.initial_steps or 'All initial steps'}")
    
    # Collect data organized by (model_name, algorithm, initial_steps)
    all_data = collect_results(args.results_dir, args.model_name, args.algorithm, args.initial_steps)
    
    if not all_data:
        print("No data found! Please check the results directory path.")
        exit(1)
    
    # Determine output directory
    if args.output_file:
        # Single file mode (backward compatibility)
        output_file = args.output_file
        model_algo_keys = list(all_data.keys())
        if len(model_algo_keys) == 1:
            model_name, algorithm, initial_steps = model_algo_keys[0]
            data = all_data[(model_name, algorithm, initial_steps)]
            title = args.title or f"Refinement Success Count - {model_name} ({algorithm}) [Initial: steps{initial_steps}]"
            # Save summary table to CSV file
            summary_file = Path(output_file).parent / Path(output_file).stem + "_summary.csv"
            print_summary(data, summary_file)
            plot_results(data, output_file, title)
            print("\nAnalysis complete!")
        else:
            print("Error: Multiple (model, algorithm, initial_steps) combinations found, "
                  "but --output_file specified.")
            print(f"Found: {model_algo_keys}")
            print("Please use --output_dir instead to generate separate plots.")
            exit(1)
    else:
        # Multi-file mode: generate one plot per (model, algorithm, initial_steps) combination
        results_dir = Path(args.results_dir)
        if args.output_dir:
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = Path("refined_analysis_plots")
        
        print(f"\nGenerating plots in: {base_output_dir}")
        print("=" * 80)
        
        # Group by initial_steps to create subdirectories
        plots_generated = 0
        for (model_name, algorithm, initial_steps), data in sorted(all_data.items()):
            # Create subdirectory for initial_steps (use "steps_any" when None, e.g. mathcorrection paths)
            steps_label = initial_steps if initial_steps is not None else "any"
            output_dir = base_output_dir / f"steps{steps_label}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            model_safe = model_name.replace('/', '_').replace(' ', '_')
            algo_safe = algorithm.replace(':', '_').replace('-', '_')
            filename = f"{model_safe}_{algo_safe}.png"
            output_file = output_dir / filename
            summary_file = output_dir / f"{model_safe}_{algo_safe}_summary.csv"
            
            title = f"Refinement Success Count - {model_name} ({algorithm}) [Initial: steps{steps_label}]"
            
            print(f"\nProcessing: {model_name} / {algorithm} / initial_steps={initial_steps}")
            print_summary(data, summary_file)
            plot_results(data, str(output_file), title)
            print(f"Saved: {output_file}")
            plots_generated += 1
        
        print("\n" + "=" * 80)
        print(f"Analysis complete! Generated {plots_generated} plots in {base_output_dir}")
