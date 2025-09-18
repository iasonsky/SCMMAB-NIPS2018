#!/usr/bin/env python3
"""
Script to create a results table similar to Table 1 from the paper.
Generates a table showing average cumulative regret at T=10,000 for different
bandit algorithms and arm selection strategies.
"""

import numpy as np
import os
import glob
from typing import Dict, List, Tuple

def load_final_regret_data(experiment_dir: str) -> Dict[str, float]:
    """Load final regret data from an experiment directory."""
    final_regret_file = os.path.join(experiment_dir, "final_regret_T10000.npz")
    if not os.path.exists(final_regret_file):
        # Try T=1000 if T=10000 doesn't exist
        final_regret_file = os.path.join(experiment_dir, "final_regret_T1000.npz")
        if not os.path.exists(final_regret_file):
            return {}
    
    data = np.load(final_regret_file)
    results = {}
    
    for key in data.keys():
        # Extract mean regret (first value in the array)
        mean_regret = data[key][0]
        results[key] = mean_regret
    
    return results

def create_results_table(experiment_dirs: List[str], scenario_names: List[str]) -> str:
    """Create a formatted table from experiment results."""
    
    # Collect all results
    all_results = {}
    for exp_dir, scenario_name in zip(experiment_dirs, scenario_names):
        results = load_final_regret_data(exp_dir)
        if results:
            all_results[scenario_name] = results
    
    # Define the table structure
    algorithms = ['TS', 'UCB']
    strategies = ['POMIS', 'MIS', 'BF']  # BF = Brute-force
    
    # Create table header
    table_lines = []
    table_lines.append("\\begin{table}[h]")
    table_lines.append("\\centering")
    table_lines.append("\\begin{tabular}{l" + "c" * len(scenario_names) + "}")
    table_lines.append("\\hline")
    
    # Header row
    header = " & " + " & ".join(scenario_names) + " \\\\"
    table_lines.append(header)
    table_lines.append("\\hline")
    
    # Data rows
    for algo in algorithms:
        for i, strategy in enumerate(strategies):
            row_parts = []
            
            # First column: algorithm name (only for first strategy)
            if i == 0:
                row_parts.append(f"\\multirow{{{len(strategies)}}}{{*}}{{{algo}}}")
            else:
                row_parts.append("")
            
            # Strategy name
            row_parts.append(strategy)
            
            # Data columns
            for scenario in scenario_names:
                if scenario in all_results:
                    # Map strategy names to our data keys
                    if strategy == 'POMIS':
                        key = f"POMIS_{algo}"
                    elif strategy == 'MIS':
                        key = f"MIS_{algo}"
                    elif strategy == 'BF':
                        key = f"Brute-force_{algo}"
                    
                    if key in all_results[scenario]:
                        value = all_results[scenario][key]
                        row_parts.append(f"{value:.2f}")
                    else:
                        row_parts.append("//")
                else:
                    row_parts.append("//")
            
            # Join row parts
            row = " & ".join(row_parts) + " \\\\"
            table_lines.append(row)
        
        # Add horizontal line after each algorithm
        if algo != algorithms[-1]:
            table_lines.append("\\hline")
    
    table_lines.append("\\hline")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\caption{Average cumulative regret at T = 10,000}")
    table_lines.append("\\label{tab:results}")
    table_lines.append("\\end{table}")
    
    return "\n".join(table_lines)

def create_markdown_table(experiment_dirs: List[str], scenario_names: List[str]) -> str:
    """Create a markdown table from experiment results."""
    
    # Collect all results
    all_results = {}
    for exp_dir, scenario_name in zip(experiment_dirs, scenario_names):
        results = load_final_regret_data(exp_dir)
        if results:
            all_results[scenario_name] = results
    
    # Define the table structure
    algorithms = ['TS', 'UCB']
    strategies = ['POMIS', 'MIS', 'BF']  # BF = Brute-force
    
    # Create table
    table_lines = []
    
    # Header row
    header = "| Algorithm | Strategy | " + " | ".join(scenario_names) + " |"
    table_lines.append(header)
    
    # Separator row
    separator = "|" + "|".join(["---"] * (len(scenario_names) + 2)) + "|"
    table_lines.append(separator)
    
    # Data rows
    for algo in algorithms:
        for i, strategy in enumerate(strategies):
            row_parts = []
            
            # First column: algorithm name (only for first strategy)
            if i == 0:
                row_parts.append(f"**{algo}**")
            else:
                row_parts.append("")
            
            # Second column: strategy name
            row_parts.append(strategy)
            
            # Data columns
            for scenario in scenario_names:
                if scenario in all_results:
                    # Map strategy names to our data keys
                    if strategy == 'POMIS':
                        key = f"POMIS_{algo}"
                    elif strategy == 'MIS':
                        key = f"MIS_{algo}"
                    elif strategy == 'BF':
                        key = f"Brute-force_{algo}"
                    
                    if key in all_results[scenario]:
                        value = all_results[scenario][key]
                        row_parts.append(f"{value:.2f}")
                    else:
                        row_parts.append("//")
                else:
                    row_parts.append("//")
            
            # Join row parts
            row = "| " + " | ".join(row_parts) + " |"
            table_lines.append(row)
    
    return "\n".join(table_lines)

def main():
    # Define experiment directories and scenario names
    experiment_dirs = [
        "experiment_results/frontdoor_scm_results",
        "experiment_results/four_variable_scm_results", 
        "experiment_results/six_variable_scm_results"
    ]
    
    scenario_names = [
        "Frontdoor",
        "Four Variable", 
        "Six Variable"
    ]
    
    # Check which experiments actually exist
    existing_dirs = []
    existing_names = []
    for exp_dir, name in zip(experiment_dirs, scenario_names):
        if os.path.exists(exp_dir):
            existing_dirs.append(exp_dir)
            existing_names.append(name)
            print(f"✅ Found: {name} ({exp_dir})")
        else:
            print(f"❌ Missing: {name} ({exp_dir})")
    
    if not existing_dirs:
        print("❌ No experiment results found!")
        return
    
    print(f"\\n📊 Creating table with {len(existing_dirs)} scenarios...")
    
    # Create LaTeX table
    latex_table = create_results_table(existing_dirs, existing_names)
    print("\\n" + "="*60)
    print("LATEX TABLE:")
    print("="*60)
    print(latex_table)
    
    # Create Markdown table
    markdown_table = create_markdown_table(existing_dirs, existing_names)
    print("\\n" + "="*60)
    print("MARKDOWN TABLE:")
    print("="*60)
    print(markdown_table)
    
    # Save tables to files
    with open("results_table.tex", "w") as f:
        f.write(latex_table)
    
    with open("results_table.md", "w") as f:
        f.write(markdown_table)
    
    # Also save in experiment_results directory
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/results_table.tex", "w") as f:
        f.write(latex_table)
    
    with open("experiment_results/results_table.md", "w") as f:
        f.write(markdown_table)
    
    print("\\n💾 Tables saved to:")
    print("   - results_table.tex (LaTeX)")
    print("   - results_table.md (Markdown)")
    print("   - experiment_results/results_table.tex (LaTeX)")
    print("   - experiment_results/results_table.md (Markdown)")

if __name__ == "__main__":
    main()
