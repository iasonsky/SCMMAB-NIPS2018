#!/usr/bin/env python3
"""
Script to view and analyze the results from the four-variable SCM bandit experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_final_regret_results(filepath):
    """Load and display final regret results at T=10000."""
    print("📊 Final Regret Results at T=10000")
    print("=" * 50)
    
    try:
        data = np.load(filepath)
        
        # Sort keys for consistent display
        keys = sorted(data.keys())
        
        print(f"{'Algorithm':<20} {'Mean Regret':<12} {'Std Dev':<10} {'95% CI Margin':<12}")
        print("-" * 60)
        
        for key in keys:
            values = data[key]
            mean_regret, std_regret, ci_margin = values
            
            # Format algorithm name for better display
            if '_' in key:
                strategy, algo = key.split('_', 1)
                display_name = f"{strategy} ({algo})"
            else:
                display_name = key
            
            print(f"{display_name:<20} {mean_regret:<12.2f} {std_regret:<10.2f} {ci_margin:<12.2f}")
        
        print()
        return data
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def load_arm_rewards(filepath):
    """Load and display arm reward information."""
    print("🎰 Arm Reward Information")
    print("=" * 30)
    
    try:
        data = np.load(filepath)
        mu = data['a']  # Arm expected rewards
        
        print(f"Number of arms: {len(mu)}")
        print(f"Optimal arm reward: {np.max(mu):.4f}")
        print(f"Arm rewards: {mu}")
        print()
        
        return mu
        
    except FileNotFoundError:
        print(f"❌ Arm rewards file not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading arm rewards: {e}")
        return None

def load_experiment_data(filepath):
    """Load and display basic experiment data."""
    print("🔬 Experiment Data")
    print("=" * 20)
    
    try:
        data = np.load(filepath)
        p_u = data['a']  # Probability distribution
        
        print(f"Probability distribution shape: {p_u.shape}")
        print(f"Probability distribution: {p_u}")
        print()
        
        return p_u
        
    except FileNotFoundError:
        print(f"❌ Experiment data file not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading experiment data: {e}")
        return None

def compare_algorithms(final_regret_data):
    """Compare algorithm performance and rank them."""
    if final_regret_data is None:
        return
    
    print("🏆 Algorithm Performance Ranking")
    print("=" * 40)
    
    # Extract mean regrets and create ranking
    rankings = []
    for key, values in final_regret_data.items():
        mean_regret = values[0]
        ci_margin = values[2]
        
        if '_' in key:
            strategy, algo = key.split('_', 1)
            display_name = f"{strategy} ({algo})"
        else:
            display_name = key
            
        rankings.append((display_name, mean_regret, ci_margin))
    
    # Sort by mean regret (lower is better)
    rankings.sort(key=lambda x: x[1])
    
    print(f"{'Rank':<4} {'Algorithm':<20} {'Mean Regret':<12} {'95% CI':<12}")
    print("-" * 50)
    
    for i, (name, mean_regret, ci_margin) in enumerate(rankings, 1):
        print(f"{i:<4} {name:<20} {mean_regret:<12.2f} {ci_margin:<12.2f}")
    
    print()

def main():
    """Main function to display all results."""
    print("🔍 Four-Variable SCM Bandit Experiment Results Viewer")
    print("=" * 60)
    print()
    
    # Define file paths
    results_dir = Path("four_variable_results")
    final_regret_file = results_dir / "final_regret_T10000.npz"
    arm_rewards_file = results_dir / "mu.npz"
    experiment_data_file = results_dir / "p_u.npz"
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run the experiment first using test_four_variable_bandits.py")
        return
    
    # Load and display results
    final_regret_data = load_final_regret_results(final_regret_file)
    arm_rewards = load_arm_rewards(arm_rewards_file)
    experiment_data = load_experiment_data(experiment_data_file)
    
    # Compare algorithms
    compare_algorithms(final_regret_data)
    
    # Additional analysis
    if final_regret_data is not None:
        print("📈 Additional Analysis")
        print("=" * 25)
        
        # Find best and worst performing algorithms
        regrets = [(key, values[0]) for key, values in final_regret_data.items()]
        best_algo = min(regrets, key=lambda x: x[1])
        worst_algo = max(regrets, key=lambda x: x[1])
        
        print(f"Best performing: {best_algo[0]} (regret: {best_algo[1]:.2f})")
        print(f"Worst performing: {worst_algo[0]} (regret: {worst_algo[1]:.2f})")
        
        # Calculate performance gap
        performance_gap = worst_algo[1] - best_algo[1]
        print(f"Performance gap: {performance_gap:.2f}")
        
        # Check for statistical significance (simple overlap check)
        print("\n🔍 Statistical Significance Check")
        print("(Algorithms with non-overlapping 95% CIs are significantly different)")
        
        regrets_with_ci = [(key, values[0], values[2]) for key, values in final_regret_data.items()]
        
        for i, (key1, mean1, ci1) in enumerate(regrets_with_ci):
            for j, (key2, mean2, ci2) in enumerate(regrets_with_ci[i+1:], i+1):
                # Check if confidence intervals overlap
                overlap = not (mean1 + ci1 < mean2 - ci2 or mean2 + ci2 < mean1 - ci1)
                significance = "Not significant" if overlap else "Significantly different"
                
                name1 = key1.replace('_', ' ') if '_' in key1 else key1
                name2 = key2.replace('_', ' ') if '_' in key2 else key2
                
                print(f"  {name1} vs {name2}: {significance}")

if __name__ == "__main__":
    main()
