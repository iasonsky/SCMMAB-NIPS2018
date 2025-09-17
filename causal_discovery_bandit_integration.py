#!/usr/bin/env python3
"""
Causal Discovery + Bandit Integration

This script implements the correct pipeline:
1. Choose one DAG as ground truth (simple_markovian_SCM)
2. Simulate data from ground truth SCM
3. Learn MEC from data using PC algorithm
4. Use POMIS on all graphs in MEC and take union
5. Test bandit strategies using ground truth system
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Import the existing causal discovery pipeline
from npsem.causal_pipeline import run_causal_discovery_pipeline

# Import SCM examples and bandit framework
from npsem.NIPS2018POMIS_exp.scm_examples import chain_SCM
from npsem.bandits import play_bandits
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of, arm_types
from npsem.utils import subseq
from npsem.data_simulation import simulate_data_from_scm
# =============================================================================
# CAUSAL DISCOVERY + BANDIT EXPERIMENT INTEGRATION
# =============================================================================


def run_discovery_bandit_experiment(
    ground_truth_scm,
    Y: str = "Y",
    n_samples: int = 4000,
    alpha: float = 0.01,
    num_trials: int = 100,
    horizon: int = 5000,
    n_jobs: int = 1,
    save_plots: bool = True,
) -> Dict:
    """
    Run the complete causal discovery + bandit experiment pipeline.

    This follows the correct pipeline:
    1. Choose one DAG as ground truth (chain_SCM: Z -> X -> Y)
    2. Simulate data from ground truth SCM
    3. Learn MEC from data using PC algorithm
    4. Use POMIS on all graphs in MEC and take union
    5. Test bandit strategies using ground truth system

    Parameters:
    -----------
    ground_truth_scm : StructuralCausalModel
        Ground truth SCM to simulate from
    Y : str
        Target variable
    n_samples : int
        Number of samples to simulate
    alpha : float
        Significance level for PC algorithm
    num_trials : int
        Number of bandit trials
    horizon : int
        Bandit horizon
    n_jobs : int
        Number of parallel jobs
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    results : Dict
        Dictionary containing all experiment results
    """
    print("=" * 80)
    print("CAUSAL DISCOVERY + BANDIT EXPERIMENT PIPELINE")
    print("   (Ground Truth → Data → MEC → POMIS Union → Bandit Tests)")
    print("=" * 80)

    # Step 1: Simulate data from ground truth SCM
    print("\n1. Simulating data from ground truth SCM...")
    data, var_names = simulate_data_from_scm(ground_truth_scm, n_samples, seed=42)
    print(f"   Generated {data.shape[0]} samples for variables: {var_names}")
    print(f"   Ground truth SCM: {ground_truth_scm.G}")
    print("   Data correlations:")
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            print(f"     {var_names[i]} - {var_names[j]}: {corr:.4f}")

    # Step 2: Causal Discovery
    print("\n2. Running PC algorithm for causal discovery...")
    A_cpdag_cl, dags, pomis_union = run_causal_discovery_pipeline(
        data,
        var_names,
        ind_test="gsq",
        alpha=alpha,
        sanity_check=True,
        ground_truth_scm=ground_truth_scm,
    )
    print("   Discovered CPDAG adjacency matrix:")
    print(f"   {A_cpdag_cl}")
    print(f"   Found {len(dags)} DAGs in the MEC")
    print(f"   POMIS union: {pomis_union}")

    # Step 3: Run bandit experiments using ground truth SCM
    print("\n5. Running bandit experiments using ground truth SCM...")
    mu, arm_setting = SCM_to_bandit_machine(ground_truth_scm, Y)
    print(f"   Created {len(mu)} bandit arms")
    print(f"   Expected rewards: {mu}")

    # Define arm strategies to compare (using all available types)
    arm_strategies = arm_types()  # ["POMIS", "MIS", "Brute-force", "All-at-once"]
    bandit_algorithms = ["TS", "UCB"]

    results = {}

    for arm_strategy in arm_strategies:
        try:
            arm_selected = arms_of(arm_strategy, arm_setting, ground_truth_scm.G, Y)
            print(
                f"   {arm_strategy}: Found {len(arm_selected)} arms out of {len(arm_setting)} total"
            )
            if len(arm_selected) == 0:
                print(f"   Warning: No arms found for strategy {arm_strategy}")
                continue

            arm_corrector = np.vectorize(lambda x: arm_selected[x])

            for bandit_algo in bandit_algorithms:
                print(f"   Running {arm_strategy} + {bandit_algo}...")
                arm_played, rewards = play_bandits(
                    horizon, subseq(mu, arm_selected), bandit_algo, num_trials, n_jobs
                )
                results[(arm_strategy, bandit_algo)] = (
                    arm_corrector(arm_played),
                    rewards,
                )

        except Exception as e:
            print(f"   Error with {arm_strategy}: {e}")
            continue

    # Step 6: Analysis and visualization
    print("\n6. Analyzing results...")
    analysis_results = analyze_bandit_results(results, mu, horizon)

    if save_plots:
        print("\n7. Creating visualizations...")
        create_visualizations(results, mu, horizon, analysis_results)

    # Compile final results
    final_results = {
        "ground_truth": {
            "scm": ground_truth_scm,
            "data": data,
            "var_names": var_names,
        },
        "discovery": {
            "cpdag": A_cpdag_cl,
            "dags": dags,
            "pomis_union": pomis_union,
        },
        "bandit": {
            "arm_setting": arm_setting,
            "expected_rewards": mu,
            "results": results,
        },
        "analysis": analysis_results,
    }

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return final_results


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================


def analyze_bandit_results(results: Dict, mu: np.ndarray, horizon: int) -> Dict:
    """Analyze bandit experiment results."""
    mu_star = np.max(mu)
    analysis = {}

    for (arm_strategy, bandit_algo), (arms, rewards) in results.items():
        # Compute cumulative regret
        cumulative_rewards = np.cumsum(rewards, axis=1)
        optimal_cumulative_rewards = np.cumsum(np.ones(rewards.shape) * mu_star, axis=1)
        cumulative_regret = optimal_cumulative_rewards - cumulative_rewards

        # Final regret statistics
        final_regret = cumulative_regret[:, -1]
        mean_final_regret = np.mean(final_regret)
        std_final_regret = np.std(final_regret)

        analysis[(arm_strategy, bandit_algo)] = {
            "cumulative_regret": cumulative_regret,
            "final_regret": final_regret,
            "mean_final_regret": mean_final_regret,
            "std_final_regret": std_final_regret,
            "num_arms": len(np.unique(arms)),
            "arm_usage": np.bincount(arms.flatten()),
        }

    return analysis


def create_visualizations(results: Dict, mu: np.ndarray, horizon: int, analysis: Dict):
    """Create visualizations of the results."""
    # Set up plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Cumulative Regret Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    strategy_colors = {"Brute-force": "red", "MIS": "purple", "POMIS": "blue"}

    for (arm_strategy, bandit_algo), data in analysis.items():
        cumulative_regret = data["cumulative_regret"]
        mean_regret = np.mean(cumulative_regret, axis=0)
        std_regret = np.std(cumulative_regret, axis=0)

        # 95% confidence interval
        n_trials = cumulative_regret.shape[0]
        se_regret = std_regret / np.sqrt(n_trials)
        ci_margin = 1.96 * se_regret

        color = strategy_colors.get(arm_strategy, "gray")
        linestyle = "-" if bandit_algo == "TS" else "--"

        trials = np.arange(1, horizon + 1)
        ax.plot(
            trials,
            mean_regret,
            label=f"{arm_strategy} ({bandit_algo})",
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        ax.fill_between(
            trials,
            mean_regret - ci_margin,
            mean_regret + ci_margin,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Trials", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("Causal Discovery + Bandit Experiment Results", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("causal_discovery_bandit_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Final Regret Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    strategies = []
    final_regrets = []
    errors = []

    for (arm_strategy, bandit_algo), data in analysis.items():
        strategies.append(f"{arm_strategy}\n({bandit_algo})")
        final_regrets.append(data["mean_final_regret"])
        errors.append(
            1.96 * data["std_final_regret"] / np.sqrt(len(data["final_regret"]))
        )

    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, final_regrets, yerr=errors, capsize=5, alpha=0.7)

    # Color bars by strategy
    for i, (arm_strategy, bandit_algo) in enumerate(analysis.keys()):
        color = strategy_colors.get(arm_strategy, "gray")
        bars[i].set_color(color)

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Final Regret", fontsize=12)
    ax.set_title("Final Regret Comparison", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("final_regret_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "📊 Visualizations saved as 'causal_discovery_bandit_results.png' and 'final_regret_comparison.png'"
    )


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================


def main():
    """Run the complete integrated pipeline demonstration."""
    print("🚀 Starting Causal Discovery + Bandit Experiment Pipeline")
    print("   (Ground Truth → Data → MEC → POMIS Union → Bandit Tests)")

    # Step 1: Choose ground truth SCM (chain: Z -> X -> Y)
    print("\n📊 Setting up ground truth SCM...")
    ground_truth_scm, p_u_params = chain_SCM(devised=True, seed=42)
    print(f"   Ground truth SCM: {ground_truth_scm.G}")
    print(f"   Variables: {list(ground_truth_scm.G.V)}")
    print(f"   Manipulable variables: {ground_truth_scm.G.manipulable_vars}")
    print(f"   Parameters: {p_u_params}")

    # Run the integrated experiment
    results = run_discovery_bandit_experiment(
        ground_truth_scm=ground_truth_scm,
        Y="Y",
        n_samples=10000,
        alpha=0.01,
        num_trials=50,  # Reduced for demo
        horizon=2000,  # Reduced for demo
        n_jobs=2,
        save_plots=True,
    )

    # Print summary
    print("\n📈 EXPERIMENT SUMMARY")
    print("-" * 40)
    print(f"Ground truth SCM: {ground_truth_scm.G}")
    print(f"Discovered {len(results['discovery']['dags'])} DAGs in MEC")
    print(f"POMIS union size: {len(results['discovery']['pomis_union'])}")
    print(f"POMIS union: {results['discovery']['pomis_union']}")
    print(f"Bandit arms: {len(results['bandit']['expected_rewards'])}")
    print(f"Optimal reward: {np.max(results['bandit']['expected_rewards']):.4f}")

    print("\nFinal Regret Results:")
    for (strategy, algo), data in results["analysis"].items():
        print(
            f"  {strategy} ({algo}): {data['mean_final_regret']:.2f} ± {1.96 * data['std_final_regret'] / np.sqrt(len(data['final_regret'])):.2f}"
        )

    return results


if __name__ == "__main__":
    results = main()
