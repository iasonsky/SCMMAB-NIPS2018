#!/usr/bin/env python3
"""
Causal Discovery + Bandit Integration

This script implements the correct pipeline:
1. Choose one DAG as ground truth (IV_SCM_strong)
2. Simulate data from ground truth SCM
3. Learn MEC from data using PC algorithm
4. Use POMIS on all graphs in MEC and take union
5. Test bandit strategies using ground truth system
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from typing import Dict

# Import the existing causal discovery pipeline
from npsem.causal_pipeline import run_causal_discovery_pipeline_UC

# Import SCM examples and bandit framework
from npsem.NIPS2018POMIS_exp.scm_examples import IV_SCM_strong, IV_SCM
from npsem.bandits import play_bandits
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of
from npsem.utils import subseq
from npsem.data_simulation import simulate_data_from_scm

# Import modularized plotting functions
from npsem.plotting import create_causal_discovery_plots
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
    save_dir: str = "experiment_results",
) -> Dict:
    """
    Run the complete causal discovery + bandit experiment pipeline.

    This follows the correct pipeline:
    1. Choose one DAG as ground truth (IV_SCM_strong: Z -> X -> Y with confounding X <-> Y (U_XY))
    2. Simulate data from ground truth SCM
    3. Learn PAG from data using FCI algorithm
    4. Use POMIS on all ADMGs in PAG and take union
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
    print("   (Ground Truth → Data → PAG → POMIS Union → Bandit Tests)")
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
    print("\n2. Running FCI algorithm for causal discovery...")
    pag, admgs, pomis_union, mis_union = run_causal_discovery_pipeline_UC(
        data,
        var_names,
        ind_test="fisherz",
        alpha=alpha,
        sanity_check=True,
        ground_truth_scm=ground_truth_scm,
        save_dir=save_dir,
    )
    print("   Discovered PAG adjacency matrix:")
    print(f"   {pag}")
    print(f"   Found {len(admgs)} ADMGs consistent with PAG")
    print(f"   POMIS union: {pomis_union}")
    print(f"   MIS union: {mis_union}")

    # Step 3: Run bandit experiments using ground truth SCM
    print("\n5. Running bandit experiments using ground truth SCM...")
    mu, arm_setting = SCM_to_bandit_machine(ground_truth_scm, Y)
    print(f"   Created {len(mu)} bandit arms")
    print(f"   Expected rewards: {mu}")

    # Define strategies to compare
    # We'll test both ground truth and discovered POMIS/MIS
    bandit_algorithms = ["TS", "UCB"]
    results = {}

    # Helper function to get arms from discovered sets
    def cd_pomis_arms_of(arm_setting, pomis_union):
        """Get arms matching discovered POMIS union."""
        arms = []
        for arm_id, intervention in arm_setting.items():
            intervened_vars = tuple(sorted(intervention.keys()))
            if intervened_vars in pomis_union:
                arms.append(arm_id)
        return tuple(arms)

    def cd_mis_arms_of(arm_setting, mis_union):
        """Get arms matching discovered MIS union."""
        arms = []
        for arm_id, intervention in arm_setting.items():
            intervened_vars = tuple(sorted(intervention.keys()))
            if intervened_vars in mis_union:
                arms.append(arm_id)
        return tuple(arms)

    # Strategy configurations: (name, arm_selector_function, description)
    strategy_configs = [
        (
            "POMIS",
            lambda: arms_of("POMIS", arm_setting, ground_truth_scm.G, Y),
            "ground truth",
        ),
        (
            "MIS",
            lambda: arms_of("MIS", arm_setting, ground_truth_scm.G, Y),
            "ground truth",
        ),
        ("CD-POMIS", lambda: cd_pomis_arms_of(arm_setting, pomis_union), "discovered"),
        ("CD-MIS", lambda: cd_mis_arms_of(arm_setting, mis_union), "discovered"),
        (
            "Brute-force",
            lambda: arms_of("Brute-force", arm_setting, ground_truth_scm.G, Y),
            None,
        ),
        (
            "All-at-once",
            lambda: arms_of("All-at-once", arm_setting, ground_truth_scm.G, Y),
            None,
        ),
    ]

    for strategy_name, arm_selector, description in strategy_configs:
        try:
            arm_selected = arm_selector()
            desc_str = f" ({description})" if description else ""
            print(
                f"   {strategy_name}{desc_str}: Found {len(arm_selected)} arms out of {len(arm_setting)} total"
            )

            # Show which intervention sets are being used
            if strategy_name in ["POMIS", "CD-POMIS", "MIS", "CD-MIS"]:
                intervention_sets = [set(arm_setting[i].keys()) for i in arm_selected]
                unique_sets = sorted(
                    [tuple(sorted(s)) for s in set(map(frozenset, intervention_sets))]
                )
                print(f"      Intervention sets: {unique_sets}")
            if len(arm_selected) == 0:
                print(f"   Warning: No arms found for strategy {strategy_name}")
                continue

            arm_corrector = np.vectorize(lambda x: arm_selected[x])

            for bandit_algo in bandit_algorithms:
                print(f"   Running {strategy_name} + {bandit_algo}...")
                arm_played, rewards = play_bandits(
                    horizon, subseq(mu, arm_selected), bandit_algo, num_trials, n_jobs
                )
                results[(strategy_name, bandit_algo)] = (
                    arm_corrector(arm_played),
                    rewards,
                )

        except Exception as e:
            print(f"   Error with {strategy_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Step 6: Analysis and visualization
    print("\n6. Analyzing results...")
    analysis_results = analyze_bandit_results(results, mu, horizon)

    if save_plots:
        print("\n7. Creating visualizations...")
        create_visualizations(results, mu, horizon, analysis_results, save_dir=save_dir)

    # Compile final results
    final_results = {
        "ground_truth": {
            "scm": ground_truth_scm,
            "data": data,
            "var_names": var_names,
        },
        "discovery": {
            "pag": pag,
            "admgs": admgs,
            "pomis_union": pomis_union,
            "mis_union": mis_union,
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


def create_visualizations(
    results: Dict, mu: np.ndarray, horizon: int, analysis: Dict, save_dir: str
):
    """Create visualizations using modular plotting functions."""
    create_causal_discovery_plots(
        analysis=analysis,
        horizon=horizon,
        show_plots=True,
        save_dir=save_dir,
        use_confidence_intervals=False,  # Use standard deviation like original
        y_lim=None,  # No y-axis limit like original
        x_ticks=None,  # Use default x-axis ticks like original
        x_tick_labels=None,
    )


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================


def main():
    """Run the complete integrated pipeline demonstration."""
    print("🚀 Starting FCI Causal Discovery + Bandit Experiment Pipeline")
    print("   (Ground Truth → Data → PAG → ADMGs → POMIS Union → Bandit Tests)")

    # Step 1: Choose ground truth SCM (chain: Z -> X -> Y)
    print("\n📊 Setting up ground truth SCM...")
    ground_truth_scm, p_u_params = IV_SCM_strong(devised=True, seed=42)
    print(f"   Ground truth SCM: {ground_truth_scm.G}")
    print(f"   Variables: {list(ground_truth_scm.G.V)}")
    print(f"   Manipulable variables: {ground_truth_scm.G.manipulable_vars}")
    print(f"   Parameters: {p_u_params}")

    # Run the integrated experiment
    # Standard NIPS 2018 settings: num_trials=200, horizon=10000
    # For quick testing, use: num_trials=50, horizon=2000
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
    print(f"Discovered {len(results['discovery']['admgs'])} ADMGs consistent with PAG")
    print(f"POMIS union size: {len(results['discovery']['pomis_union'])}")
    print(f"POMIS union: {results['discovery']['pomis_union']}")
    print(f"MIS union size: {len(results['discovery']['mis_union'])}")
    print(f"MIS union: {results['discovery']['mis_union']}")
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
