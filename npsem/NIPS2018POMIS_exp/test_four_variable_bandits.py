"""Run simulated bandit experiments on the four-variable SCM.

This script runs Thompson Sampling and UCB experiments on the four-variable
causal diagram and plots the average cumulative regret over time.
"""

import multiprocessing
import numpy as np

from npsem.NIPS2018POMIS_exp.scm_examples import four_variable_SCM
from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of
from npsem.utils import subseq, mkdirs
from npsem.plotting import (
    compute_cumulative_regret,
    compute_confidence_intervals,
    create_experiment_summary_plot,
)


def main_experiment(
    M: StructuralCausalModel, Y="Y", num_trial=200, horizon=10000, n_jobs=1
):
    """Run bandit experiments for all arm strategies and algorithms."""
    results = dict()
    mu, arm_setting = SCM_to_bandit_machine(M)
    arm_types = ["POMIS", "MIS", "Brute-force"]  # , "All-at-once"]
    for arm_strategy in arm_types:
        arm_selected = arms_of(arm_strategy, arm_setting, M.G, Y)
        arm_corrector = np.vectorize(lambda x: arm_selected[x])
        for bandit_algo in ["TS", "UCB"]:
            arm_played, rewards = play_bandits(
                horizon, subseq(mu, arm_selected), bandit_algo, num_trial, n_jobs
            )
            results[(arm_strategy, bandit_algo)] = arm_corrector(arm_played), rewards

    return results, mu


def save_results(directory, p_u, mu, results, final_regret_data):
    """Save experiment results to files."""
    mkdirs(directory)

    for arm_strategy, bandit_algo in results:
        arms, rewards = results[(arm_strategy, bandit_algo)]
        np.savez_compressed(
            directory + f"/{arm_strategy}---{bandit_algo}", a=arms, b=rewards
        )

    np.savez_compressed(directory + "/p_u", a=p_u)
    np.savez_compressed(directory + "/mu", a=mu)

    # Save final regret data
    np.savez_compressed(directory + "/final_regret_T10000", **final_regret_data)


def main():
    """Main experiment function."""
    print("🚀 Starting Four Variable SCM Bandit Experiments")
    print("=" * 60)

    # Experiment parameters
    num_simulation_repeats = 200
    horizon = 10000
    n_jobs = max(1, multiprocessing.cpu_count() // 2)

    # Create the four-variable SCM
    print("📊 Creating four-variable SCM...")
    model, p_u = four_variable_SCM(seed=42)

    print("🎯 Target variable: Y")
    print(f"🔄 Number of trials: {num_simulation_repeats}")
    print(f"⏱️  Horizon: {horizon}")
    print(f"💻 Parallel jobs: {n_jobs}")
    print()

    # Run experiments
    print("🏃 Running bandit experiments...")
    results, mu = main_experiment(model, "Y", num_simulation_repeats, horizon, n_jobs)

    # Print arm information
    print(f"🎰 Available arms: {len(mu)}")
    print(f"📈 Arm expected rewards: {mu}")
    print(f"⭐ Optimal arm reward: {np.max(mu):.4f}")
    print()

    # Compute final regret at T=10000
    print("📊 Computing final regret at T=10000...")
    final_regret_data = {}
    mu_star = np.max(mu)

    for (arm_strategy, bandit_algo), (arms, rewards) in results.items():
        cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        final_regret = cumulative_regret[:, -1]  # Last column (T=10000)

        # Use modularized confidence interval computation
        mean_final_regret, lower, upper = compute_confidence_intervals(
            final_regret.reshape(-1, 1), confidence_level=0.95
        )
        std_final_regret = np.std(final_regret)
        ci_margin = upper[0] - mean_final_regret[0]  # Margin of error

        key = f"{arm_strategy}_{bandit_algo}"
        final_regret_data[key] = np.array(
            [mean_final_regret[0], std_final_regret, ci_margin]
        )

        print(
            f"  {arm_strategy} ({bandit_algo}): {mean_final_regret[0]:.2f} ± {ci_margin:.2f}"
        )

    print()

    # Save results
    directory = "four_variable_results"
    print(f"💾 Saving results to: {directory}/")
    save_results(directory, p_u, mu, results, final_regret_data)

    # Create and save plots using modularized plotting functions
    print("📊 Creating experiment summary plots...")
    create_experiment_summary_plot(
        results=results,
        mu=mu,
        final_regret_data=final_regret_data,
        horizon=horizon,
        save_dir=directory,
        show_plots=True,
    )

    print("✅ Experiment complete!")


if __name__ == "__main__":
    main()
