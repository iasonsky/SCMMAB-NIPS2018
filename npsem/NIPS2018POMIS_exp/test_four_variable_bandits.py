"""Run simulated bandit experiments on the four-variable SCM.

This script runs Thompson Sampling and UCB experiments on the four-variable
causal diagram and plots the average cumulative regret over time.
"""

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from npsem.NIPS2018POMIS_exp.scm_examples import four_variable_SCM
from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of
from npsem.utils import subseq, mkdirs


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


def compute_cumulative_regret(rewards: np.ndarray, mu_star: float) -> np.ndarray:
    """Compute cumulative regret for each trial."""
    cumulative_rewards = np.cumsum(rewards, axis=1)
    optimal_cumulative_rewards = np.cumsum(np.ones(rewards.shape) * mu_star, axis=1)
    cumulative_regret = optimal_cumulative_rewards - cumulative_rewards
    return cumulative_regret


def plot_cumulative_regret(
    results, mu, horizon=10000, save_path="four_variable_regret.png"
):
    """Plot average cumulative regret over time in the style of the original paper."""
    mu_star = np.max(mu)

    # Set up plotting style to match the paper
    import matplotlib as mpl
    mpl.rc("font", family="sans-serif")
    mpl.rc("font", serif="Helvetica")
    
    # Use specific colors as requested: Red for BF, Purple for MIS, Blue for POMIS
    strategy_colors = {
        "Brute-force": "red",
        "MIS": "purple", 
        "POMIS": "blue"
    }

    # Set up the plot with paper's figure size
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, ((arm_strategy, bandit_algo), (arms, rewards)) in enumerate(results.items()):
        cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        mean_regret = np.mean(cumulative_regret, axis=0)
        std_regret = np.std(cumulative_regret, axis=0)

        # Use 95% confidence intervals (like the paper)
        from scipy import stats
        n_trials = cumulative_regret.shape[0]
        confidence_level = 0.95
        t_value = stats.t.ppf((1 + confidence_level) / 2, n_trials - 1)
        margin_error = t_value * std_regret / np.sqrt(n_trials)
        lower, upper = mean_regret - margin_error, mean_regret + margin_error
        
        color = strategy_colors.get(arm_strategy, "gray")
        linestyle = "-" if bandit_algo == "TS" else "--"

        # Sparse time points for cleaner visualization (like paper)
        from npsem.viz_util import sparse_index
        time_points = sparse_index(horizon, 200)

        # Plot with paper's styling
        ax.plot(
            time_points,
            mean_regret[time_points],
            lw=1,
            label=arm_strategy.split(" ")[0] if "(TS)" in f"{arm_strategy} ({bandit_algo})" else None,
            color=color,
            linestyle=linestyle,
        )
        
        # Fill between with 95% confidence intervals
        ax.fill_between(
            time_points,
            lower[time_points],
            upper[time_points],
            color=color,
            alpha=0.1,  # Paper uses band_alpha=0.1
            lw=0,
        )

    # Paper-style formatting
    ax.set_xlabel("Trials")
    ax.set_ylabel("Cum. Regrets")
    ax.legend(loc=2, frameon=False)  # Paper uses loc=2, frameon=False
    
    # Set y-axis limit to 200 and x-axis ticks to 0, 5k, 10k
    ax.set_ylim(0, 200)
    ax.set_xticks([0, 5000, 10000])
    ax.set_xticklabels(['0', '5k', '10k'])
    
    # Remove spines and add paper-style formatting
    sns.despine()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"📊 Plot saved as: {save_path}")


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
        mean_final_regret = np.mean(final_regret)
        std_final_regret = np.std(final_regret)

        # Calculate 95% confidence interval for final regret
        n_trials = final_regret.shape[0]
        se_final_regret = std_final_regret / np.sqrt(n_trials)
        t_critical = 1.96
        ci_margin = t_critical * se_final_regret

        key = f"{arm_strategy}_{bandit_algo}"
        final_regret_data[key] = np.array(
            [mean_final_regret, std_final_regret, ci_margin]
        )

        print(
            f"  {arm_strategy} ({bandit_algo}): {mean_final_regret:.2f} ± {ci_margin:.2f}"
        )

    print()

    # Save results
    directory = "four_variable_results"
    print(f"💾 Saving results to: {directory}/")
    save_results(directory, p_u, mu, results, final_regret_data)

    # Create and save plot
    print("📊 Creating cumulative regret plot...")
    plot_cumulative_regret(results, mu, horizon, f"{directory}/cumulative_regret.png")

    print("✅ Experiment complete!")


if __name__ == "__main__":
    main()
