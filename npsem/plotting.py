"""Plotting utilities for bandit experiments and causal discovery results.

This module provides standardized plotting functions for visualizing
bandit experiment results, cumulative regret curves, and other
experimental outcomes in the style of the original papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, Optional


def setup_paper_style():
    """Set up matplotlib style to match academic papers."""
    mpl.rc("font", family="sans-serif")
    mpl.rc("font", serif="Helvetica")


def compute_cumulative_regret(rewards: np.ndarray, mu_star: float) -> np.ndarray:
    """Compute cumulative regret for each trial.

    Args:
        rewards: Array of shape (n_trials, horizon) containing rewards
        mu_star: Optimal expected reward per trial

    Returns:
        Array of shape (n_trials, horizon) containing cumulative regret
    """
    cumulative_rewards = np.cumsum(rewards, axis=1)
    optimal_cumulative_rewards = np.cumsum(np.ones(rewards.shape) * mu_star, axis=1)
    cumulative_regret = optimal_cumulative_rewards - cumulative_rewards
    return cumulative_regret


def compute_confidence_intervals(
    data: np.ndarray, confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute confidence intervals for data across trials.

    Args:
        data: Array of shape (n_trials, horizon) or (n_trials,)
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    n_trials = data.shape[0]
    t_value = stats.t.ppf((1 + confidence_level) / 2, n_trials - 1)
    margin_error = t_value * std_data / np.sqrt(n_trials)

    lower = mean_data - margin_error
    upper = mean_data + margin_error

    return mean_data, lower, upper


def plot_cumulative_regret(
    results: Dict,
    mu: np.ndarray,
    horizon: int = 10000,
    save_path: str = "cumulative_regret.png",
    strategy_colors: Optional[Dict[str, str]] = None,
    y_lim: Tuple[float, float] = (0, 200),
    x_ticks: Tuple[int, int, int] = (0, 5000, 10000),
    x_tick_labels: Tuple[str, str, str] = ("0", "5k", "10k"),
    figure_size: Tuple[float, float] = (8, 5),
    show_plot: bool = True,
    y_max: Optional[float] = None,
) -> None:
    """Plot average cumulative regret over time in academic paper style.

    Args:
        results: Dictionary mapping (strategy, algorithm) to (arms, rewards)
        mu: Array of expected rewards for each arm
        horizon: Number of time steps in the experiment
        save_path: Path to save the plot
        strategy_colors: Dict mapping strategy names to colors
        y_lim: Y-axis limits (min, max)
        x_ticks: X-axis tick positions
        x_tick_labels: X-axis tick labels
        figure_size: Figure size (width, height)
        show_plot: Whether to display the plot
    """
    mu_star = np.max(mu)

    # Set up plotting style
    setup_paper_style()

    # Default color scheme: Red for BF, Purple for MIS, Blue for POMIS
    if strategy_colors is None:
        strategy_colors = {"Brute-force": "red", "MIS": "purple", "POMIS": "blue"}

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    for (arm_strategy, bandit_algo), (arms, rewards) in results.items():
        cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        mean_regret, lower, upper = compute_confidence_intervals(cumulative_regret)

        color = strategy_colors.get(arm_strategy, "gray")
        linestyle = "-" if bandit_algo == "TS" else "--"

        # Sparse time points for cleaner visualization
        from npsem.viz_util import sparse_index

        time_points = sparse_index(horizon, 200)

        # Plot with paper's styling
        ax.plot(
            time_points,
            mean_regret[time_points],
            lw=1,
            label=arm_strategy.split(" ")[0]
            if "(TS)" in f"{arm_strategy} ({bandit_algo})"
            else None,
            color=color,
            linestyle=linestyle,
        )

        # Fill between with confidence intervals
        ax.fill_between(
            time_points,
            lower[time_points],
            upper[time_points],
            color=color,
            alpha=0.1,
            lw=0,
        )

    # Paper-style formatting
    ax.set_xlabel("Trials")
    ax.set_ylabel("Cum. Regrets")
    ax.legend(loc=2, frameon=False)

    # Set axis limits and ticks
    if y_max is not None:
        ax.set_ylim((0, y_max))
    else:
        ax.set_ylim(y_lim)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    # Remove spines and add paper-style formatting
    sns.despine()

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"📊 Plot saved as: {save_path}")


def plot_final_regret_comparison(
    final_regret_data: Dict,
    save_path: str = "final_regret_comparison.png",
    figure_size: Tuple[float, float] = (10, 6),
    show_plot: bool = True,
) -> None:
    """Plot final regret comparison across strategies.

    Args:
        final_regret_data: Dictionary containing final regret statistics
        save_path: Path to save the plot
        figure_size: Figure size (width, height)
        show_plot: Whether to display the plot
    """
    setup_paper_style()

    # Extract data for plotting
    strategies = []
    means = []
    errors = []
    colors = []

    strategy_color_map = {"Brute-force": "red", "MIS": "purple", "POMIS": "blue"}

    for key, data in final_regret_data.items():
        strategy = key.split("_")[0]
        algorithm = key.split("_")[1]

        mean_val, std_val, ci_margin = data

        strategies.append(f"{strategy}\n({algorithm})")
        means.append(mean_val)
        errors.append(ci_margin)  # Use confidence interval margin as error
        colors.append(strategy_color_map.get(strategy, "gray"))

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color=colors, alpha=0.7)

    # Customize the plot
    ax.set_xlabel("Strategy (Algorithm)")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_title("Final Regret at T=10,000")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha="right")

    # Add value labels on bars
    for bar, mean_val, error in zip(bars, means, errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + error + 2,
            f"{mean_val:.1f}±{error:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Remove spines
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"📊 Final regret comparison saved as: {save_path}")


def create_experiment_summary_plot(
    results: Dict,
    mu: np.ndarray,
    final_regret_data: Dict,
    horizon: int = 10000,
    save_dir: str = "results",
    show_plots: bool = True,
    y_lim: Optional[float] = None,
) -> None:
    """Create comprehensive experiment summary plots.

    Args:
        results: Dictionary mapping (strategy, algorithm) to (arms, rewards)
        mu: Array of expected rewards for each arm
        final_regret_data: Dictionary containing final regret statistics
        horizon: Number of time steps in the experiment
        save_dir: Directory to save plots
        show_plots: Whether to display the plots
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Create cumulative regret plot
    cumulative_regret_path = os.path.join(save_dir, "cumulative_regret.png")
    plot_cumulative_regret(
        results=results,
        mu=mu,
        horizon=horizon,
        save_path=cumulative_regret_path,
        show_plot=show_plots,
        y_max=y_lim,
    )

    # Create final regret comparison plot
    final_regret_path = os.path.join(save_dir, "final_regret_comparison.png")
    plot_final_regret_comparison(
        final_regret_data=final_regret_data,
        save_path=final_regret_path,
        show_plot=show_plots,
    )

    print(f"📊 All plots saved in: {save_dir}/")


def plot_cumulative_regret_from_analysis(
    analysis: Dict,
    horizon: int = 5000,
    save_path: str = "causal_discovery_bandit_results.png",
    show_plot: bool = True,
    use_confidence_intervals: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    x_ticks: Optional[Tuple[int, int, int]] = None,
    x_tick_labels: Optional[Tuple[str, str, str]] = None,
) -> None:
    """Plot cumulative regret from analysis data (for causal discovery integration).

    This function is designed to work with the analysis format from
    causal_discovery_bandit_integration.py, which already has cumulative regret computed.

    Args:
        analysis: Dictionary containing analysis results with 'cumulative_regret' key
        horizon: Time horizon for the experiment
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        use_confidence_intervals: Whether to use 95% CI instead of std dev
        y_lim: Y-axis limits (min, max)
        x_ticks: X-axis tick positions
        x_tick_labels: X-axis tick labels
    """
    setup_paper_style()

    # Use the same color palette as the paper
    c__ = sns.color_palette("Set1", 4)
    COLORS = [c__[0], c__[0], c__[1], c__[1], c__[2], c__[2], c__[3], c__[3]]

    # Map strategies to colors (matching paper's approach)
    strategy_algo_pairs = [(strategy, algo) for (strategy, algo) in analysis.keys()]
    strategy_colors = {}
    for i, (arm_strategy, bandit_algo) in enumerate(strategy_algo_pairs):
        strategy_colors[(arm_strategy, bandit_algo)] = COLORS[i]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for (arm_strategy, bandit_algo), data in analysis.items():
        cumulative_regret = data["cumulative_regret"]
        mean_regret = np.mean(cumulative_regret, axis=0)
        std_regret = np.std(cumulative_regret, axis=0)

        if use_confidence_intervals:
            # Use 95% confidence intervals
            mean_val, lower, upper = compute_confidence_intervals(cumulative_regret)
        else:
            # Use standard deviation for bands (like the paper)
            lower, upper = mean_regret - std_regret, mean_regret + std_regret

        color = strategy_colors.get((arm_strategy, bandit_algo), "gray")
        linestyle = "-" if bandit_algo == "TS" else "--"

        # Sparse time points for cleaner visualization (like paper)
        from npsem.viz_util import sparse_index

        time_points = sparse_index(horizon, 200)

        # Plot with paper's styling
        ax.plot(
            time_points,
            mean_regret[time_points],
            lw=1,
            label=arm_strategy.split(" ")[0]
            if "(TS)" in f"{arm_strategy} ({bandit_algo})"
            else None,
            color=color,
            linestyle=linestyle,
        )

        # Fill between with paper's alpha
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

    # Apply custom axis settings if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None and x_tick_labels is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    # Remove spines and add paper-style formatting
    sns.despine()

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"📊 Plot saved as: {save_path}")


def plot_final_regret_comparison_from_analysis(
    analysis: Dict,
    save_path: str = "final_regret_comparison.png",
    show_plot: bool = True,
    figure_size: Tuple[float, float] = (10, 6),
) -> None:
    """Plot final regret comparison from analysis data (for causal discovery integration).

    Args:
        analysis: Dictionary containing analysis results with regret statistics
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        figure_size: Size of the figure
    """
    setup_paper_style()

    # Use the same color palette as the paper
    c__ = sns.color_palette("Set1", 4)
    COLORS = [c__[0], c__[0], c__[1], c__[1], c__[2], c__[2], c__[3], c__[3]]

    # Map strategies to colors (matching paper's approach)
    strategy_algo_pairs = [(strategy, algo) for (strategy, algo) in analysis.keys()]
    strategy_colors = {}
    for i, (arm_strategy, bandit_algo) in enumerate(strategy_algo_pairs):
        strategy_colors[(arm_strategy, bandit_algo)] = COLORS[i]

    # Extract data for plotting
    strategies = []
    final_regrets = []
    errors = []

    for (arm_strategy, bandit_algo), data in analysis.items():
        strategies.append(f"{arm_strategy}\n({bandit_algo})")
        final_regrets.append(data["mean_final_regret"])
        errors.append(
            1.96 * data["std_final_regret"] / np.sqrt(len(data["final_regret"]))
        )

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, final_regrets, yerr=errors, capsize=5, alpha=0.7)

    # Color bars by strategy
    for i, (arm_strategy, bandit_algo) in enumerate(analysis.keys()):
        color = strategy_colors.get((arm_strategy, bandit_algo), "gray")
        bars[i].set_color(color)

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Final Regret", fontsize=12)
    ax.set_title("Final Regret Comparison", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies)
    ax.grid(True, alpha=0.3)

    # Remove spines
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"📊 Final regret comparison saved as: {save_path}")


def create_causal_discovery_plots(
    analysis: Dict,
    horizon: int = 5000,
    save_dir: str = "experiment_results",
    show_plots: bool = True,
    use_confidence_intervals: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    x_ticks: Optional[Tuple[int, int, int]] = None,
    x_tick_labels: Optional[Tuple[str, str, str]] = None,
) -> None:
    """Create comprehensive plots for causal discovery + bandit experiments.

    Args:
        analysis: Dictionary containing analysis results
        horizon: Time horizon for the experiment
        save_dir: Directory to save plots
        show_plots: Whether to display plots
        use_confidence_intervals: Whether to use 95% CI instead of std dev
        y_lim: Y-axis limits (min, max)
        x_ticks: X-axis tick positions
        x_tick_labels: X-axis tick labels
    """
    import os

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create cumulative regret plot
    cumulative_regret_path = os.path.join(
        save_dir, "causal_discovery_bandit_results.png"
    )
    plot_cumulative_regret_from_analysis(
        analysis=analysis,
        horizon=horizon,
        save_path=cumulative_regret_path,
        show_plot=show_plots,
        use_confidence_intervals=use_confidence_intervals,
        y_lim=y_lim,
        x_ticks=x_ticks,
        x_tick_labels=x_tick_labels,
    )

    # Create final regret comparison plot
    final_regret_path = os.path.join(save_dir, "final_regret_comparison.png")
    plot_final_regret_comparison_from_analysis(
        analysis=analysis,
        save_path=final_regret_path,
        show_plot=show_plots,
    )

    print(
        f"📊 Visualizations saved in '{save_dir}/' as 'causal_discovery_bandit_results.png' and 'final_regret_comparison.png'"
    )
