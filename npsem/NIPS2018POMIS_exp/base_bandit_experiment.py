"""Base bandit experiment framework for SCMs.

This module provides a reusable framework for running bandit experiments
on different structural causal models without code duplication.
"""

import multiprocessing
import numpy as np
from typing import Dict, Tuple, Optional, Callable

from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of
from npsem.utils import subseq, mkdirs
from npsem.plotting import (
    compute_cumulative_regret,
    compute_confidence_intervals,
    create_experiment_summary_plot,
)


class BanditExperiment:
    """Base class for running bandit experiments on SCMs."""

    def __init__(
        self,
        scm_factory: Callable[[], Tuple[StructuralCausalModel, Dict[str, float]]],
        scm_name: str,
        target_variable: str = "Y",
        num_trials: int = 200,
        horizon: int = 10000,
        n_jobs: Optional[int] = None,
        arm_strategies: Optional[list] = None,
        bandit_algorithms: Optional[list] = None,
    ):
        """Initialize the bandit experiment.

        Parameters
        ----------
        scm_factory : callable
            Function that returns (model, p_u) tuple when called
        scm_name : str
            Name of the SCM for display and file naming
        target_variable : str, default "Y"
            Target variable for bandit experiments
        num_trials : int, default 200
            Number of simulation trials
        horizon : int, default 10000
            Time horizon for bandit experiments
        n_jobs : int, optional
            Number of parallel jobs. If None, uses half of CPU count
        arm_strategies : list, optional
            List of arm strategies to test. Default: ["POMIS", "MIS", "Brute-force"]
        bandit_algorithms : list, optional
            List of bandit algorithms to test. Default: ["TS", "UCB"]
        """
        self.scm_factory = scm_factory
        self.scm_name = scm_name
        self.target_variable = target_variable
        self.num_trials = num_trials
        self.horizon = horizon
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() // 2)
        self.arm_strategies = arm_strategies or ["POMIS", "MIS", "Brute-force"]
        self.bandit_algorithms = bandit_algorithms or ["TS", "UCB"]

        # Will be set during run
        self.model = None
        self.p_u = None
        self.mu = None
        self.arm_setting = None
        self.results = None

    def setup(self, seed: int = 42):
        """Setup the SCM and bandit machine."""
        print(f"📊 Creating {self.scm_name} SCM...")
        self.model, self.p_u = self.scm_factory(seed=seed)
        self.mu, self.arm_setting = SCM_to_bandit_machine(
            self.model, Y=self.target_variable
        )

    def run_experiments(self):
        """Run all bandit experiments."""
        if self.model is None:
            raise ValueError("Must call setup() before run_experiments()")

        print("🏃 Running bandit experiments...")
        self.results = {}

        for arm_strategy in self.arm_strategies:
            arm_selected = arms_of(
                arm_strategy, self.arm_setting, self.model.G, self.target_variable
            )
            arm_corrector = np.vectorize(lambda x: arm_selected[x])

            for bandit_algo in self.bandit_algorithms:
                arm_played, rewards = play_bandits(
                    self.horizon,
                    subseq(self.mu, arm_selected),
                    bandit_algo,
                    self.num_trials,
                    self.n_jobs,
                )
                self.results[(arm_strategy, bandit_algo)] = (
                    arm_corrector(arm_played),
                    rewards,
                )

    def compute_final_regret(self) -> Dict[str, np.ndarray]:
        """Compute final regret statistics at T=horizon."""
        if self.results is None:
            raise ValueError(
                "Must call run_experiments() before compute_final_regret()"
            )

        print("📊 Computing final regret...")
        final_regret_data = {}
        mu_star = np.max(self.mu)

        for (arm_strategy, bandit_algo), (arms, rewards) in self.results.items():
            cumulative_regret = compute_cumulative_regret(rewards, mu_star)
            final_regret = cumulative_regret[:, -1]  # Last column (T=horizon)

            # Compute confidence intervals
            mean_final_regret, lower, upper = compute_confidence_intervals(
                final_regret.reshape(-1, 1), confidence_level=0.95
            )
            std_final_regret = np.std(final_regret)
            ci_margin = upper[0] - mean_final_regret[0]

            key = f"{arm_strategy}_{bandit_algo}"
            final_regret_data[key] = np.array(
                [mean_final_regret[0], std_final_regret, ci_margin]
            )

            print(
                f"  {arm_strategy} ({bandit_algo}): {mean_final_regret[0]:.2f} ± {ci_margin:.2f}"
            )

        return final_regret_data

    def save_results(self, directory: str, final_regret_data: Dict[str, np.ndarray]):
        """Save experiment results to files."""
        mkdirs(directory)

        for arm_strategy, bandit_algo in self.results:
            arms, rewards = self.results[(arm_strategy, bandit_algo)]
            np.savez_compressed(
                f"{directory}/{arm_strategy}---{bandit_algo}", a=arms, b=rewards
            )

        np.savez_compressed(f"{directory}/p_u", a=self.p_u)
        np.savez_compressed(f"{directory}/mu", a=self.mu)
        np.savez_compressed(
            f"{directory}/final_regret_T{self.horizon}", **final_regret_data
        )

    def create_plots(
        self,
        directory: str,
        final_regret_data: Dict[str, np.ndarray],
        show_plots: bool = True,
        y_lim: Optional[float] = None,
    ):
        """Create and save experiment plots."""
        print("📊 Creating experiment summary plots...")
        create_experiment_summary_plot(
            results=self.results,
            mu=self.mu,
            final_regret_data=final_regret_data,
            horizon=self.horizon,
            save_dir=directory,
            show_plots=show_plots,
            y_lim=y_lim,
        )

    def print_summary(self):
        """Print experiment summary."""
        if self.model is None:
            print("❌ Experiment not yet run")
            return

        print(f"🎯 Target variable: {self.target_variable}")
        print(f"🔄 Number of trials: {self.num_trials}")
        print(f"⏱️  Horizon: {self.horizon}")
        print(f"💻 Parallel jobs: {self.n_jobs}")
        print(f"🎰 Available arms: {len(self.mu)}")
        print(f"📈 Arm expected rewards: {self.mu}")
        print(f"⭐ Optimal arm reward: {np.max(self.mu):.4f}")
        print()

    def run_full_experiment(
        self,
        seed: int = 42,
        save_dir: Optional[str] = None,
        show_plots: bool = True,
        y_lim: Optional[float] = None,
    ):
        """Run the complete experiment pipeline."""
        print(f"🚀 Starting {self.scm_name} SCM Bandit Experiments")
        print("=" * 60)

        # Setup
        self.setup(seed=seed)
        self.print_summary()

        # Run experiments
        self.run_experiments()

        # Compute final regret
        final_regret_data = self.compute_final_regret()
        print()

        # Save results
        if save_dir is None:
            save_dir = f"{self.scm_name.lower().replace(' ', '_')}_results"
        else:
            # Create SCM-specific subdirectory within the provided save_dir
            save_dir = f"{save_dir}/{self.scm_name.lower().replace(' ', '_')}_results"

        print(f"💾 Saving results to: {save_dir}/")
        self.save_results(save_dir, final_regret_data)

        # Create plots
        self.create_plots(save_dir, final_regret_data, show_plots, y_lim)

        print("✅ Experiment complete!")
        return self.results, self.mu, final_regret_data


def create_experiment(
    scm_factory: Callable[[], Tuple[StructuralCausalModel, Dict[str, float]]],
    scm_name: str,
    **kwargs,
) -> BanditExperiment:
    """Convenience function to create a bandit experiment."""
    return BanditExperiment(scm_factory, scm_name, **kwargs)
