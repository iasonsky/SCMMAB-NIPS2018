"""Run simulated bandit experiments on the frontdoor SCM.

This script runs Thompson Sampling and UCB experiments on the frontdoor
causal diagram and plots the average cumulative regret over time.

This is now a thin wrapper around the modular BanditExperiment framework.
"""

from npsem.NIPS2018POMIS_exp.base_bandit_experiment import BanditExperiment
from npsem.NIPS2018POMIS_exp.scm_examples import frontdoor_SCM


def main():
    """Main experiment function using modular framework."""
    # Create experiment using the modular framework
    experiment = BanditExperiment(
        scm_factory=frontdoor_SCM,
        scm_name="Frontdoor SCM",
        target_variable="Y",
        num_trials=200,
        horizon=10000,
    )
    
    # Run the complete experiment
    experiment.run_full_experiment(seed=42, save_dir="frontdoor_results")


if __name__ == "__main__":
    main()
