#!/usr/bin/env python3
"""Unified bandit experiment runner for all SCMs.

This script provides a modular way to run bandit experiments on any
registered SCM without code duplication.

Usage:
    python run_bandit_experiments.py [--scm SCM_NAME] [--quick] [--list]
    
Options:
    --scm SCM_NAME: Run specific SCM (use --list to see available)
    --quick: Run with shorter horizon (T=1000) and fewer trials (50)
    --list: List all available SCMs
    --all: Run all SCMs
"""

import argparse
import sys
from typing import List, Optional

from npsem.NIPS2018POMIS_exp.base_bandit_experiment import BanditExperiment
from npsem.NIPS2018POMIS_exp.scm_registry import registry, get_scm_factory, get_scm_config


def run_single_experiment(scm_name: str, quick: bool = False, show_plots: bool = True) -> bool:
    """Run experiment for a single SCM."""
    try:
        # Get SCM configuration
        scm_info = registry.get_scm(scm_name)
        scm_factory = scm_info['factory']
        config = scm_info['config'].copy()
        
        # Override config for quick mode
        if quick:
            config['num_trials'] = 50
            config['horizon'] = 1000
            
        # Extract y_lim if present
        y_lim = config.pop('y_lim', None)
            
        # Create and run experiment
        experiment = BanditExperiment(
            scm_factory=scm_factory,
            scm_name=scm_info['display_name'],
            **config
        )
        
        experiment.run_full_experiment(save_dir="experiment_results", show_plots=show_plots, y_lim=y_lim)
        return True
        
    except Exception as e:
        print(f"❌ Failed to run {scm_name}: {e}")
        return False


def run_all_experiments(scm_names: List[str], quick: bool = False, show_plots: bool = True) -> dict:
    """Run experiments for multiple SCMs."""
    results = {}
    
    for scm_name in scm_names:
        print(f"\n{'='*60}")
        print(f"🚀 Running {scm_name.upper()} experiment")
        print(f"{'='*60}")
        
        success = run_single_experiment(scm_name, quick, show_plots)
        results[scm_name] = success
        
    return results


def print_summary(results: dict):
    """Print experiment summary."""
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(results.values())
    total = len(results)
    
    for scm_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {scm_name:20} : {status}")
    
    print(f"\n🎯 Results: {successful}/{total} experiments completed successfully")
    
    if successful == total:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {total - successful} experiment(s) failed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run bandit experiments on SCMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bandit_experiments.py --list
  python run_bandit_experiments.py --scm frontdoor
  python run_bandit_experiments.py --scm frontdoor four_variable six_variable
  python run_bandit_experiments.py --scm frontdoor --quick
  python run_bandit_experiments.py --all
  python run_bandit_experiments.py --all --quick
        """
    )
    
    parser.add_argument("--scm", type=str, nargs='+', help="Run specific SCM(s). Use space-separated list for multiple: --scm frontdoor four_variable six_variable")
    parser.add_argument("--all", action="store_true", help="Run all SCMs")
    parser.add_argument("--quick", action="store_true", help="Quick mode (T=1000, 50 trials)")
    parser.add_argument("--list", action="store_true", help="List available SCMs")
    parser.add_argument("--no-plots", action="store_true", help="Don't show plots")
    
    args = parser.parse_args()
    
    # List available SCMs
    if args.list:
        print("📋 Available SCMs:")
        print("=" * 40)
        scms = registry.list_scms()
        for name, description in scms.items():
            print(f"  {name:20} : {description}")
        return
    
    # Determine which SCMs to run
    if args.scm:
        # args.scm is already a list of SCM names
        scm_names = args.scm
    elif args.all:
        scm_names = registry.get_all_scm_names()
    else:
        print("❌ Must specify --scm SCM_NAME, --all, or --list")
        print("Use --list to see available SCMs")
        print("For multiple SCMs: --scm frontdoor four_variable six_variable")
        sys.exit(1)
    
    # Validate SCM names
    available_scms = registry.get_all_scm_names()
    invalid_scms = [name for name in scm_names if name not in available_scms]
    if invalid_scms:
        print(f"❌ Invalid SCM names: {invalid_scms}")
        print(f"Available SCMs: {available_scms}")
        sys.exit(1)
    
    # Run experiments
    print("🎯 Bandit Experiment Runner")
    print("=" * 60)
    
    if args.quick:
        print("⚡ Quick mode: T=1000, 50 trials")
    else:
        print("🐌 Full mode: T=10000, 200 trials")
    
    print(f"📊 Running {len(scm_names)} experiment(s)")
    print()
    
    # Run experiments
    results = run_all_experiments(
        scm_names, 
        quick=args.quick, 
        show_plots=not args.no_plots
    )
    
    # Print summary
    print_summary(results)
    
    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
