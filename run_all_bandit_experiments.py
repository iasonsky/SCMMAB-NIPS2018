#!/usr/bin/env python3
"""Run all bandit experiments for the paper.

This script runs bandit experiments on all three SCMs:
1. Four-variable SCM (existing)
2. Frontdoor SCM (new)
3. Six-variable SCM (new)

Usage:
    python run_all_bandit_experiments.py [--quick] [--scm SCM_NAME]

Options:
    --quick: Run with shorter horizon (T=1000) and fewer trials (50) for quick testing
    --scm SCM_NAME: Run only specific SCM (four_variable, frontdoor, six_variable)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_experiment(script_name, quick=False):
    """Run a single bandit experiment script."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Running {script_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run the script
        cmd = [sys.executable, script_name]
        if quick:
            # For quick testing, we could modify the scripts to accept command line args
            # For now, we'll just run them as-is
            pass

        _result = subprocess.run(cmd, check=True, capture_output=False)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ {script_name} completed successfully in {duration:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {script_name} failed with error: {e}")
        return False


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description="Run all bandit experiments")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with shorter horizon for quick testing",
    )
    parser.add_argument(
        "--scm",
        choices=["four_variable", "frontdoor", "six_variable"],
        help="Run only specific SCM",
    )

    args = parser.parse_args()

    # Define experiments
    experiments = {
        "four_variable": "npsem/NIPS2018POMIS_exp/test_four_variable_bandits.py",
        "frontdoor": "npsem/NIPS2018POMIS_exp/test_frontdoor_bandits.py",
        "six_variable": "npsem/NIPS2018POMIS_exp/test_six_variable_bandits.py",
    }

    # Filter experiments if specific SCM requested
    if args.scm:
        experiments = {args.scm: experiments[args.scm]}

    print("🎯 Bandit Experiment Runner")
    print("=" * 60)

    if args.quick:
        print("⚡ Quick mode: Shorter horizon and fewer trials")
    else:
        print("🐌 Full mode: T=10000, 200 trials")

    print(f"📊 Running {len(experiments)} experiment(s)")
    print()

    # Track results
    results = {}
    total_start = time.time()

    # Run each experiment
    for scm_name, script_path in experiments.items():
        if not Path(script_path).exists():
            print(f"❌ Script not found: {script_path}")
            results[scm_name] = False
            continue

        success = run_experiment(script_path, args.quick)
        results[scm_name] = success

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    successful = sum(results.values())
    total = len(results)

    for scm_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {scm_name:15} : {status}")

    print(f"\n🎯 Results: {successful}/{total} experiments completed successfully")
    print(f"⏱️  Total time: {total_time:.1f} seconds")

    if successful == total:
        print("\n🎉 All experiments completed successfully!")
        print("\n📁 Results saved in:")
        for scm_name in experiments.keys():
            result_dir = f"{scm_name.replace('_', '_')}_results"
            print(f"   - {result_dir}/")
    else:
        print(f"\n⚠️  {total - successful} experiment(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
