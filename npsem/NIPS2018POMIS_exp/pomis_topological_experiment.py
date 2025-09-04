"""Experiment to study the effect of topological ordering on POMIS computation.

This script runs the POMIS algorithm with different topological orderings and
collects metrics about performance, recursion depth, and result consistency.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from npsem.model import CausalDiagram
from npsem.pomis_instrumentation import (
    InstrumentedPOMISComputer,
    generate_topological_orders_strategies,
)
from npsem.NIPS2018POMIS_exp.scm_examples import XYZWST, XYZW, IV_CD, simple_markovian


def load_test_graphs() -> Dict[str, tuple]:
    """Load various test graphs for the experiment."""
    graphs = {}

    # Add the XYZWST example
    try:
        G = XYZWST()
        Y = G.causal_order(backward=True)[0]
        G_filtered = G[G.An(Y)]
        graphs["XYZWST"] = (G_filtered, Y)
    except Exception as e:
        print(f"Warning: Could not load XYZWST: {e}")

    # Add XYZW example (simpler version)
    try:
        G = XYZW()
        Y = G.causal_order(backward=True)[0]
        G_filtered = G[G.An(Y)]
        graphs["XYZW"] = (G_filtered, Y)
    except Exception as e:
        print(f"Warning: Could not load XYZW: {e}")

    # Add IV example
    try:
        G = IV_CD()
        Y = G.causal_order(backward=True)[0]
        G_filtered = G[G.An(Y)]
        graphs["IV"] = (G_filtered, Y)
    except Exception as e:
        print(f"Warning: Could not load IV: {e}")

    # Add simple markovian example
    try:
        G = simple_markovian()
        Y = G.causal_order(backward=True)[0]
        G_filtered = G[G.An(Y)]
        graphs["simple_markovian"] = (G_filtered, Y)
    except Exception as e:
        print(f"Warning: Could not load simple_markovian: {e}")

    return graphs


def run_single_experiment(
    graph_name: str,
    G: CausalDiagram,
    Y: str,
    output_dir: Path,
    num_random_orders: int = 10,
) -> List[Dict[str, Any]]:
    """Run POMIS computation with different topological orders for a single graph."""
    computer = InstrumentedPOMISComputer()
    results = []

    print(f"Running experiment on graph '{graph_name}'...")
    print(f"  Graph has {len(G.V)} vertices and {len(G.edges)} edges")

    # Get different topological ordering strategies
    strategies = generate_topological_orders_strategies(G)
    print(f"  Generated {len(strategies)} strategic orderings")

    # Add some random orders
    try:
        all_orders = computer.get_all_topological_orders(G)
        print(f"  Found {len(all_orders)} possible topological orderings")

        # Sample random orders if we have more than requested
        if len(all_orders) > num_random_orders:
            random.shuffle(all_orders)
            selected_orders = all_orders[:num_random_orders]
        else:
            selected_orders = all_orders

        for i, order in enumerate(selected_orders):
            strategies[f"random_{i:02d}"] = order
    except Exception as e:
        print(f"  Warning: Could not generate all orders: {e}")

    print(f"  Total orderings to test: {len(strategies)}")

    # Reference result for verification (using default order)
    reference_result = None

    # Run experiments with each ordering
    for strategy_name, order in strategies.items():
        print(f"  Testing strategy: {strategy_name}")

        try:
            result_set = computer.compute_pomis_with_order(G, Y, order)
            metrics = computer.metrics.to_dict()

            # Store reference result
            if strategy_name == "default":
                reference_result = result_set

            # Check if result matches reference
            result_matches = reference_result is None or frozenset(
                result_set
            ) == frozenset(reference_result)

            result_dict = {
                "graph_name": graph_name,
                "strategy": strategy_name,
                "order": str(order),
                "result_matches_reference": result_matches,
                **metrics,
            }

            results.append(result_dict)

            if not result_matches:
                print(
                    f"    WARNING: Strategy {strategy_name} produced different POMIS set!"
                )

        except Exception as e:
            print(f"    ERROR in strategy {strategy_name}: {e}")
            continue

    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "pomis_topological_results.csv"

    # Write CSV
    fieldnames = results[0].keys()
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_file}")

    # Also save as JSON for easier programmatic access
    json_file = output_dir / "pomis_topological_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results also saved to: {json_file}")


def create_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Create plots summarizing the results."""
    if not results:
        print("No results to plot.")
        return

    import pandas as pd

    df = pd.DataFrame(results)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("POMIS Topological Ordering Effects", fontsize=16)

    # 1. Distribution of recursive calls
    ax1 = axes[0, 0]
    for graph in df["graph_name"].unique():
        graph_data = df[df["graph_name"] == graph]
        ax1.hist(graph_data["total_subpomis_calls"], alpha=0.7, label=graph, bins=15)
    ax1.set_xlabel("Total SubPOMIS Calls")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Recursive Calls")
    ax1.legend()

    # 2. Distribution of computation time
    ax2 = axes[0, 1]
    for graph in df["graph_name"].unique():
        graph_data = df[df["graph_name"] == graph]
        ax2.hist(graph_data["computation_time"], alpha=0.7, label=graph, bins=15)
    ax2.set_xlabel("Computation Time (seconds)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Computation Time")
    ax2.legend()

    # 3. IB evaluations vs recursive calls
    ax3 = axes[1, 0]
    for graph in df["graph_name"].unique():
        graph_data = df[df["graph_name"] == graph]
        ax3.scatter(
            graph_data["total_subpomis_calls"],
            graph_data["total_ib_evaluations"],
            alpha=0.7,
            label=graph,
        )
    ax3.set_xlabel("Total SubPOMIS Calls")
    ax3.set_ylabel("Total IB Evaluations")
    ax3.set_title("IB Evaluations vs Recursive Calls")
    ax3.legend()

    # 4. Max recursion depth distribution
    ax4 = axes[1, 1]
    for graph in df["graph_name"].unique():
        graph_data = df[df["graph_name"] == graph]
        ax4.hist(graph_data["max_recursion_depth"], alpha=0.7, label=graph, bins=10)
    ax4.set_xlabel("Max Recursion Depth")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Max Recursion Depth")
    ax4.legend()

    plt.tight_layout()

    # Save the plot
    plot_file = output_dir / "pomis_topological_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pomis_topological_distributions.pdf", bbox_inches="tight")
    print(f"Plots saved to: {plot_file}")

    plt.show()

    # Create summary statistics table
    summary_stats = []
    for graph in df["graph_name"].unique():
        graph_data = df[df["graph_name"] == graph]
        for metric in [
            "total_subpomis_calls",
            "total_ib_evaluations",
            "total_pruned_branches",
            "computation_time",
            "max_recursion_depth",
        ]:
            stats = {
                "graph": graph,
                "metric": metric,
                "min": graph_data[metric].min(),
                "median": graph_data[metric].median(),
                "max": graph_data[metric].max(),
                "mean": graph_data[metric].mean(),
                "std": graph_data[metric].std(),
            }
            summary_stats.append(stats)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_file = output_dir / "pomis_topological_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")


def verify_result_consistency(results: List[Dict[str, Any]]) -> bool:
    """Verify that all topological orders produce the same POMIS sets."""
    consistent = True

    print("\nVerifying result consistency...")

    for graph_name in set(r["graph_name"] for r in results):
        graph_results = [r for r in results if r["graph_name"] == graph_name]

        mismatches = [r for r in graph_results if not r["result_matches_reference"]]

        if mismatches:
            print(
                f"  ❌ Graph '{graph_name}': {len(mismatches)}/{len(graph_results)} orders produced different results!"
            )
            consistent = False
        else:
            print(
                f"  ✅ Graph '{graph_name}': All {len(graph_results)} orders produced identical POMIS sets"
            )

    return consistent


def main():
    parser = argparse.ArgumentParser(
        description="Study effect of topological ordering on POMIS computation"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="pomis_topological_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-random",
        "-n",
        type=int,
        default=20,
        help="Number of random topological orders to test per graph",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--graphs", nargs="*", help="Specific graphs to test (default: all available)"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("POMIS Topological Ordering Experiment")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Random orders per graph: {args.num_random}")

    # Load test graphs
    all_graphs = load_test_graphs()
    if args.graphs:
        graphs = {name: all_graphs[name] for name in args.graphs if name in all_graphs}
    else:
        graphs = all_graphs

    print(f"Testing {len(graphs)} graphs: {list(graphs.keys())}")

    # Run experiments
    all_results = []
    for graph_name, (G, Y) in graphs.items():
        try:
            results = run_single_experiment(
                graph_name, G, Y, args.output_dir, args.num_random
            )
            all_results.extend(results)
        except Exception as e:
            print(f"ERROR processing graph {graph_name}: {e}")
            continue

    if not all_results:
        print("No results generated. Exiting.")
        return 1

    # Save results
    save_results(all_results, args.output_dir)

    # Verify consistency
    consistent = verify_result_consistency(all_results)

    # Create plots
    if not args.no_plots:
        try:
            create_plots(all_results, args.output_dir)
        except Exception as e:
            print(f"Error creating plots: {e}")

    # Summary
    print(f"\n{'=' * 50}")
    print("Experiment completed!")
    print(f"Total runs: {len(all_results)}")
    print(f"Graphs tested: {len(graphs)}")
    print(f"Result consistency: {'✅ PASS' if consistent else '❌ FAIL'}")
    print(f"Results saved to: {args.output_dir}")

    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
