# Structural Causal Bandits: Where to Intervene?

*Sanghack Lee and Elias Bareinboim* Structural Causal Bandits: Where to Intervene? In _Advances in Neural Information Processing System 31 (NIPS'2018), 2018

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
(At this moment, the code is not well-documented.) 

The code is tested with the following configuration: `python=3.9`, `numpy=1.21.2`,
`scipy=1.7.1`, `joblib=1.0.1`, `matplotlib=3.4.3`, `seaborn=0.11.2`, and
`networkx=2.6.3`, on Linux and MacOS machines.

## Getting Started

Install the package in editable mode together with the required dependencies:

```bash
python -m pip install -e .
```

After installation the `npsem` module can be imported from anywhere.

## Module Overview

- **bandits.py** – implementations of KL-UCB and Thompson Sampling algorithms and
  utilities for running bandit simulations.
- **model.py** – data structures for causal diagrams and structural causal models
  with simple inference routines.
- **where_do.py** – algorithms for computing Minimal Intervention Sets (MIS) and
  POMISs used for deciding where to intervene.
- **scm_bandits.py** – helpers to convert a structural causal model into a bandit
  machine and utilities for selecting arms.
- **utils.py** – small helper functions for randomization, seeding and misc
  utilities.
- **viz_util.py** – basic plotting helpers such as sparse index generation.

## Getting Started

For full environment setup instructions, see
[snellius_uv_setup.md](snellius_uv_setup.md).


## Reproducing the Experiments

Run the following command to execute the bandit experiments (it uses 3/4 of the
available CPU cores):

```bash
python3 -m npsem.NIPS2018POMIS_exp.test_bandit_strategies
```

This step produces a `bandit_results` directory with subdirectories for each of
the three tasks discussed in the paper. To generate the plots from these
results run:

```bash
python3 -m npsem.NIPS2018POMIS_exp.test_drawing_re
```

## POMIS Topological Ordering Experiments

This repository now includes instrumentation to study the effect of topological ordering on POMIS computation performance. The experiment measures how different topological orderings affect:

- Total recursive calls to subPOMIS
- Number of IB (Interventional Border) evaluations  
- Number of pruned branches due to overlap checks
- Computation time per run
- Verification that final POMIS sets are invariant across orders

### Running the Topological Ordering Experiment

Use the following command to run the experiment:

```bash
# Basic usage - test all available graphs with default settings
python3 -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment

# With custom parameters
python3 -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment \
    --output-dir my_results \
    --num-random 20 \
    --seed 42 \
    --graphs XYZWST IV

# Skip plot generation (useful for large experiments)  
python3 -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment --no-plots
```

### Command Line Options

- `--output-dir, -o`: Output directory for results (default: `pomis_topological_results`)
- `--num-random, -n`: Number of random topological orders to test per graph (default: 20)
- `--seed, -s`: Random seed for reproducibility (default: 42)
- `--graphs`: Specific graphs to test (default: all available graphs)
- `--no-plots`: Skip plot generation

### Experiment Output

The experiment generates:

1. **CSV Logs** (`pomis_topological_results.csv`): Detailed metrics for each graph and topological ordering
2. **JSON Results** (`pomis_topological_results.json`): Same data in JSON format for programmatic access
3. **Summary Statistics** (`pomis_topological_summary.csv`): Min/median/max statistics by graph and metric
4. **Plots** (`pomis_topological_distributions.pdf/png`): Visualization of metric distributions across orderings
5. **Console Output**: Real-time verification that all orderings produce identical POMIS sets

### Available Test Graphs

The experiment automatically tests several causal graphs:
- **XYZWST**: 6-node graph with bidirected edges (main example from paper)  
- **XYZW**: Simplified 4-node version
- **IV**: Instrumental variable graph (3 nodes)
- **simple_markovian**: 5-node Markovian graph

### Key Findings

The instrumentation demonstrates that:
- ✅ All topological orderings produce **identical POMIS sets** (algorithm correctness)
- 📊 Different orderings can have **varying computational costs**
- 🔍 The instrumentation captures detailed metrics for performance analysis

### Using uv (Recommended)

If you have `uv` installed:

```bash
# Install dependencies and run experiment
uv sync
uv run python -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment

# Quick test with fewer random orders
uv run python -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment --num-random 5 --no-plots
```
