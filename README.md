# Structural Causal Bandits: Where to Intervene?

*Sanghack Lee and Elias Bareinboim* Structural Causal Bandits: Where to Intervene? In _Advances in Neural Information Processing System 31 (NIPS'2018), 2018

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
(At this moment, the code is not well-documented.) 

The code is tested with the following configuration: `python=3.13`, `numpy=2.3.2`,
`scipy=1.16.1`, `joblib=1.5.2`, `matplotlib=3.10.5`, `seaborn=0.13.2`, and
`networkx=3.5`, on Linux and MacOS machines.

## Getting Started

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

Install the package in editable mode together with the required dependencies:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

After installation the `npsem` module can be imported from anywhere.

### 3. Deactivate When Done

```bash
deactivate
```

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

## Environment Setup

For detailed environment setup instructions, see
[snellius_uv_setup.md](snellius_uv_setup.md) (note: this file contains uv-specific instructions for Snellius supercomputer).


## Reproducing the Experiments

### Original Paper Experiments

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

### Modular Bandit Experiments

The repository now includes a modular framework for running bandit experiments on various SCMs:

```bash
# List available SCMs
python npsem/NIPS2018POMIS_exp/run_bandit_experiments.py --list

# Run specific SCM
python npsem/NIPS2018POMIS_exp/run_bandit_experiments.py --scm frontdoor

# Run all SCMs
python npsem/NIPS2018POMIS_exp/run_bandit_experiments.py --all

# Quick testing mode
python npsem/NIPS2018POMIS_exp/run_bandit_experiments.py --scm six_variable --quick
```

### Results

Experiment results and visualizations are available in the [`results/`](results/) directory:
- **Images**: PNG files showing cumulative regret, final regret comparisons, and causal discovery results
- **Documentation**: Detailed explanation of each result type and key findings

The modular framework supports:
- **8 different SCMs**: Including frontdoor, six-variable, and four-variable graphs
- **3 arm strategies**: POMIS (with latent projection), MIS, and Brute-force
- **2 bandit algorithms**: Thompson Sampling and UCB
- **Automatic handling**: Non-manipulable variables via latent projection

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

### Running with Virtual Environment

Make sure to activate your virtual environment first:

```bash
# Activate virtual environment
source venv/bin/activate

# Run experiment
python -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment

# Quick test with fewer random orders
python -m npsem.NIPS2018POMIS_exp.pomis_topological_experiment --num-random 5 --no-plots
```
