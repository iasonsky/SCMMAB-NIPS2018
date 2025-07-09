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
