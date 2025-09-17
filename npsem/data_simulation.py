#!/usr/bin/env python3
"""
Data Simulation Utilities

This module provides functions for simulating data from Structural Causal Models.
"""

import numpy as np
from typing import List, Tuple, Optional
from npsem.utils import seeded


def simulate_data_from_scm(
    scm, n_samples: int = 4000, seed: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Simulate data from a ground truth SCM.

    Parameters:
    -----------
    scm : StructuralCausalModel
        Ground truth SCM to simulate from
    n_samples : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    data : np.ndarray
        Simulated data matrix (n_samples x n_variables)
    var_names : List[str]
        Variable names in order
    """
    with seeded(seed):
        # Get variable names in causal order
        var_names = list(scm.G.causal_order())

        # Generate samples
        data = []
        for _ in range(n_samples):
            sample = {}

            # Sample exogenous variables
            for u in scm.G.U | scm.more_U:
                sample[u] = np.random.choice(scm.D[u])

            # Evaluate structural equations in causal order
            for var in var_names:
                if var in scm.F:
                    sample[var] = scm.F[var](sample)
                else:
                    # If no structural equation, sample from domain
                    sample[var] = np.random.choice(scm.D[var])

            # Extract values in correct order
            data.append([sample[var] for var in var_names])

        return np.array(data), var_names
