#!/usr/bin/env python3
"""
Causal Discovery Pipeline

This module provides high-level functions for running complete causal discovery
pipelines, integrating all the individual components.
"""

import numpy as np
from typing import List, Tuple, Set

from npsem.causal_discovery import pc_cpdag_adjacency, cl_cpdag_to_pcalg
from npsem.dag_enumeration import enumerate_dags_from_cpdag
from npsem.pomis_analysis import pomis_union_over_dags


def run_causal_discovery_pipeline(
    data: np.ndarray,
    var_names: List[str],
    alpha: float = 0.05,
    ind_test: str = "fisherz",
    Y: str = "Y",
    save_plot: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], Set[Tuple[str, ...]]]:
    """
    Run complete causal discovery pipeline: CPDAG → DAGs → POMIS union.

    Parameters:
    -----------
    data : np.ndarray
        Data matrix (n x p)
    var_names : List[str]
        Variable names
    alpha : float
        Significance level for independence tests
    ind_test : str
        Independence test to use
    Y : str
        Target variable for POMIS analysis
    save_plot : bool
        Whether to save CPDAG visualization

    Returns:
    --------
    cpdag_adj : np.ndarray
        CPDAG adjacency matrix
    dags : List[np.ndarray]
        List of DAGs in the MEC
    pomis_union : Set[Tuple[str, ...]]
        Union of POMIS sets across all DAGs
    """
    # Step 1: Discover CPDAG
    cpdag_cl, names_pc = pc_cpdag_adjacency(data, var_names, alpha, ind_test, save_plot)

    # Step 2: Convert to pcalg format
    cpdag_pcalg = cl_cpdag_to_pcalg(cpdag_cl)

    # Step 3: Enumerate DAGs
    dags = enumerate_dags_from_cpdag(cpdag_pcalg, names_pc)

    # Step 4: Compute POMIS union
    pomis_union = pomis_union_over_dags(dags, names_pc, Y)

    return cpdag_pcalg, dags, pomis_union
