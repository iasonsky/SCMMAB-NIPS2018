#!/usr/bin/env python3
"""
Causal Discovery Algorithms

This module provides functions for running causal discovery algorithms,
primarily the PC algorithm for discovering CPDAGs.
"""

import numpy as np
from typing import List, Tuple
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ConstraintBased.PC import pc


def pc_cpdag_adjacency(
    data: np.ndarray,
    names: List[str],
    alpha: float = 0.05,
    ind_test: str = "fisherz",
    save_plot: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Run PC algorithm to discover CPDAG from data.

    Parameters:
    -----------
    data : np.ndarray
        Data matrix (n x p)
    names : List[str]
        Variable names
    alpha : float
        Significance level for independence tests
    ind_test : str
        Independence test to use ('fisherz', 'chisq', 'gsq')
    save_plot : bool
        Whether to save CPDAG visualization

    Returns:
    --------
    A : np.ndarray
        CPDAG adjacency matrix
    names : List[str]
        Variable names (same as input)
    """
    cg = pc(data, alpha=alpha, ind_test=ind_test)
    A = np.array(cg.G.graph)

    if save_plot:
        pyd = GraphUtils.to_pydot(cg.G, labels=list(names))
        pyd.write_png("cpdag.png")
        print("CPDAG visualization saved as 'cpdag.png'")

    return A.astype(float), list(names)


def cl_cpdag_to_pcalg(A_cl: np.ndarray) -> np.ndarray:
    """
    Convert causal-learn CPDAG adjacency to pcalg PDAG/CPDAG adjacency.

    Causal-learn format:
    - Directed i->j: A[i,j]!=0 and A[j,i]==0
    - Undirected i--j: A[i,j]!=0 and A[j,i]!=0

    pcalg format:
    - Directed i->j: 1 at (i,j), 0 at (j,i)
    - Undirected i--j: 1 at both (i,j) and (j,i)

    Parameters:
    -----------
    A_cl : np.ndarray
        Causal-learn CPDAG adjacency matrix

    Returns:
    --------
    A_pcalg : np.ndarray
        pcalg-compatible CPDAG adjacency matrix
    """
    A = np.asarray(A_cl)
    p = A.shape[0]
    B = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if A[i, j] != 0:  # any nonzero means "there is an endpoint mark"
                if A[j, i] == 0:  # only one direction nonzero => directed i->j
                    B[i, j] = 1
                else:  # both nonzero => undirected
                    B[i, j] = 1
                    B[j, i] = 1

    np.fill_diagonal(B, 0)
    return B
