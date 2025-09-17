#!/usr/bin/env python3
"""
POMIS Analysis Utilities

This module provides functions for computing POMIS sets and analyzing
them across multiple DAGs in a Markov equivalence class.
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from npsem.where_do import POMISs
from npsem.causal_diagram_utils import dagmatrix_to_CausalDiagram


def pomis_union_over_dags(
    dag_mats: List[np.ndarray],
    names: List[str],
    Y: str,
    N: Optional[List[str]] = None,
    enforce_Y_sink: bool = True,
) -> Set[Tuple[str, ...]]:
    """
    Compute union of POMIS sets across all DAGs in the MEC.

    Parameters:
    -----------
    dag_mats : List[np.ndarray]
        List of DAG adjacency matrices
    names : List[str]
        Variable names
    Y : str
        Target variable
    N : List[str], optional
        Non-manipulable variables (for latent projection)
    enforce_Y_sink : bool
        Whether to filter out DAGs where Y has outgoing edges

    Returns:
    --------
    Set[Tuple[str, ...]]
        Union of all POMIS sets across DAGs
    """
    union = set()

    for A in dag_mats:
        # Enforce Y is sink (filter out DAGs with Y->*)
        if enforce_Y_sink:
            y_idx = names.index(Y)
            if np.any(A[y_idx, :] == 1):
                continue

        cd = dagmatrix_to_CausalDiagram(A, names)

        # TODO: Add latent projection if N is specified
        # if N:
        #     keep = set(names) - set(N)
        #     keep.add(Y)
        #     cd = latent_project_cd(cd, keep)

        for S in POMISs(cd, Y):
            union.add(tuple(sorted(S)))

    return {tuple(s) for s in union}
