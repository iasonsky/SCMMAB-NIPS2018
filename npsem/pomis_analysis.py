#!/usr/bin/env python3
"""
POMIS Analysis Utilities

This module provides functions for computing POMIS sets and analyzing
them across multiple DAGs in a Markov equivalence class.
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from npsem.where_do import POMISs, MISs
from npsem.causal_diagram_utils import (
    dagmatrix_to_CausalDiagram,
    admgmatrix_to_CausalDiagram,
)


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


def mis_union_over_dags(
    dag_mats: List[np.ndarray],
    names: List[str],
    Y: str,
    enforce_Y_sink: bool = True,
) -> Set[Tuple[str, ...]]:
    """
    Compute union of MIS sets across all DAGs in the MEC.

    Parameters:
    -----------
    dag_mats : List[np.ndarray]
        List of DAG adjacency matrices
    names : List[str]
        Variable names
    Y : str
        Target variable
    enforce_Y_sink : bool
        Whether to filter out DAGs where Y has outgoing edges

    Returns:
    --------
    Set[Tuple[str, ...]]
        Union of all MIS sets across DAGs
    """
    union = set()

    for A in dag_mats:
        # Enforce Y is sink (filter out DAGs with Y->*)
        if enforce_Y_sink:
            y_idx = names.index(Y)
            if np.any(A[y_idx, :] == 1):
                continue

        cd = dagmatrix_to_CausalDiagram(A, names)

        for S in MISs(cd, Y):
            union.add(tuple(sorted(S)))

    return {tuple(s) for s in union}


def pomis_union_over_admgs(
    admg_mats: List[np.ndarray],
    names: List[str],
    Y: str,
    N: Optional[List[str]] = None,
    enforce_Y_sink: bool = True,
) -> Set[Tuple[str, ...]]:
    """
    Compute union of POMIS sets across all ADMGs.

    Parameters:
    -----------
    admg_mats : List[np.ndarray]
        List of ADMG adjacency matrices
    names : List[str]
        Variable names
    Y : str
        Target variable
    N : List[str], optional
        Non-manipulable variables (for latent projection)
    enforce_Y_sink : bool
        Whether to filter out ADMGs where Y has outgoing edges

    Returns:
    --------
    Set[Tuple[str, ...]]
        Union of all POMIS sets across ADMGs
    """
    union = set()
    count = 0
    for i, A in enumerate(admg_mats):
        # Enforce Y is sink (filter out ADMGs with Y->*)
        if enforce_Y_sink:
            y_idx = names.index(Y)
            # Check for directed edges from Y (value 1 or 101)
            if np.any((A[y_idx, :] == 1) | (A[y_idx, :] == 101)):
                count += 1  # Count discarded ADMGs
                print(
                    f"Discarded ADMG {i + 1}"
                )  # Print index of discarded ADMGs for debugging
                continue
        cd = admgmatrix_to_CausalDiagram(A, names)
        # POMIS algorithm automatically handles bidirected edges (latent confounders)
        for S in POMISs(cd, Y):
            union.add(tuple(sorted(S)))
    print(
        f"Discarded total {count} ADMGs"
    )  # Print number of discarded ADMGs and their index for debugging
    return {tuple(s) for s in union}


def mis_union_over_admgs(
    admg_mats: List[np.ndarray],
    names: List[str],
    Y: str,
    enforce_Y_sink: bool = True,
) -> Set[Tuple[str, ...]]:
    """
    Compute union of MIS sets across all ADMGs.

    Parameters:
    -----------
    admg_mats : List[np.ndarray]
        List of ADMG adjacency matrices
    names : List[str]
        Variable names
    Y : str
        Target variable
    enforce_Y_sink : bool
        Whether to filter out ADMGs where Y has outgoing edges

    Returns:
    --------
    Set[Tuple[str, ...]]
        Union of all MIS sets across ADMGs
    """
    union = set()

    for A in admg_mats:
        # Enforce Y is sink (filter out ADMGs with Y->*)
        if enforce_Y_sink:
            y_idx = names.index(Y)
            # Check for directed edges from Y (value 1 or 101)
            if np.any((A[y_idx, :] == 1) | (A[y_idx, :] == 101)):
                continue

        cd = admgmatrix_to_CausalDiagram(A, names)

        # MIS algorithm automatically handles bidirected edges (latent confounders)
        for S in MISs(cd, Y):
            union.add(tuple(sorted(S)))

    return {tuple(s) for s in union}
