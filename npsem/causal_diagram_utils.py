#!/usr/bin/env python3
"""
Causal Diagram Utilities

This module provides functions for working with CausalDiagram objects,
including conversions from adjacency matrices.
"""

import numpy as np
from typing import List
from npsem.model import CausalDiagram


def dagmatrix_to_CausalDiagram(A: np.ndarray, names: List[str]) -> CausalDiagram:
    """
    Convert DAG adjacency matrix to CausalDiagram object.

    Parameters:
    -----------
    A : np.ndarray
        DAG adjacency matrix
    names : List[str]
        Variable names

    Returns:
    --------
    CausalDiagram
        CausalDiagram object representing the DAG
    """
    V = set(names)
    dir_edges = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(len(names))
        if A[i, j] == 1
    ]
    return CausalDiagram(V, dir_edges, [])  # no bidirected edges after PC


def cpdag_to_CausalDiagram(A: np.ndarray, names: List[str]) -> CausalDiagram:
    """
    Convert CPDAG adjacency matrix to CausalDiagram object.

    Parameters:
    -----------
    A : np.ndarray
        CPDAG adjacency matrix
    names : List[str]
        Variable names

    Returns:
    --------
    CausalDiagram
        CausalDiagram object representing the CPDAG
    """
    V = set(names)
    dir_edges = []
    undir_edges = []

    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            if A[i, j] == 1 and A[j, i] == 0:
                # Directed edge i -> j
                dir_edges.append((names[i], names[j]))
            elif A[i, j] == 1 and A[j, i] == 1:
                # Undirected edge i -- j
                undir_edges.append((names[i], names[j]))

    # Convert undirected edges to bidirected edges format (x, y, name)
    bidirected_edges = [(x, y, f"U_{x}_{y}") for x, y in undir_edges]

    return CausalDiagram(V, dir_edges, bidirected_edges)


def causal_diagram_to_adjacency(cd: CausalDiagram, names: List[str]) -> np.ndarray:
    """
    Convert CausalDiagram to adjacency matrix.

    Parameters:
    -----------
    cd : CausalDiagram
        CausalDiagram object
    names : List[str]
        Variable names in order

    Returns:
    --------
    np.ndarray
        Adjacency matrix
    """
    n = len(names)
    A = np.zeros((n, n), dtype=int)

    # Add directed edges
    for i, j in cd.edges:
        if i in names and j in names:
            A[names.index(i), names.index(j)] = 1

    # Add undirected edges (represented as bidirectional)
    for i, j in cd.bidirected_edges:
        if i in names and j in names:
            A[names.index(i), names.index(j)] = 1
            A[names.index(j), names.index(i)] = 1

    return A
