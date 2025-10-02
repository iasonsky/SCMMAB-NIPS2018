#!/usr/bin/env python3
"""
Causal Diagram Utilities

This module provides functions for working with CausalDiagram objects,
including conversions from adjacency matrices.
"""

import numpy as np
from typing import List, Tuple
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


def pag_to_admg(
    pag_adj: np.ndarray, var_names: List[str], edges: List[Tuple[str, str]]
) -> List[np.ndarray]:
    """
    Convert PAG adjacency matrix to list of ADMG adjacency matrices using R implementation.

    Parameters:
    -----------
    pag_adj : np.ndarray
        PAG adjacency matrix from FCI (causal-learn format)
    var_names : List[str]
        Variable names
    edges : List[Tuple[str, str]]
        Edge information from FCI

    Returns:
    --------
    List[np.ndarray]
        List of ADMG adjacency matrices
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, StrVector
    from rpy2.robjects.conversion import localconverter
    from pathlib import Path

    try:
        # Load the R scripts
        script_dir = Path(__file__).parent.parent / "r_scripts"
        ro.r(f'source("{script_dir}/full_admg_learning.R")')

        # Convert FCI PAG format to R format
        # FCI uses: 0=no edge, 2=undirected edge
        # R expects: 0=no edge, 1=undirected edge
        pag_r_format = _convert_fci_pag_to_r_format(pag_adj)

        # Set up R matrix with proper names
        with localconverter(ro.default_converter + numpy2ri.converter):
            pag_r = ro.conversion.py2rpy(pag_r_format)

        # Set row and column names
        pag_r.rownames = StrVector(var_names)
        pag_r.colnames = StrVector(var_names)

        # Create PAG object in R
        ro.r.assign("pag_matrix", pag_r)
        ro.r.assign("var_names", StrVector(var_names))

        # Create PAG object using R function
        ro.r("""
        pag_obj <- make_pag_from_amat(pag_matrix)
        """)

        # Convert PAG to ADMGs
        ro.r("""
        admg_list <- pag2admg(pag_obj)
        """)

        # Get the ADMG list from R
        admg_list_r = ro.r("admg_list")

        # Convert back to Python format
        admg_matrices = []
        if len(admg_list_r) > 0:
            for i in range(len(admg_list_r)):
                admg_matrix = np.array(admg_list_r[i])
                admg_matrices.append(admg_matrix)

        # Save results for verification
        _save_pag_admg_results(pag_adj, pag_r_format, admg_matrices, var_names)

        return admg_matrices

    except Exception as e:
        print(f"Error in pag_to_admg: {e}")
        import traceback

        traceback.print_exc()
        return []
    finally:
        pass


def _convert_fci_pag_to_r_format(pag_adj: np.ndarray) -> np.ndarray:
    """
    Convert FCI PAG format to R format.

    FCI format (Causal-Learn):
    - 0: no edge
    - 1: arrowhead (→)
    - 2: circle (o)
    - -1: tail (-)

    Edge types in FCI:
    - A → B: G[A,B] = -1, G[B,A] = 1  (direct causal link)
    - A o→ B: G[A,B] = 2, G[B,A] = 1  (possible causal link, no ancestor)
    - A o-o B: G[A,B] = 2, G[B,A] = 2 (undetermined causal link)
    - A <-> B: G[A,B] = 1, G[B,A] = 1 (latent common cause)

    R format (pcalg):
    - 0: no edge
    - 1: undirected edge (circle-circle)
    - 2: directed edge (arrow)
    - 3: partially directed edge (circle-arrow)
    """
    pag_r = pag_adj.copy()
    n = pag_adj.shape[0]

    # Process each edge pair
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            val_ij = pag_adj[i, j]
            val_ji = pag_adj[j, i]

            # No edge
            if val_ij == 0 and val_ji == 0:
                pag_r[i, j] = 0
                pag_r[j, i] = 0

            # Direct causal link: A → B (G[A,B] = -1, G[B,A] = 1)
            elif val_ij == -1 and val_ji == 1:
                pag_r[i, j] = 2  # directed edge
                pag_r[j, i] = 0

            # Direct causal link: B → A (G[A,B] = 1, G[B,A] = -1)
            elif val_ij == 1 and val_ji == -1:
                pag_r[i, j] = 0
                pag_r[j, i] = 2  # directed edge

            # Possible causal link: A o→ B (G[A,B] = 2, G[B,A] = 1)
            elif val_ij == 2 and val_ji == 1:
                pag_r[i, j] = 3  # partially directed edge
                pag_r[j, i] = 0

            # Possible causal link: B o→ A (G[A,B] = 1, G[B,A] = 2)
            elif val_ij == 1 and val_ji == 2:
                pag_r[i, j] = 0
                pag_r[j, i] = 3  # partially directed edge

            # Undetermined causal link: A o-o B (G[A,B] = 2, G[B,A] = 2)
            elif val_ij == 2 and val_ji == 2:
                pag_r[i, j] = 1  # undirected edge
                pag_r[j, i] = 1

            # Latent common cause: A <-> B (G[A,B] = 1, G[B,A] = 1)
            elif val_ij == 1 and val_ji == 1:
                pag_r[i, j] = 1  # undirected edge (represents bidirected)
                pag_r[j, i] = 1

            # Handle other cases (shouldn't happen in valid PAGs)
            else:
                print(f"Warning: Unknown edge type at ({i},{j}): {val_ij}, {val_ji}")
                pag_r[i, j] = 0
                pag_r[j, i] = 0

    return pag_r.astype(int)


def _save_pag_admg_results(
    pag_adj_fci: np.ndarray,
    pag_adj_r: np.ndarray,
    admg_matrices: List[np.ndarray],
    var_names: List[str],
):
    """Save PAG and ADMG results for verification."""
    from pathlib import Path

    # Create output directory
    output_dir = Path("pag_admg_results")
    output_dir.mkdir(exist_ok=True)

    # Save FCI PAG matrix
    np.savetxt(
        output_dir / "pag_fci_format.txt",
        pag_adj_fci,
        fmt="%d",
        header=f"FCI PAG format\nVariables: {var_names}",
    )

    # Save R PAG matrix
    np.savetxt(
        output_dir / "pag_r_format.txt",
        pag_adj_r,
        fmt="%d",
        header=f"R PAG format\nVariables: {var_names}",
    )

    # Save ADMG matrices
    for i, admg in enumerate(admg_matrices):
        np.savetxt(
            output_dir / f"admg_{i + 1}.txt",
            admg,
            fmt="%d",
            header=f"ADMG {i + 1}\nVariables: {var_names}",
        )

    # Save summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("PAG to ADMG Conversion Results\n")
        f.write(f"Variables: {var_names}\n")
        f.write(f"Number of ADMGs found: {len(admg_matrices)}\n")
        f.write(f"FCI PAG shape: {pag_adj_fci.shape}\n")
        f.write(f"R PAG shape: {pag_adj_r.shape}\n")
        f.write(f"\nFCI PAG matrix:\n{pag_adj_fci}\n")
        f.write(f"\nR PAG matrix:\n{pag_adj_r}\n")

    print(f"📊 PAG and ADMG results saved to: {output_dir}")
    print(f"   Found {len(admg_matrices)} ADMGs")
    print(
        "   Files saved: pag_fci_format.txt, pag_r_format.txt, admg_*.txt, summary.txt"
    )
