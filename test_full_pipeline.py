#!/usr/bin/env python3
"""
Complete Pipeline: Data Generation → CPDAG Discovery → DAG Enumeration → POMIS Analysis

This script demonstrates the full workflow for:
1. Generating synthetic data from a causal chain Z -> X -> Y
2. Discovering the CPDAG using PC algorithm
3. Enumerating all DAGs in the Markov equivalence class
4. Computing POMIS sets across all DAGs
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri, StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Causal discovery
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ConstraintBased.PC import pc

# Your project modules
from npsem.model import CausalDiagram
from npsem.where_do import POMISs

# =============================================================================
# R INTEGRATION SETUP
# =============================================================================

# Import R packages
pcalg = importr("pcalg")

# Define R helper function for DAG enumeration
ro.r("""
cpdag_enum_from_amat <- function(amat) {
  res <- pdag2allDags(amat)
  if (is.null(res) || is.null(res$dags) || nrow(res$dags) == 0) return(list())
  p <- nrow(amat)
  out <- vector("list", nrow(res$dags))
  for (i in seq_len(nrow(res$dags))) {
    M <- matrix(res$dags[i,], p, p, byrow=TRUE)
    dimnames(M) <- list(res$nodeNms, res$nodeNms)
    out[[i]] <- M
  }
  out
}
""")
cpdag_enum_from_amat = ro.globalenv["cpdag_enum_from_amat"]

# =============================================================================
# DATA GENERATION
# =============================================================================


def make_chain_data(n=3000, seed=0, bz=0.8, ax=0.9, sz=1.0, sx=0.5, sy=0.5):
    """
    Generate synthetic data from a causal chain: Z -> X -> Y

    Parameters:
    -----------
    n : int
        Sample size
    seed : int
        Random seed for reproducibility
    bz : float
        Coefficient for Z -> X
    ax : float
        Coefficient for X -> Y
    sz, sx, sy : float
        Standard deviations for noise terms

    Returns:
    --------
    data : np.ndarray
        Generated data matrix (n x 3)
    names : list
        Variable names ['Z', 'X', 'Y']
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, sz, n)
    X = bz * Z + rng.normal(0, sx, n)
    Y = ax * X + rng.normal(0, sy, n)
    return np.column_stack([Z, X, Y]), ["Z", "X", "Y"]


# =============================================================================
# CAUSAL DISCOVERY
# =============================================================================


def pc_cpdag_adjacency(data, names, alpha=0.01, save_plot=True):
    """
    Run PC algorithm to discover CPDAG from data.

    Parameters:
    -----------
    data : np.ndarray
        Data matrix (n x p)
    names : list
        Variable names
    alpha : float
        Significance level for independence tests
    save_plot : bool
        Whether to save CPDAG visualization

    Returns:
    --------
    A : np.ndarray
        CPDAG adjacency matrix
    names : list
        Variable names (same as input)
    """
    cg = pc(data, alpha=alpha, ind_test=fisherz)
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


# =============================================================================
# DAG ENUMERATION
# =============================================================================


def enumerate_dags_from_cpdag(adj_cpdag: np.ndarray, var_names):
    """
    Enumerate all DAGs in the Markov equivalence class of a CPDAG.

    Parameters:
    -----------
    adj_cpdag : np.ndarray
        CPDAG adjacency matrix (1 both directions = undirected; 1→0 = directed)
    var_names : list
        Variable names

    Returns:
    --------
    list
        List of DAG adjacency matrices
    """
    A = np.asarray(adj_cpdag, dtype=float)

    # Set dimnames in R so pcalg can keep node names
    with localconverter(ro.default_converter + numpy2ri.converter):
        A_r = ro.conversion.py2rpy(A)
    A_r.rownames = StrVector(list(var_names))
    A_r.colnames = StrVector(list(var_names))

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_list = cpdag_enum_from_amat(A_r)
        mats = ro.conversion.rpy2py(r_list)

    return [np.asarray(M, dtype=int) for M in mats]


# =============================================================================
# CAUSAL DIAGRAM CONVERSION
# =============================================================================


def dagmatrix_to_CausalDiagram(A, names):
    """
    Convert DAG adjacency matrix to CausalDiagram object.

    Parameters:
    -----------
    A : np.ndarray
        DAG adjacency matrix
    names : list
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


# =============================================================================
# POMIS ANALYSIS
# =============================================================================


def pomis_union_over_dags(dag_mats, names, Y, N=None, enforce_Y_sink=True):
    """
    Compute union of POMIS sets across all DAGs in the MEC.

    Parameters:
    -----------
    dag_mats : list
        List of DAG adjacency matrices
    names : list
        Variable names
    Y : str
        Target variable
    N : list, optional
        Non-manipulable variables (for latent projection)
    enforce_Y_sink : bool
        Whether to filter out DAGs where Y has outgoing edges

    Returns:
    --------
    set
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


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================


def main():
    """Run the complete pipeline demonstration."""
    print("=" * 60)
    print("CAUSAL DISCOVERY AND POMIS ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data from Z -> X -> Y...")
    data, names = make_chain_data(n=4000, seed=42)
    print(f"   Generated {data.shape[0]} samples for variables: {names}")

    # 2. Discover CPDAG using PC algorithm
    print("\n2. Running PC algorithm to discover CPDAG...")
    A_cpdag_cl, names_pc = pc_cpdag_adjacency(data, names, alpha=0.01)
    print("   Raw CPDAG from causal-learn:")
    print(f"   {A_cpdag_cl}")

    # 3. Convert to pcalg format
    print("\n3. Converting CPDAG to pcalg format...")
    A_cpdag = cl_cpdag_to_pcalg(A_cpdag_cl)
    print("   CPDAG normalized for pcalg:")
    print(f"   {A_cpdag}")

    # 4. Enumerate all DAGs in the MEC
    print("\n4. Enumerating DAGs in Markov equivalence class...")
    dags = enumerate_dags_from_cpdag(A_cpdag, names_pc)
    print(f"   Found {len(dags)} DAGs in the MEC:")
    for k, A in enumerate(dags, 1):
        print(f"   DAG {k}:\n{A}\n")

    # 5. Compute POMIS union
    print("\n5. Computing POMIS union across all DAGs...")
    union = pomis_union_over_dags(dags, names_pc, Y="Y", N=None, enforce_Y_sink=True)
    print(f"   Union of POMIS sets: {union}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
