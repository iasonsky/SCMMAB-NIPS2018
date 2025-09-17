#!/usr/bin/env python3
"""
DAG Enumeration Utilities

This module provides functions for enumerating DAGs from CPDAGs using R's pcalg package.
"""

import numpy as np
from typing import List
from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri, StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Import R packages
pcalg = importr("pcalg")

# Define R helper function for DAG enumeration
ro.r("""
cpdag_enum_from_amat <- function(amat) {
  res <- pdag2allDags(amat)
  if (is.null(res) || is.null(res$dags) || is.null(nrow(res$dags)) || nrow(res$dags) == 0) return(list())
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


def enumerate_dags_from_cpdag(
    adj_cpdag: np.ndarray, var_names: List[str]
) -> List[np.ndarray]:
    """
    Enumerate all DAGs in the Markov equivalence class of a CPDAG.

    Parameters:
    -----------
    adj_cpdag : np.ndarray
        CPDAG adjacency matrix (1 both directions = undirected; 1→0 = directed)
    var_names : List[str]
        Variable names

    Returns:
    --------
    List[np.ndarray]
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
