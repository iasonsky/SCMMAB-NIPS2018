#!/usr/bin/env python3
"""
Causal Discovery Pipeline

This module provides high-level functions for running complete causal discovery
pipelines, integrating all the individual components.
"""

import numpy as np
from typing import List, Tuple, Set, Optional

from npsem.causal_discovery import (
    pc_cpdag_adjacency,
    cl_cpdag_to_pcalg,
    fci_pag_adjacency,
)
from npsem.dag_enumeration import enumerate_dags_from_cpdag
from npsem.pomis_analysis import (
    pomis_union_over_dags,
    mis_union_over_dags,
    pomis_union_over_admgs,
    mis_union_over_admgs,
)
from npsem.causal_visualization import (
    create_combined_sanity_check_visualization,
)
from npsem.causal_diagram_utils import pag_to_admg


def run_causal_discovery_pipeline_UC(
    data: np.ndarray,
    var_names: List[str],
    alpha: float = 0.05,
    ind_test: str = "fisherz",
    Y: str = "Y",
    sanity_check: bool = False,
    ground_truth_scm: Optional[object] = None,
    save_dir: str = "figures",
) -> Tuple[np.ndarray, List[np.ndarray], Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
    """
    Run complete causal discovery pipeline: PAG → ADMGs → POMIS/MIS unions.
    """

    if sanity_check:
        print("\n🔍 SANITY CHECK MODE: Step-by-step verification")
        print("=" * 60)

    # Step 1: Discover PAG
    pag_cl, var_names, edges = fci_pag_adjacency(
        data, var_names, alpha, ind_test, save_plot=True
    )

    if sanity_check:
        print("\n1️⃣ PAG Discovery Complete")
        print(f"   Found PAG with {np.sum(pag_cl != 0)} edges")
        print(f"   PAG matrix:\n{pag_cl}")

    # Step 2: Convert PAG to ADMGs
    admgs = pag_to_admg(
        pag_cl, var_names, edges
    )  # TODO: need to fix this for four_variable_SCM_strong
    # TODO: Also for frontdoor_SCM_strong it crashes because of fisherz test (need to fix this)
    if sanity_check:
        print("\n2️⃣ ADMG Enumeration Complete")
        print(f"   Found {len(admgs)} ADMGs consistent with PAG")

    # Step 3: Compute POMIS union over ADMGs
    pomis_union = pomis_union_over_admgs(admgs, var_names, Y, enforce_Y_sink=True)

    # Step 4: Compute MIS union over ADMGs
    mis_union = mis_union_over_admgs(admgs, var_names, Y, enforce_Y_sink=True)
    if sanity_check:
        print("\n3️⃣ POMIS Analysis Complete")
        print(f"   POMIS Union: {pomis_union}")
        print(f"   MIS Union: {mis_union}")
        print("\n✅ SANITY CHECK COMPLETE")
        print("=" * 60)

    return pag_cl, admgs, pomis_union, mis_union


def run_causal_discovery_pipeline(
    data: np.ndarray,
    var_names: List[str],
    alpha: float = 0.05,
    ind_test: str = "chisq",
    Y: str = "Y",
    sanity_check: bool = False,
    ground_truth_scm: Optional[object] = None,
    save_dir: str = "figures",
) -> Tuple[np.ndarray, List[np.ndarray], Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
    """
    Run complete causal discovery pipeline: CPDAG → DAGs → POMIS/MIS unions.

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
        Target variable for POMIS/MIS analysis
    sanity_check : bool
        Whether to show step-by-step visualizations
    ground_truth_scm : StructuralCausalModel, optional
        Ground truth SCM for sanity check comparison

    Returns:
    --------
    cpdag_adj : np.ndarray
        CPDAG adjacency matrix
    dags : List[np.ndarray]
        List of DAGs in the MEC
    pomis_union : Set[Tuple[str, ...]]
        Union of POMIS sets across all DAGs
    mis_union : Set[Tuple[str, ...]]
        Union of MIS sets across all DAGs
    """

    if sanity_check:
        print("\n🔍 SANITY CHECK MODE: Step-by-step verification")
        print("=" * 60)

    # Step 1: Discover CPDAG
    cpdag_cl, names_pc = pc_cpdag_adjacency(
        data, var_names, alpha, ind_test, save_plot=False
    )

    if sanity_check:
        print("\n1️⃣ CPDAG Discovery Complete")
        print(f"   Found CPDAG with {np.sum(cpdag_cl != 0)} edges")
        print(f"   CPDAG matrix:\n{cpdag_cl}")

    # Step 2: Convert to pcalg format
    cpdag_pcalg = cl_cpdag_to_pcalg(cpdag_cl)

    # Step 3: Enumerate DAGs
    dags = enumerate_dags_from_cpdag(cpdag_pcalg, names_pc)

    if sanity_check:
        print("\n2️⃣ DAG Enumeration Complete")
        print(f"   Found {len(dags)} DAGs in the Markov Equivalence Class")

    # Step 4: Compute POMIS union
    pomis_union = pomis_union_over_dags(
        dags, names_pc, Y, enforce_Y_sink=True
    )  # False to include DAGs with Y->*

    # Step 5: Compute MIS union
    mis_union = mis_union_over_dags(
        dags, names_pc, Y, enforce_Y_sink=True
    )  # False to include DAGs with Y->*

    if sanity_check:
        print("\n3️⃣ POMIS Analysis Complete")
        print(f"   POMIS Union: {pomis_union}")

        print("\n4️⃣ MIS Analysis Complete")
        print(f"   MIS Union: {mis_union}")

        # Create combined visualization
        if ground_truth_scm is not None:
            combined_path = create_combined_sanity_check_visualization(
                ground_truth_scm=ground_truth_scm,
                cpdag_matrix=cpdag_cl,
                dags=dags,
                var_names=var_names,
                Y=Y,
                figures_dir=save_dir,
            )
            print(f"📊 Combined visualization saved: {combined_path}")

        print("\n✅ SANITY CHECK COMPLETE")
        print("=" * 60)

    return cpdag_pcalg, dags, pomis_union, mis_union
