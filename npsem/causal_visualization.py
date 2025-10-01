#!/usr/bin/env python3
"""
Causal Graph Visualization Module

This module provides visualization functions for causal graphs using pydot,
offering clean, publication-ready graph visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pydot
from typing import List, Optional


def ensure_figures_directory(figures_dir: str = "figures") -> str:
    """Ensure a figures directory exists and return its path."""
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def plot_causal_diagram_pydot(
    causal_diagram,
    var_names: List[str],
    filename: str,
    title: Optional[str] = None,
    highlight_nodes: Optional[List[str]] = None,
    figures_dir: Optional[str] = None,
    consistent_sizing: bool = False,
    highlight_color: str = "red",
) -> str:
    """
    Plot a causal diagram using pydot with optional node highlighting.

    Parameters:
    -----------
    causal_diagram : CausalDiagram
        The causal diagram to visualize
    var_names : List[str]
        Variable names for the nodes
    filename : str
        Base filename (without extension)
    title : str, optional
        Title for the plot
    highlight_nodes : List[str], optional
        Nodes to highlight
    figures_dir : str, optional
        Directory to save figures in (defaults to ./figures/)
    highlight_color : str, optional
        Color to use for highlighting nodes (default: "red")

    Returns:
    --------
    str
        Full path to the saved PNG file
    """
    # Set up directory
    if figures_dir is None:
        figures_dir = ensure_figures_directory("figures")

    full_path = os.path.join(figures_dir, f"{filename}.png")

    # Create pydot graph
    graph = pydot.Dot(graph_type="digraph")
    graph.set_rankdir("TB")  # Top to bottom layout
    graph.set_bgcolor("white")

    # Set consistent sizing for combined visualizations
    if consistent_sizing:
        graph.set_size("2,2!")  # Force smaller, consistent size
        graph.set_dpi(300)
        graph.set_margin(0.05)  # Minimal margins
        graph.set_ranksep(0.3)  # Reduce spacing between ranks
        graph.set_nodesep(0.3)  # Reduce spacing between nodes

    # Add nodes
    # Define color mappings
    color_map = {
        "red": ("red", "lightcoral"),
        "blue": ("blue", "lightblue"),
        "green": ("green", "lightgreen"),
        "orange": ("orange", "lightyellow"),
        "purple": ("purple", "lavender"),
    }

    for var in var_names:
        # Determine node color
        if highlight_nodes and var in highlight_nodes:
            color, fillcolor = color_map.get(highlight_color, ("red", "lightcoral"))
        else:
            color = "black"
            fillcolor = "white"

        # Use consistent node sizing
        node_size = "0.4" if consistent_sizing else "0.8"
        font_size = "10" if consistent_sizing else "14"

        node = pydot.Node(
            var,
            style="filled",
            fillcolor=fillcolor,
            color=color,
            fontname="Arial",
            fontsize=font_size,
            fontweight="bold",
            width=node_size,
            height=node_size,
        )
        graph.add_node(node)

    # Add edges
    for edge in causal_diagram.edges:
        graph.add_edge(pydot.Edge(edge[0], edge[1], color="black"))

    # Save the graph
    graph.write_png(full_path)

    return full_path


def plot_cpdag_pydot(
    cpdag_matrix: np.ndarray,
    var_names: List[str],
    filename: str,
    title: Optional[str] = None,
    figures_dir: Optional[str] = None,
    consistent_sizing: bool = False,
) -> str:
    """
    Plot a CPDAG from adjacency matrix using pydot.

    Parameters:
    -----------
    cpdag_matrix : np.ndarray
        CPDAG adjacency matrix
    var_names : List[str]
        Variable names for the nodes
    filename : str
        Base filename (without extension)
    title : str, optional
        Title for the plot
    figures_dir : str, optional
        Directory to save figures in (defaults to ./figures/)

    Returns:
    --------
    str
        Full path to the saved PNG file
    """
    # Set up directory
    if figures_dir is None:
        figures_dir = ensure_figures_directory("figures")

    full_path = os.path.join(figures_dir, f"{filename}.png")

    # Create pydot graph
    graph = pydot.Dot(graph_type="graph")  # Undirected for CPDAG
    graph.set_rankdir("TB")  # Top to bottom layout
    graph.set_bgcolor("white")

    # Set consistent sizing for combined visualizations
    if consistent_sizing:
        graph.set_size("2,2!")  # Force smaller, consistent size
        graph.set_dpi(300)
        graph.set_margin(0.05)  # Minimal margins
        graph.set_ranksep(0.3)  # Reduce spacing between ranks
        graph.set_nodesep(0.3)  # Reduce spacing between nodes

    # Add nodes
    for var in var_names:
        # Use consistent node sizing
        node_size = "0.4" if consistent_sizing else "0.8"
        font_size = "10" if consistent_sizing else "14"

        node = pydot.Node(
            var,
            style="filled",
            fillcolor="white",
            color="black",
            fontname="Arial",
            fontsize=font_size,
            fontweight="bold",
            width=node_size,
            height=node_size,
        )
        graph.add_node(node)

    # Add edges (undirected for CPDAG)
    n = len(var_names)
    for i in range(n):
        for j in range(i + 1, n):
            # Check for undirected edges (causal-learn uses -1 for undirected)
            if (cpdag_matrix[i, j] == -1 and cpdag_matrix[j, i] == -1) or (
                cpdag_matrix[i, j] == 1 and cpdag_matrix[j, i] == 1
            ):
                # Undirected edge
                graph.add_edge(pydot.Edge(var_names[i], var_names[j], color="black"))
            elif (cpdag_matrix[i, j] == 1 and cpdag_matrix[j, i] == 0) or (
                cpdag_matrix[i, j] == -1 and cpdag_matrix[j, i] == 0
            ):
                # Directed edge i -> j (convert to directed graph for this case)
                if not any(
                    e.get_source() == var_names[i]
                    and e.get_destination() == var_names[j]
                    for e in graph.get_edges()
                ):
                    graph.add_edge(
                        pydot.Edge(var_names[i], var_names[j], color="black")
                    )

    # Save the graph
    graph.write_png(full_path)

    return full_path


def get_pomis_for_dag(dag_matrix: np.ndarray, var_names: List[str], Y: str):
    """
    Get POMIS sets for a specific DAG.

    Parameters:
    -----------
    dag_matrix : np.ndarray
        DAG adjacency matrix
    var_names : List[str]
        Variable names
    Y : str
        Target variable for POMIS analysis

    Returns:
    --------
    List[List[str]]
        List of POMIS sets, each containing variable names
    """
    try:
        from npsem.causal_diagram_utils import dagmatrix_to_CausalDiagram
        from npsem.where_do import POMISs

        temp_g = dagmatrix_to_CausalDiagram(dag_matrix, var_names)
        pomis_sets = POMISs(temp_g, Y)

        # Convert frozensets to sorted lists for display
        return [sorted(list(s)) if s else [] for s in pomis_sets]
    except Exception:
        return []


def get_mis_for_dag(dag_matrix: np.ndarray, var_names: List[str], Y: str):
    """
    Get MIS sets for a specific DAG.

    Parameters:
    -----------
    dag_matrix : np.ndarray
        DAG adjacency matrix
    var_names : List[str]
        Variable names
    Y : str
        Target variable for MIS analysis

    Returns:
    --------
    List[List[str]]
        List of MIS sets, each containing variable names
    """
    try:
        from npsem.causal_diagram_utils import dagmatrix_to_CausalDiagram
        from npsem.where_do import MISs

        temp_g = dagmatrix_to_CausalDiagram(dag_matrix, var_names)
        mis_sets = MISs(temp_g, Y)

        # Convert frozensets to sorted lists for display
        return [sorted(list(s)) if s else [] for s in mis_sets]
    except Exception:
        return []


def create_combined_sanity_check_visualization(
    ground_truth_scm,
    cpdag_matrix: np.ndarray,
    dags: List[np.ndarray],
    var_names: List[str],
    Y: str,
    figures_dir: Optional[str] = None,
) -> str:
    """
    Create a single comprehensive visualization showing all sanity check results.

    Parameters:
    -----------
    ground_truth_scm : StructuralCausalModel
        Ground truth SCM
    cpdag_matrix : np.ndarray
        Discovered CPDAG adjacency matrix
    dags : List[np.ndarray]
        List of enumerated DAGs
    var_names : List[str]
        Variable names
    Y : str
        Target variable for POMIS analysis
    figures_dir : str, optional
        Directory to save figures in (defaults to ./figures/)

    Returns:
    --------
    str
        Path to the saved combined visualization
    """
    from npsem.causal_diagram_utils import dagmatrix_to_CausalDiagram

    # Set up directory
    if figures_dir is None:
        figures_dir = "figures"
    figures_dir = ensure_figures_directory(figures_dir)

    # Create individual plots first with consistent sizing
    ground_truth_path = plot_causal_diagram_pydot(
        ground_truth_scm.G,
        var_names,
        "temp_ground_truth",
        "Ground Truth",
        figures_dir=figures_dir,
        consistent_sizing=True,
    )

    cpdag_path = plot_cpdag_pydot(
        cpdag_matrix,
        var_names,
        "temp_cpdag",
        "Discovered CPDAG",
        figures_dir=figures_dir,
        consistent_sizing=True,
    )

    # Create DAG plots
    dag_paths = []
    for i, dag in enumerate(dags):
        temp_g = dagmatrix_to_CausalDiagram(dag, var_names)
        dag_path = plot_causal_diagram_pydot(
            temp_g,
            var_names,
            f"temp_dag_{i + 1}",
            f"DAG {i + 1}",
            figures_dir=figures_dir,
            consistent_sizing=True,
        )
        dag_paths.append(dag_path)

    # Create POMIS plots
    pomis_paths = []
    for i, dag in enumerate(dags):
        temp_g = dagmatrix_to_CausalDiagram(dag, var_names)
        pomis_sets = get_pomis_for_dag(dag, var_names, Y)
        # Highlight all variables that appear in any POMIS
        pomis_vars = list(set([var for pomis_set in pomis_sets for var in pomis_set]))
        pomis_label = ", ".join([str(s) if s else "∅" for s in pomis_sets])
        pomis_path = plot_causal_diagram_pydot(
            temp_g,
            var_names,
            f"temp_pomis_{i + 1}",
            f"DAG {i + 1} (POMIS: {pomis_label})",
            highlight_nodes=pomis_vars,
            figures_dir=figures_dir,
            consistent_sizing=True,
        )
        pomis_paths.append(pomis_path)

    # Create MIS plots
    mis_paths = []
    for i, dag in enumerate(dags):
        temp_g = dagmatrix_to_CausalDiagram(dag, var_names)
        mis_sets = get_mis_for_dag(dag, var_names, Y)
        # Highlight all variables that appear in any MIS
        mis_vars = list(set([var for mis_set in mis_sets for var in mis_set]))
        mis_label = ", ".join([str(s) if s else "∅" for s in mis_sets])
        mis_path = plot_causal_diagram_pydot(
            temp_g,
            var_names,
            f"temp_mis_{i + 1}",
            f"DAG {i + 1} (MIS: {mis_label})",
            highlight_nodes=mis_vars,
            figures_dir=figures_dir,
            consistent_sizing=True,
            highlight_color="blue",  # Use blue for MIS to distinguish from POMIS
        )
        mis_paths.append(mis_path)

    # Create combined visualization
    combined_path = os.path.join(figures_dir, "combined_sanity_check.png")

    # Calculate grid layout
    n_dags = len(dags)
    n_cols = max(3, n_dags + 2)  # Ground truth + CPDAG + DAGs + POMIS + MIS
    n_rows = 4  # Ground truth row, CPDAG row, DAGs row, POMIS row, MIS row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Adjust spacing to prevent title overlap
    plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.2)

    # Load and display images
    images = [(ground_truth_path, "Ground Truth"), (cpdag_path, "Discovered CPDAG")]

    # Add DAG images
    for i, dag_path in enumerate(dag_paths):
        images.append((dag_path, f"DAG {i + 1}"))

    # Add POMIS images
    for i, pomis_path in enumerate(pomis_paths):
        pomis_vars = get_pomis_for_dag(dags[i], var_names, Y)
        pomis_label = ", ".join([str(s) if s else "∅" for s in pomis_vars])
        images.append((pomis_path, f"DAG {i + 1} (POMIS: {pomis_label})"))

    # Add MIS images
    for i, mis_path in enumerate(mis_paths):
        mis_sets = get_mis_for_dag(dags[i], var_names, Y)
        mis_label = ", ".join([str(s) if s else "∅" for s in mis_sets])
        images.append((mis_path, f"DAG {i + 1} (MIS: {mis_label})"))

    # Display images in grid
    for idx, (image_path, title) in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols

        if row < n_rows and col < n_cols:
            axes[row, col].imshow(plt.imread(image_path))
            axes[row, col].set_title(title, fontsize=12, fontweight="bold")
            axes[row, col].axis("off")

    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= len(images):
                axes[row, col].axis("off")

    plt.suptitle(
        "Causal Discovery Sanity Check - Complete Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")

    plt.close()

    # Clean up temporary files
    temp_files = [ground_truth_path, cpdag_path] + dag_paths + pomis_paths + mis_paths
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass  # File might not exist

    return combined_path
