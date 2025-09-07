"""Latent projection via Ananke's ADMG implementation.

This module uses Ananke for correctness. If Ananke is not available,
an informative ImportError is raised.
"""

from typing import Set

from npsem.model import CausalDiagram


# Optional Ananke dependency
try:
    from ananke.graphs.admg import ADMG  # type: ignore

    _ANANKE_AVAILABLE = True
except Exception:  # pragma: no cover - purely environmental
    _ANANKE_AVAILABLE = False


def latent_projection(G: CausalDiagram, variables_to_remove: Set[str]) -> CausalDiagram:
    """Compute latent projection using Ananke.

    Parameters
    ----------
    G : CausalDiagram
        Input causal diagram.
    variables_to_remove : Set[str]
        Variables to marginalize out.

    Returns
    -------
    CausalDiagram
        Projected diagram over retained vertices.
    """
    if not variables_to_remove or variables_to_remove.isdisjoint(G.V):
        return G

    if not _ANANKE_AVAILABLE:
        raise ImportError(
            "ananke-causal is required for latent_projection. Install it (e.g., with uv: `uv add ananke-causal`)"
        )

    admg = _to_ananke_admg(G)
    retain = [
        v for v in G.V if (v not in variables_to_remove and not str(v).startswith("U_"))
    ]
    projected = admg.latent_projection(retain)
    return _from_ananke_admg(projected, G.manipulable_vars)


def _to_ananke_admg(G: CausalDiagram) -> "ADMG":
    """Convert internal CausalDiagram to Ananke ADMG."""
    if not _ANANKE_AVAILABLE:
        raise ImportError("ananke-causal is not available")

    # Only include observed vertices (exclude synthetic U_ variables)
    vertices = [v for v in G.V if not str(v).startswith("U_")]

    # Directed edges as list of tuples
    di_edges = [
        (x, y)
        for x, y in G.edges
        if not str(x).startswith("U_") and not str(y).startswith("U_")
    ]

    # Bidirected edges as list of tuples
    bi_edges = [
        (x, y)
        for x, y, _u in G.confounded_to_3tuples()
        if not str(x).startswith("U_") and not str(y).startswith("U_")
    ]

    return ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)


def _from_ananke_admg(admg: "ADMG", manipulable_vars: Set[str]) -> CausalDiagram:
    """Convert Ananke ADMG back into our CausalDiagram."""
    # Vertices (observed only)
    vertices = {str(v) for v in admg.vertices if not str(v).startswith("U_")}

    # Directed edges
    directed_edges = set()
    for src, dst in getattr(admg, "di_edges", set()):
        if not str(src).startswith("U_") and not str(dst).startswith("U_"):
            directed_edges.add((str(src), str(dst)))

    # Bidirected edges (de-duplicate undirected pairs)
    seen = set()
    bidirected_edges = set()
    for a, b in getattr(admg, "bi_edges", set()):
        if str(a).startswith("U_") or str(b).startswith("U_"):
            continue
        key = tuple(sorted((str(a), str(b))))
        if key in seen:
            continue
        seen.add(key)
        conf_name = f"U_{key[0]}_{key[1]}"
        bidirected_edges.add((key[0], key[1], conf_name))

    return CausalDiagram(
        vs=vertices,
        directed_edges=directed_edges,
        bidirected_edges=bidirected_edges,
        manipulable_vars=manipulable_vars & vertices if manipulable_vars else vertices,
    )
