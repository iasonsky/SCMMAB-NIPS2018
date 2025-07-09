"""Computation of minimal intervention sets for causal bandits.

This module implements algorithms from the NIPS 2018 paper for computing
POMIS (Probability of Minimal Intervention Set) and MIS (Minimal Intervention
Set) families on a given :class:`~npsem.model.CausalDiagram`.  These routines
are used to restrict the set of arms in the bandit experiments.
"""

from typing import Set, List, Tuple, FrozenSet, AbstractSet

from npsem.model import CausalDiagram
from npsem.utils import pop, only, combinations


def CC(G: CausalDiagram, X: str):
    """Return the c-component of ``G`` that contains ``X``."""
    return G.c_component(X)


def MISs(G: CausalDiagram, Y: str) -> FrozenSet[FrozenSet[str]]:
    """Compute all Minimal Intervention Sets (MIS) for ``Y``.

    Parameters
    ----------
    G : CausalDiagram
        Diagram in which interventions are considered.
    Y : str
        Target outcome variable.

    Returns
    -------
    FrozenSet[FrozenSet[str]]
        Collection of all MISs represented as frozensets of variable names.
    """
    II = G.V - {Y}
    assert II <= G.V
    assert Y not in II

    G = G[G.An(Y)]
    Ws = G.causal_order(backward=True)
    Ws = only(Ws, II)
    return subMISs(G, Y, frozenset(), Ws)


def subMISs(G: CausalDiagram, Y: str, Xs: FrozenSet[str], Ws: List[str]) -> FrozenSet[FrozenSet[str]]:
    """Recursive helper used by :func:`MISs`."""
    out = frozenset({Xs})
    for i, W_i in enumerate(Ws):
        H = G.do({W_i})
        H = H[H.An(Y)]
        out |= subMISs(H, Y, Xs | {W_i}, only(Ws[i + 1:], H.V))
    return out


def bruteforce_POMISs(G: CausalDiagram, Y: str) -> FrozenSet[FrozenSet[str]]:
    """Compute all POMISs by exhaustive search."""
    return frozenset({frozenset(IB(G.do(Ws), Y))
                      for Ws in combinations(list(G.V - {Y}))})


def MUCT(G: CausalDiagram, Y: str) -> FrozenSet[str]:
    """Minimal Unobserved Confounder's Territory for ``Y``."""
    H = G[G.An(Y)]

    Qs = {Y}
    Ts = frozenset({Y})
    while Qs:
        Q1 = pop(Qs)
        Ws = CC(H, Q1)
        Ts |= Ws
        Qs = (Qs | H.de(Ws)) - Ts

    return Ts


def IB(G: CausalDiagram, Y: str) -> FrozenSet[str]:
    """Interventional Border for ``Y``."""
    Zs = MUCT(G, Y)
    return G.pa(Zs) - Zs


def MUCT_IB(G: CausalDiagram, Y) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    Zs = MUCT(G, Y)
    return Zs, G.pa(Zs) - Zs


def POMISs(G: CausalDiagram, Y: str) -> Set[FrozenSet[str]]:
    """Compute all Probability-One Minimal Intervention Sets (POMISs)."""
    G = G[G.An(Y)]

    Ts, Xs = MUCT_IB(G, Y)
    H = G.do(Xs)[Ts | Xs]
    return subPOMISs(H, Y, only(H.causal_order(backward=True), Ts - {Y})) | {frozenset(Xs)}


def subPOMISs(G: CausalDiagram, Y, Ws: List, obs=None) -> Set[FrozenSet[str]]:
    if obs is None:
        obs = set()

    """Recursive helper used by :func:`POMISs`."""

    out = []
    for i, W_i in enumerate(Ws):
        Ts, Xs = MUCT_IB(G.do({W_i}), Y)
        new_obs = obs | set(Ws[:i])
        if not (Xs & new_obs):
            out.append(Xs)
            new_Ws = only(Ws[i + 1:], Ts)
            if new_Ws:
                out.extend(subPOMISs(G.do(Xs)[Ts | Xs], Y, new_Ws, new_obs))
    return {frozenset(_) for _ in out}


def minimal_do(G: CausalDiagram, Y: str, Xs: AbstractSet[str]) -> FrozenSet[str]:
    """Remove redundant interventions from ``Xs`` with respect to ``Y``."""
    return frozenset(Xs & G.do(Xs).An(Y))
