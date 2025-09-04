"""Instrumented version of POMIS algorithm for studying topological ordering effects.

This module provides instrumented versions of the POMIS algorithms that collect
metrics about recursive calls, IB evaluations, pruning, and timing to study
how different topological orderings affect algorithm performance.
"""

import time
from dataclasses import dataclass, field
from typing import Set, List, Tuple, FrozenSet, Dict, Any, Optional
import random

from npsem.model import CausalDiagram
from npsem.utils import only
from npsem.where_do import MUCT_IB


@dataclass
class POMISMetrics:
    """Metrics collected during POMIS computation."""

    # Core metrics
    total_subpomis_calls: int = 0
    total_ib_evaluations: int = 0
    total_pruned_branches: int = 0
    computation_time: float = 0.0

    # Detailed tracking
    recursion_depths: List[int] = field(default_factory=list)
    call_stack_sizes: List[int] = field(default_factory=list)

    # Result verification
    final_pomis_set: Optional[Set[FrozenSet[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for CSV export."""
        return {
            "total_subpomis_calls": self.total_subpomis_calls,
            "total_ib_evaluations": self.total_ib_evaluations,
            "total_pruned_branches": self.total_pruned_branches,
            "computation_time": self.computation_time,
            "max_recursion_depth": max(self.recursion_depths)
            if self.recursion_depths
            else 0,
            "mean_recursion_depth": sum(self.recursion_depths)
            / len(self.recursion_depths)
            if self.recursion_depths
            else 0,
            "final_pomis_set_size": len(self.final_pomis_set)
            if self.final_pomis_set
            else 0,
            "final_pomis_set_hash": hash(frozenset(self.final_pomis_set))
            if self.final_pomis_set
            else 0,
        }


class InstrumentedPOMISComputer:
    """Instrumented version of POMIS computation with different topological orderings."""

    def __init__(self):
        self.metrics = POMISMetrics()
        self._current_depth = 0

    def reset_metrics(self):
        """Reset metrics for a new computation."""
        self.metrics = POMISMetrics()
        self._current_depth = 0

    def get_all_topological_orders(self, G: CausalDiagram) -> List[Tuple[str, ...]]:
        """Generate all possible topological orderings for a given graph.

        For large graphs, this returns a random sample to avoid exponential explosion.
        """
        import networkx as nx

        # Create NetworkX graph
        gg = nx.DiGraph(G.edges)
        gg.add_nodes_from(G.V)

        # Get all topological sorts (this can be expensive for large graphs)
        try:
            all_orders = list(nx.all_topological_sorts(gg))

            # If too many orderings, sample randomly
            if len(all_orders) > 50:  # Reasonable limit
                random.shuffle(all_orders)
                all_orders = all_orders[:50]

            return [tuple(reversed(order)) for order in all_orders]  # backward=True

        except MemoryError:
            # Fallback: generate some random permutations of a valid topological sort
            base_order = tuple(reversed(list(nx.topological_sort(gg))))
            orders = [base_order]

            # Generate some random variations
            for _ in range(min(20, len(G.V) * 2)):
                order_list = list(base_order)
                # Randomly swap adjacent elements that don't violate topological constraints
                for _ in range(len(order_list) // 2):
                    i = random.randint(0, len(order_list) - 2)
                    # Check if swap is valid (no dependency violation)
                    if order_list[i + 1] not in G.de({order_list[i]}):
                        order_list[i], order_list[i + 1] = (
                            order_list[i + 1],
                            order_list[i],
                        )
                orders.append(tuple(order_list))

            return list(set(orders))  # Remove duplicates

    def compute_pomis_with_order(
        self, G: CausalDiagram, Y: str, custom_order: Optional[Tuple[str, ...]] = None
    ) -> Set[FrozenSet[str]]:
        """Compute POMISs using a specific topological ordering."""
        self.reset_metrics()
        start_time = time.time()

        # Temporarily override causal_order if custom order provided
        if custom_order is not None:
            original_causal_order = G.causal_order
            G.causal_order = (
                lambda backward=False: custom_order
                if backward
                else tuple(reversed(custom_order))
            )

        try:
            result = self._instrumented_pomis(G, Y)
            self.metrics.final_pomis_set = result
        finally:
            if custom_order is not None:
                G.causal_order = original_causal_order

        self.metrics.computation_time = time.time() - start_time
        return result

    def _instrumented_pomis(self, G: CausalDiagram, Y: str) -> Set[FrozenSet[str]]:
        """Instrumented version of POMISs function."""
        G = G[G.An(Y)]

        Ts, Xs = MUCT_IB(G, Y)
        self.metrics.total_ib_evaluations += 1

        H = G.do(Xs)[Ts | Xs]
        order = only(H.causal_order(backward=True), Ts - {Y})

        result = self._instrumented_subpomis(H, Y, order) | {frozenset(Xs)}
        return result

    def _instrumented_subpomis(
        self, G: CausalDiagram, Y: str, Ws: List[str], obs=None
    ) -> Set[FrozenSet[str]]:
        """Instrumented version of subPOMISs function."""
        if obs is None:
            obs = set()

        # Track recursion metrics
        self.metrics.total_subpomis_calls += 1
        self._current_depth += 1
        self.metrics.recursion_depths.append(self._current_depth)
        self.metrics.call_stack_sizes.append(len(Ws))

        try:
            out = []
            for i, W_i in enumerate(Ws):
                Ts, Xs = MUCT_IB(G.do({W_i}), Y)
                self.metrics.total_ib_evaluations += 1

                new_obs = obs | set(Ws[:i])
                if not (Xs & new_obs):
                    out.append(Xs)
                    new_Ws = only(Ws[i + 1 :], Ts)
                    if new_Ws:
                        result = self._instrumented_subpomis(
                            G.do(Xs)[Ts | Xs], Y, new_Ws, new_obs
                        )
                        out.extend(result)
                else:
                    # This branch was pruned due to overlap
                    self.metrics.total_pruned_branches += 1

            return {frozenset(_) for _ in out}

        finally:
            self._current_depth -= 1


def generate_topological_orders_strategies(
    G: CausalDiagram,
) -> Dict[str, Tuple[str, ...]]:
    """Generate different topological ordering strategies for comparison.

    Returns a dictionary mapping strategy names to topological orders.
    """
    import networkx as nx

    # Create NetworkX graph
    gg = nx.DiGraph(G.edges)
    gg.add_nodes_from(G.V)

    strategies = {}

    # 1. Default NetworkX topological sort (lexicographic)
    base_order = list(nx.topological_sort(gg))
    strategies["default"] = tuple(reversed(base_order))

    # 2. Random topological sort
    nodes_shuffled = list(G.V)
    random.shuffle(nodes_shuffled)
    gg_shuffled = nx.DiGraph(G.edges)
    gg_shuffled.add_nodes_from(nodes_shuffled)
    random_order = list(nx.topological_sort(gg_shuffled))
    strategies["random"] = tuple(reversed(random_order))

    # 3. Depth-first based order
    try:
        dfs_order = list(nx.dfs_preorder_nodes(gg))
        # Ensure it's still topologically valid
        if nx.is_directed_acyclic_graph(gg):
            # Check if DFS order is topologically valid
            pos = {node: i for i, node in enumerate(dfs_order)}
            valid = all(pos[u] < pos[v] for u, v in gg.edges())
            if valid:
                strategies["dfs"] = tuple(reversed(dfs_order))
    except Exception:
        pass

    # 4. Breadth-first based order
    try:
        # Find nodes with no incoming edges as starting points
        sources = [n for n in gg.nodes() if gg.in_degree(n) == 0]
        if sources:
            bfs_order = []
            for source in sources:
                bfs_order.extend(nx.bfs_tree(gg, source).nodes())
            # Remove duplicates while preserving order
            seen = set()
            bfs_unique = []
            for node in bfs_order:
                if node not in seen:
                    bfs_unique.append(node)
                    seen.add(node)
            strategies["bfs"] = tuple(reversed(bfs_unique))
    except Exception:
        pass

    # 5. Reverse of default
    strategies["reverse"] = tuple(base_order)

    return strategies
