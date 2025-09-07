"""Demo script to visualize latent projection results.

This script shows the original graph and the results of latent projection
when marginalizing A, B, and C respectively.
"""
from npsem.model import CausalDiagram
from npsem.latent_projection import latent_projection


def build_base_graph() -> CausalDiagram:
    """Construct the base graph used in the demo."""
    directed_edges = {
        ("B", "C"),
        ("C", "Y"),
        ("A", "C"),
        ("A", "Y"),
    }
    bidirected_edges = {
        ("B", "A", "U_A_B"),
        ("B", "Y", "U_B_Y"),
    }
    return CausalDiagram(
        vs={"A", "B", "C", "Y"},
        directed_edges=directed_edges,
        bidirected_edges=bidirected_edges,
    )


def print_graph_info(G: CausalDiagram, title: str):
    """Print graph information in a readable format."""
    print(f"\n=== {title} ===")
    print(f"Vertices: {sorted(G.V)}")
    print(f"Directed edges: {sorted(G.edges)}")
    print(f"Confounders: {sorted([(x, y) for x, y, _ in G.confounded_to_3tuples()])}")


def main():
    print("🔍 Latent Projection Demo")
    print("=" * 50)

    # Build the original graph
    G = build_base_graph()
    print_graph_info(G, "Original Graph G")
    print("Structure:")
    print("  B -> C -> Y")
    print("  A -> C")
    print("  A -> Y")
    print("  B <-> A (confounder)")
    print("  B <-> Y (confounder)")

    # Test marginalizing A
    print("\n" + "=" * 50)
    projected_A = latent_projection(G, {"A"})
    print_graph_info(projected_A, "Marginalize A")

    # Test marginalizing B
    print("\n" + "=" * 50)
    projected_B = latent_projection(G, {"B"})
    print_graph_info(projected_B, "Marginalize B")

    # Test marginalizing C
    print("\n" + "=" * 50)
    projected_C = latent_projection(G, {"C"})
    print_graph_info(projected_C, "Marginalize C")

    print("\n" + "=" * 50)
    print("✅ Demo complete!")


if __name__ == "__main__":
    main()
