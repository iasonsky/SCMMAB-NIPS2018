"""Example causal diagrams and structural causal models used in experiments.

This module is typically imported by other scripts.  It defines a variety of
helper functions such as :func:`IV_SCM` or :func:`XYZWST_SCM` that construct
example causal graphs and their corresponding SCMs.  The module is not intended
to be executed directly from the command line.
"""

from collections import defaultdict

from npsem.model import CausalDiagram, StructuralCausalModel, default_P_U
from npsem.utils import rand_bw, seeded


def IV_CD(uname="U_XY", manipulable_vars=None):
    """Instrumental Variable Causal Diagram"""
    X, Y, Z = "X", "Y", "Z"
    return CausalDiagram(
        {X, Y, Z}, [(Z, X), (X, Y)], [(X, Y, uname)], manipulable_vars=manipulable_vars
    )


def IV_SCM(devised=True, seed=None):
    with seeded(seed):
        G = IV_CD()

        # parametrization for U
        if devised:
            mu1 = {
                "U_X": rand_bw(0.01, 0.2, precision=2),
                "U_Y": rand_bw(0.01, 0.2, precision=2),
                "U_Z": rand_bw(0.01, 0.99, precision=2),
                "U_XY": rand_bw(0.4, 0.6, precision=2),
            }
        else:
            mu1 = {
                "U_X": rand_bw(0.01, 0.99, precision=2),
                "U_Y": rand_bw(0.01, 0.99, precision=2),
                "U_Z": rand_bw(0.01, 0.99, precision=2),
                "U_XY": rand_bw(0.01, 0.99, precision=2),
            }

        P_U = default_P_U(mu1)

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(
            G,
            F={
                "Z": lambda v: v["U_Z"],
                "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"],
                "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"],
            },
            P_U=P_U,
            D=domains,
            more_U={"U_X", "U_Y", "U_Z"},
        )
        return M, mu1


def XYZWST(u_wx="U0", u_yz="U1", manipulable_vars=None):
    W, X, Y, Z, S, T = "W", "X", "Y", "Z", "S", "T"
    return CausalDiagram(
        {"W", "X", "Y", "Z", "S", "T"},
        [(Z, X), (X, Y), (W, Y), (S, W), (T, X), (T, Y)],
        [(X, W, u_wx), (Z, Y, u_yz)],
        manipulable_vars=manipulable_vars,
    )


def XYZW(u_wx="U0", u_yz="U1", manipulable_vars=None):
    return XYZWST(u_wx, u_yz, manipulable_vars) - {"S", "T"}


def XYZW_SCM(devised=True, seed=None):
    with seeded(seed):
        G = XYZW("U_WX", "U_YZ")

        # parametrization for U
        if devised:
            mu1 = {
                "U_WX": rand_bw(0.4, 0.6, precision=2),
                "U_YZ": rand_bw(0.4, 0.6, precision=2),
                "U_X": rand_bw(0.01, 0.1, precision=2),
                "U_Y": rand_bw(0.01, 0.1, precision=2),
                "U_Z": rand_bw(0.01, 0.1, precision=2),
                "U_W": rand_bw(0.01, 0.1, precision=2),
            }
        else:
            mu1 = {
                "U_WX": rand_bw(0.01, 0.99, precision=2),
                "U_YZ": rand_bw(0.01, 0.99, precision=2),
                "U_X": rand_bw(0.01, 0.99, precision=2),
                "U_Y": rand_bw(0.01, 0.99, precision=2),
                "U_Z": rand_bw(0.01, 0.99, precision=2),
                "U_W": rand_bw(0.01, 0.99, precision=2),
            }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(
            G,
            F={
                "W": lambda v: v["U_W"] ^ v["U_WX"],
                "Z": lambda v: v["U_Z"] ^ v["U_YZ"],
                "X": lambda v: 1 ^ v["U_X"] ^ v["Z"] ^ v["U_WX"],
                "Y": lambda v: v["U_Y"] ^ v["U_YZ"] ^ v["X"] ^ v["W"],
            },
            P_U=default_P_U(mu1),
            D=domains,
            more_U={"U_W", "U_X", "U_Y", "U_Z"},
        )
        return M, mu1


def XYZWST_SCM(devised=True, seed=None):
    with seeded(seed):
        G = XYZWST("U_WX", "U_YZ")

        # parametrization for U
        if devised:
            mu1 = {
                "U_WX": rand_bw(0.4, 0.6, precision=2),
                "U_YZ": rand_bw(0.4, 0.6, precision=2),
                "U_X": rand_bw(0.01, 0.1, precision=2),
                "U_Y": rand_bw(0.01, 0.1, precision=2),
                "U_Z": rand_bw(0.01, 0.1, precision=2),
                "U_W": rand_bw(0.01, 0.1, precision=2),
                "U_S": rand_bw(0.1, 0.9, precision=2),
                "U_T": rand_bw(0.1, 0.9, precision=2),
            }
        else:
            mu1 = {
                "U_WX": rand_bw(0.01, 0.99, precision=2),
                "U_YZ": rand_bw(0.01, 0.99, precision=2),
                "U_X": rand_bw(0.01, 0.99, precision=2),
                "U_Y": rand_bw(0.01, 0.99, precision=2),
                "U_Z": rand_bw(0.01, 0.99, precision=2),
                "U_W": rand_bw(0.01, 0.99, precision=2),
                "U_S": rand_bw(0.01, 0.99, precision=2),
                "U_T": rand_bw(0.01, 0.99, precision=2),
            }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(
            G,
            F={
                "S": lambda v: v["U_S"],
                "T": lambda v: v["U_T"],
                "W": lambda v: v["U_W"] ^ v["U_WX"] ^ v["S"],
                "Z": lambda v: v["U_Z"] ^ v["U_YZ"],
                "X": lambda v: 1 ^ v["U_X"] ^ v["Z"] ^ v["U_WX"] ^ v["T"],
                "Y": lambda v: v["U_Y"] ^ v["U_YZ"] ^ v["X"] ^ v["W"] ^ v["T"],
            },
            P_U=default_P_U(mu1),
            D=domains,
            more_U={"U_W", "U_X", "U_Y", "U_Z", "U_S", "U_T"},
        )
        return M, mu1


def simple_markovian(manipulable_vars=None):
    X1, X2, Y, Z1, Z2 = "X1", "X2", "Y", "Z1", "Z2"
    return CausalDiagram(
        {"X1", "X2", "Y", "Z1", "Z2"},
        [(X1, Y), (X2, Y), (Z1, X1), (Z1, X2), (Z2, X1), (Z2, X2)],
        manipulable_vars=manipulable_vars,
    )


def simple_markovian_SCM(seed=None) -> [StructuralCausalModel, dict]:
    with seeded(seed):
        G = simple_markovian()
        mu1 = {("U_" + v): rand_bw(0.1, 0.9, precision=2) for v in sorted(G.V)}

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(
            G,
            F={
                "Z1": lambda v: v["U_Z1"],
                "Z2": lambda v: v["U_Z2"],
                "X1": lambda v: v["U_X1"] ^ v["Z1"] ^ v["Z2"],
                "X2": lambda v: 1 ^ v["U_X2"] ^ v["Z1"] ^ v["Z2"],
                "Y": lambda v: v["U_Y"] | (v["X1"] & v["X2"]),
            },
            P_U=default_P_U(mu1),
            D=domains,
            more_U={"U_" + v for v in G.V},
        )
        return M, mu1

def four_variable_CD(manipulable_vars=None):
    """4-variable Causal Diagram from second paper
    
    Variables: A, B, C, Y
    N = {A} (A is non-manipulable)
    Bidirected edges: A ↔ B, B ↔ Y
    Directed edges: A → C, B → C, A → Y, C → Y
    """
    A, B, C, Y = "A", "B", "C", "Y"
    
    # If no manipulable_vars specified, default excludes A (since N = {A})
    if manipulable_vars is None:
        manipulable_vars = {B, C, Y}
    
    return CausalDiagram(
        {A, B, C, Y},
        [(A, C), (B, C), (A, Y), (C, Y)],  # directed edges
        [(A, B, "U_AB"), (B, Y, "U_BY")],  # bidirected edges
        manipulable_vars=manipulable_vars,
    )


def four_variable_SCM(seed=None):
    """4-variable SCM from second paper with exact parameter values
    
    Based on specifications:
    - P(U_B = 1) = 0.5, P(U_C = 1) = 0.25, P(U_Y = 1) = 0.25
    - P(U_BY = 1) = 0.25, P(U_AB = 1) = 0.4
    - f_A = u_AB
    - f_B = u_B ⊕ u_AB ⊕ u_BY  
    - f_C = u_C ⊕ a ⊕ b
    - f_Y = 1 - (u_BY ⊕ u_Y ⊕ a ⊕ c)
    """
    with seeded(seed):
        G = four_variable_CD()
        
        # Exact parameter values from the paper
        mu1 = {
            "U_B": 0.5,
            "U_C": 0.25, 
            "U_Y": 0.25,
            "U_BY": 0.25,
            "U_AB": 0.4,
        }
        
        P_U = default_P_U(mu1)
        domains = defaultdict(lambda: (0, 1))
        
        # Structural equations matching the paper exactly
        M = StructuralCausalModel(
            G,
            F={
                "A": lambda v: v["U_AB"],
                "B": lambda v: v["U_B"] ^ v["U_AB"] ^ v["U_BY"],
                "C": lambda v: v["U_C"] ^ v["A"] ^ v["B"],
                "Y": lambda v: 1 ^ (v["U_BY"] ^ v["U_Y"] ^ v["A"] ^ v["C"]),
            },
            P_U=P_U,
            D=domains,
            more_U={"U_B", "U_C", "U_Y"},
        )
        return M, mu1