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


def IV_SCM_strong(devised=True, seed=None):
    """
    Strong IV SCM adapted from IV_SCM for better FCI detection.

    Uses OR operations instead of XOR and smaller noise parameters to create
    stronger correlations that FCI algorithm can detect.

    Structure: Z -> X -> Y with confounding X <-> Y (U_XY)

    Returns
    -------
    M : StructuralCausalModel
    mu1 : dict   # P(U=1) for each exogenous variable
    """
    with seeded(seed):
        G = IV_CD()

        # Original noise parameters from IV_SCM
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

        # SCM with OR operations for stronger correlations
        M = StructuralCausalModel(
            G,
            F={
                "Z": lambda v: v["U_Z"],
                "X": lambda v: v["U_X"] | v["U_XY"] | v["Z"],
                "Y": lambda v: v["U_Y"] | v["U_XY"] | v["X"],
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


def chain_CD(manipulable_vars=None):
    """Chain Causal Diagram: Z -> X -> Y (Markovian: no bidirected edges)"""
    Z, X, Y = "Z", "X", "Y"
    return CausalDiagram(
        {Z, X, Y},
        [(Z, X), (X, Y)],
        [],  # no bidirected edges
        manipulable_vars=manipulable_vars,
    )


def chain_SCM(devised=True, seed=None):
    """
    Binary Markovian chain SCM: Z -> X -> Y

    - Z is exogenous: Z := U_Z
    - X depends on Z with small noise (when devised=True): X := Z OR U_X
    - Y depends on X with small noise: Y := X OR U_Y

    Uses OR operations instead of XOR for stronger correlations that PC algorithm can detect.

    Returns
    -------
    M : StructuralCausalModel
    mu1 : dict   # P(U=1) for each exogenous variable
    """
    with seeded(seed):
        G = chain_CD()

        # parameterization for U (keep noise very small for stronger dependencies when devised=True)
        if devised:
            mu1 = {
                "U_Z": rand_bw(0.3, 0.7, precision=2),  # Z prevalence
                "U_X": rand_bw(0.01, 0.1, precision=2),  # very small noise on X
                "U_Y": rand_bw(0.01, 0.1, precision=2),  # very small noise on Y
            }
        else:
            mu1 = {
                "U_Z": rand_bw(0.01, 0.99, precision=2),
                "U_X": rand_bw(0.01, 0.99, precision=2),
                "U_Y": rand_bw(0.01, 0.99, precision=2),
            }

        P_U = default_P_U(mu1)
        domains = defaultdict(lambda: (0, 1))

        M = StructuralCausalModel(
            G,
            F={
                "Z": lambda v: v["U_Z"],
                "X": lambda v: v["Z"]
                | v["U_X"],  # X = Z ∨ U_X (OR for stronger correlation)
                "Y": lambda v: v["X"]
                | v["U_Y"],  # Y = X ∨ U_Y (OR for stronger correlation)
            },
            P_U=P_U,
            D=domains,
            more_U={"U_Z", "U_X", "U_Y"},
        )
        return M, mu1


def frontdoor_CD(manipulable_vars=None):
    """Frontdoor Causal Diagram

    Variables: X, Y, Z
    Observed variables: N = {Z}
    Directed edges: X -> Z, Z -> Y
    Bidirected edge: X <-> Y (confounding)
    """
    X, Y, Z = "X", "Y", "Z"

    # If no manipulable_vars specified, default excludes Z (since N = {Z})
    if manipulable_vars is None:
        manipulable_vars = {X, Y}

    return CausalDiagram(
        {X, Y, Z},
        [(X, Z), (Z, Y)],  # directed edges
        [(X, Y, "U_XY")],  # bidirected edge for confounding
        manipulable_vars=manipulable_vars,
    )


def frontdoor_SCM(seed=None):
    """Frontdoor SCM with exact parameter values from specification

    Based on frontdoor graph specification:
    - P(U_X = 1) = 0.5, P(U_Y = 1) = 0.4, P(U_Z = 1) = 0.4, P(U_XY = 1) = 0.5
    - f_X = u_X XOR u_XY
    - f_Z = u_Z XOR x
    - f_Y = (u_Y AND u_XY) XOR z

    Returns
    -------
    M : StructuralCausalModel
    mu1 : dict   # P(U=1) for each exogenous variable
    """
    with seeded(seed):
        G = frontdoor_CD()

        # Exact parameter values from the specification
        mu1 = {
            "U_X": 0.5,
            "U_Y": 0.4,
            "U_Z": 0.4,
            "U_XY": 0.5,
        }

        P_U = default_P_U(mu1)
        domains = defaultdict(lambda: (0, 1))

        # Structural equations matching the specification exactly
        M = StructuralCausalModel(
            G,
            F={
                "X": lambda v: v["U_X"] ^ v["U_XY"],  # f_X = u_X XOR u_XY
                "Z": lambda v: v["U_Z"] ^ v["X"],  # f_Z = u_Z XOR x
                "Y": lambda v: (v["U_Y"] & v["U_XY"])
                ^ v["Z"],  # f_Y = (u_Y AND u_XY) XOR z
            },
            P_U=P_U,
            D=domains,
            more_U={"U_X", "U_Y", "U_Z"},
        )
        return M, mu1


def six_variable_CD(manipulable_vars=None):
    """6-variable Causal Diagram

    Variables: A, B, C, D, E, Y
    Observed variables: N = {A, C}
    Directed edges: B → C, C → D, A → E, C → E, D → Y, E → Y
    Bidirected edges: A ↔ Y (U_AY), B ↔ Y (U_BY)
    """
    A, B, C, D, E, Y = "A", "B", "C", "D", "E", "Y"

    # If no manipulable_vars specified, default excludes A and C (since N = {A, C})
    if manipulable_vars is None:
        manipulable_vars = {B, D, E, Y}

    return CausalDiagram(
        {A, B, C, D, E, Y},
        [(B, C), (C, D), (A, E), (C, E), (D, Y), (E, Y)],  # directed edges
        [(A, Y, "U_AY"), (B, Y, "U_BY")],  # bidirected edges for confounding
        manipulable_vars=manipulable_vars,
    )


def six_variable_SCM(seed=None):
    """6-variable SCM with exact parameter values from specification

    Based on 6-variable graph specification:
    - P(U_A = 1) = 0.22, P(U_B = 1) = 0.2, P(U_C = 1) = 0.12, P(U_D = 1) = 0.2
    - P(U_E = 1) = 0.04, P(U_Y = 1) = 0.87, P(U_BY = 1) = 0.04, P(U_AY = 1) = 0.45
    - f_A = u_A XOR u_AY
    - f_B = u_B XOR u_BY
    - f_C = u_C XOR b
    - f_D = u_D XOR c
    - f_E = u_E XOR a XOR c
    - f_Y = 1 - (u_Y XOR u_AY XOR u_BY XOR d XOR e)

    Returns
    -------
    M : StructuralCausalModel
    mu1 : dict   # P(U=1) for each exogenous variable
    """
    with seeded(seed):
        G = six_variable_CD()

        # Exact parameter values from the specification
        mu1 = {
            "U_A": 0.22,
            "U_B": 0.2,
            "U_C": 0.12,
            "U_D": 0.2,
            "U_E": 0.04,
            "U_Y": 0.87,
            "U_BY": 0.04,
            "U_AY": 0.45,
        }

        P_U = default_P_U(mu1)
        domains = defaultdict(lambda: (0, 1))

        # Structural equations matching the specification exactly
        M = StructuralCausalModel(
            G,
            F={
                "A": lambda v: v["U_A"] ^ v["U_AY"],  # f_A = u_A XOR u_AY
                "B": lambda v: v["U_B"] ^ v["U_BY"],  # f_B = u_B XOR u_BY
                "C": lambda v: v["U_C"] ^ v["B"],  # f_C = u_C XOR b
                "D": lambda v: v["U_D"] ^ v["C"],  # f_D = u_D XOR c
                "E": lambda v: v["U_E"] ^ v["A"] ^ v["C"],  # f_E = u_E XOR a XOR c
                "Y": lambda v: 1
                ^ (
                    v["U_Y"] ^ v["U_AY"] ^ v["U_BY"] ^ v["D"] ^ v["E"]
                ),  # f_Y = 1 - (u_Y XOR u_AY XOR u_BY XOR d XOR e)
            },
            P_U=P_U,
            D=domains,
            more_U={"U_A", "U_B", "U_C", "U_D", "U_E", "U_Y"},
        )
        return M, mu1
