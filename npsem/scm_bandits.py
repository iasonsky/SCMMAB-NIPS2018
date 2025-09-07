"""Helpers for turning structural causal models into bandit problems."""

from itertools import product

from typing import Dict, Tuple, Union, Any

from npsem.model import StructuralCausalModel
from npsem.utils import combinations
from npsem.where_do import POMISs, MISs
from npsem.latent_projection import latent_projection


def SCM_to_bandit_machine(
    M: StructuralCausalModel, Y="Y"
) -> Tuple[Tuple, Dict[Union[int, Any], Dict]]:
    """Convert an SCM into a set of bandit arms.

    Parameters
    ----------
    M : StructuralCausalModel
        Model from which expected rewards are computed.
    Y : str, default ``'Y'``
        Outcome variable used as reward.

    Returns
    -------
    mu : tuple
        Expected reward of each arm.
    arm_setting : dict
        Mapping from arm index to intervention dictionary.
    """
    G = M.G
    mu_arm = list()
    arm_setting = dict()
    arm_id = 0
    # Only consider manipulable variables (excluding the reward variable Y)
    manipulable_vars = G.manipulable_vars - {Y}
    all_subsets = list(combinations(sorted(manipulable_vars)))
    for subset in all_subsets:
        for values in product(*[M.D[variable] for variable in subset]):
            arm_setting[arm_id] = dict(zip(subset, values))

            result = M.query((Y,), intervention=arm_setting[arm_id])
            expectation = sum(y_val * result[(y_val,)] for y_val in M.D[Y])
            mu_arm.append(expectation)
            arm_id += 1

    return tuple(mu_arm), arm_setting


def arm_types():
    return ["POMIS", "MIS", "Brute-force", "All-at-once"]


def arms_of(arm_type: str, arm_setting, G, Y) -> Tuple[int, ...]:
    """Return the indices of arms of a given type."""
    if arm_type == "POMIS":
        return pomis_arms_of(arm_setting, G, Y)
    elif arm_type == "All-at-once":
        return controlphil_arms_of(arm_setting, G, Y)
    elif arm_type == "MIS":
        return mis_arms_of(arm_setting, G, Y)
    elif arm_type == "Brute-force":
        return tuple(range(len(arm_setting)))
    raise AssertionError(f"unknown: {arm_type}")


def pomis_arms_of(arm_setting, G, Y):
    """Indices of arms that correspond to POMIS interventions."""
    to_remove = G.V - G.manipulable_vars
    if to_remove != frozenset():
        G = latent_projection(G, to_remove)
    pomiss = POMISs(G, Y)
    return tuple(
        arm_x for arm_x in range(len(arm_setting)) if set(arm_setting[arm_x]) in pomiss
    )


def mis_arms_of(arm_setting, G, Y):
    """Indices of arms that correspond to MIS interventions."""
    miss = MISs(G, Y)
    return tuple(
        arm_x for arm_x in range(len(arm_setting)) if set(arm_setting[arm_x]) in miss
    )


def controlphil_arms_of(arm_setting, G, Y):
    """Index of the arm that intervenes on all manipulable variables at once."""
    intervenable = G.manipulable_vars - {Y}
    return tuple(
        arm_x
        for arm_x in range(len(arm_setting))
        if arm_setting[arm_x].keys() == intervenable
    )
