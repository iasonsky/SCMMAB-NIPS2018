"""Registry for SCM configurations and experiment settings.

This module provides a centralized registry for all SCMs and their
experiment configurations, making it easy to add new SCMs and run
experiments consistently.
"""

from typing import Dict, Any, Callable, Tuple
from npsem.model import StructuralCausalModel
from npsem.NIPS2018POMIS_exp.scm_examples import (
    four_variable_SCM,
    frontdoor_SCM, 
    six_variable_SCM,
    chain_SCM,
    IV_SCM,
    XYZW_SCM,
    XYZWST_SCM,
    simple_markovian_SCM,
)


class SCMRegistry:
    """Registry for SCM configurations and experiment settings."""
    
    def __init__(self):
        self._scms = {}
        self._default_config = {
            'target_variable': 'Y',
            'num_trials': 200,
            'horizon': 10000,
            'arm_strategies': ['POMIS', 'MIS', 'Brute-force'],
            'bandit_algorithms': ['TS', 'UCB'],
        }
        
    def register_scm(
        self, 
        name: str, 
        scm_factory: Callable[[], Tuple[StructuralCausalModel, Dict[str, float]]],
        display_name: str = None,
        description: str = "",
        config: Dict[str, Any] = None
    ):
        """Register an SCM with its configuration.
        
        Parameters
        ----------
        name : str
            Internal name for the SCM
        scm_factory : callable
            Function that returns (model, p_u) when called
        display_name : str, optional
            Human-readable name. If None, uses name
        description : str
            Description of the SCM
        config : dict, optional
            Experiment configuration overrides
        """
        if config is None:
            config = {}
            
        self._scms[name] = {
            'factory': scm_factory,
            'display_name': display_name or name,
            'description': description,
            'config': {**self._default_config, **config}
        }
        
    def get_scm(self, name: str) -> Dict[str, Any]:
        """Get SCM configuration by name."""
        if name not in self._scms:
            raise ValueError(f"SCM '{name}' not found. Available: {list(self._scms.keys())}")
        return self._scms[name]
        
    def list_scms(self) -> Dict[str, str]:
        """List all registered SCMs with their descriptions."""
        return {name: info['description'] for name, info in self._scms.items()}
        
    def get_all_scm_names(self) -> list:
        """Get list of all registered SCM names."""
        return list(self._scms.keys())


# Create global registry instance
registry = SCMRegistry()

# Register all available SCMs
registry.register_scm(
    'four_variable',
    four_variable_SCM,
    'Four Variable SCM',
    '4-variable causal diagram from second paper with exact parameter values',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'frontdoor',
    frontdoor_SCM,
    'Frontdoor SCM', 
    'Frontdoor causal diagram with X->Z->Y and confounding X<->Y',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'six_variable',
    six_variable_SCM,
    'Six Variable SCM',
    '6-variable causal diagram with complex confounding relationships',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'chain',
    chain_SCM,
    'Chain SCM',
    'Simple Markovian chain Z->X->Y with no confounding',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'iv',
    IV_SCM,
    'Instrumental Variable SCM',
    'Instrumental variable setup with Z->X->Y and confounding X<->Y',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'xyzw',
    XYZW_SCM,
    'XYZW SCM',
    '4-variable SCM with W->Y, Z->X, and confounding relationships',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'xyzwst',
    XYZWST_SCM,
    'XYZWST SCM',
    '6-variable SCM with additional S and T variables',
    {'target_variable': 'Y'}
)

registry.register_scm(
    'simple_markovian',
    simple_markovian_SCM,
    'Simple Markovian SCM',
    '5-variable Markovian SCM with no confounding',
    {'target_variable': 'Y'}
)


def get_scm_factory(name: str) -> Callable[[], Tuple[StructuralCausalModel, Dict[str, float]]]:
    """Get SCM factory function by name."""
    return registry.get_scm(name)['factory']


def get_scm_config(name: str) -> Dict[str, Any]:
    """Get SCM experiment configuration by name."""
    return registry.get_scm(name)['config']


def list_available_scms() -> Dict[str, str]:
    """List all available SCMs with descriptions."""
    return registry.list_scms()
