"""JAX-based opinion dynamics for differentiable inference."""

from .opinion_dynamics_jax import (
    simulate_scenario,
    simulate_scenario_jit,
    scenario_loss,
    ScenarioData,
    build_sparse_interaction,
)
from .param_utils import (
    CALIBRABLE_PARAMS,
    FROZEN_PARAMS,
    ALL_PARAMS,
    get_default_params,
    get_default_frozen_params,
    get_default_calibrable_params,
    split_params,
    merge_params,
    constrained_to_unconstrained,
    unconstrained_to_constrained,
)
