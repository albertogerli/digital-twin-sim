"""Parameter utilities for DynamicsV2-JAX.

Handles the split between calibrable and frozen parameters,
default values, and constrained ↔ unconstrained transforms.
"""

import jax.numpy as jnp
from jax.nn import sigmoid

# ── Parameter lists ────────────────────────────────────────────────
# Moving a parameter between these lists is the ONLY change needed
# to freeze/unfreeze it for calibration.

CALIBRABLE_PARAMS = [
    "alpha_herd",
    "alpha_anchor",
    "alpha_social",
    "alpha_event",
]

FROZEN_PARAMS = [
    "log_lambda_elite",
    "log_lambda_citizen",
    "logit_herd_threshold",
    "logit_anchor_drift",
]

ALL_PARAMS = CALIBRABLE_PARAMS + FROZEN_PARAMS

# Force order: direct(gauge-fixed=0), herd, anchor, social, event
FORCE_NAMES = ["direct", "herd", "anchor", "social", "event"]
ALPHA_KEYS = ["alpha_herd", "alpha_anchor", "alpha_social", "alpha_event"]


# ── Defaults ───────────────────────────────────────────────────────

def get_default_frozen_params() -> dict:
    """Default values for frozen parameters (unconstrained space)."""
    return {
        "log_lambda_elite": jnp.log(jnp.array(0.15)),
        "log_lambda_citizen": jnp.log(jnp.array(0.25)),
        "logit_herd_threshold": _logit(jnp.array(0.21)),
        "logit_anchor_drift": _logit(jnp.array(0.25)),
    }


def get_default_calibrable_params() -> dict:
    """Default values for calibrable parameters."""
    return {
        "alpha_herd": jnp.array(0.0),
        "alpha_anchor": jnp.array(0.0),
        "alpha_social": jnp.array(0.0),
        "alpha_event": jnp.array(0.0),
    }


def get_default_params() -> dict:
    """Full default parameter dict."""
    return {**get_default_calibrable_params(), **get_default_frozen_params()}


# ── Split / Merge ──────────────────────────────────────────────────

def split_params(
    full_params: dict,
    calibrable_list: list[str] = CALIBRABLE_PARAMS,
) -> tuple[dict, dict]:
    """Split full params into (calibrable, frozen) dicts."""
    calibrable = {k: full_params[k] for k in calibrable_list if k in full_params}
    frozen = {k: v for k, v in full_params.items() if k not in calibrable_list}
    return calibrable, frozen


def merge_params(calibrable: dict, frozen: dict) -> dict:
    """Merge calibrable and frozen dicts into full params."""
    return {**frozen, **calibrable}


# ── Constrained ↔ Unconstrained ───────────────────────────────────

def constrained_to_unconstrained(params: dict) -> dict:
    """Transform constrained params to unconstrained space.

    Alpha logits are already unconstrained (-inf, +inf).
    Lambda and threshold/drift are positive/bounded:
      lambda -> log(lambda)
      threshold/drift in (0,1) -> logit(x)
    """
    out = {}
    for k, v in params.items():
        if k.startswith("alpha_"):
            out[k] = v  # already unconstrained
        elif k.startswith("log_lambda_"):
            out[k] = v  # already in log space
        elif k.startswith("logit_"):
            out[k] = v  # already in logit space
        else:
            out[k] = v
    return out


def unconstrained_to_constrained(params: dict) -> dict:
    """Transform unconstrained params back to constrained space.

    Returns a dict with native-scale values:
      alpha_* -> same (unconstrained logits)
      log_lambda_* -> exp(.) -> lambda (positive)
      logit_* -> sigmoid(.) -> value in (0, 1)
    """
    out = {}
    for k, v in params.items():
        if k.startswith("alpha_"):
            out[k] = v
        elif k.startswith("log_lambda_"):
            native_key = k.replace("log_lambda_", "lambda_")
            out[native_key] = jnp.exp(v)
        elif k.startswith("logit_"):
            native_key = k.replace("logit_", "")
            out[native_key] = sigmoid(v)
        else:
            out[k] = v
    return out


# ── Internal helpers ───────────────────────────────────────────────

def _logit(x):
    """Logit transform: log(x / (1-x))."""
    return jnp.log(x / (1.0 - x))
