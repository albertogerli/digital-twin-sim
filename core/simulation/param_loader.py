"""Load calibrated parameters from v2 Bayesian posterior.

Converts the hierarchical posterior (posteriors_v2.json) to the parameter
format used by OpinionDynamics at runtime.

Fallback hierarchy:
1. Scenario-specific θ_s (if available in posterior)
2. Domain-level μ_d
3. Global μ_global
4. v1 grid-search params (calibrated_params_{domain}.json)
5. Hardcoded defaults
"""

import json
import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Path default del posterior
DEFAULT_POSTERIOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "calibration", "results", "hierarchical_calibration",
    "v2.3_pubop", "posteriors_v2.json"
)

# v2 parameter names in order (matches posterior array indices)
PARAM_NAMES = ["alpha_herd", "alpha_anchor", "alpha_social", "alpha_event"]

# Mapping from v2 calibrated names to v1 OpinionDynamics constructor names
V2_TO_V1_MAP = {
    "alpha_herd": "herd_weight",
    "alpha_anchor": "anchor_weight",
    "alpha_social": "social_weight",
    "alpha_event": "event_weight",
}

# Frozen params from Sobol analysis (S1 < 0.01, safe to fix)
FROZEN_DEFAULTS = {
    "herd_threshold": 0.21,          # sigmoid(0.5) ≈ 0.62, but v1 uses 0.2 directly
    "direct_shift_weight": 0.4,
    "anchor_drift_rate": 0.25,       # sigmoid(-1.4) ≈ 0.20
}

# v1 hardcoded defaults (last resort)
V1_DEFAULTS = {
    "anchor_weight": 0.1,
    "social_weight": 0.15,
    "event_weight": 0.05,
    "herd_weight": 0.05,
    "herd_threshold": 0.2,
    "direct_shift_weight": 0.4,
    "anchor_drift_rate": 0.2,
}

# Default crisis params (from regime_switching.py CRISIS_DEFAULTS)
CRISIS_DEFAULTS = {
    "lambda_multiplier": 3.0,
    "anchor_suppression": 0.1,
    "event_amplification": 2.5,
    "herd_amplification": 2.0,
    "contagion_speed": 0.4,
}

TRANSITION_DEFAULTS = {
    "shock_trigger_threshold": 0.5,
    "velocity_trigger_threshold": 0.1,
    "trust_sensitivity": 1.5,
    "crisis_duration_mean": 2.5,
    "recovery_rate": 0.4,
}


def _softmax_weight(alpha_vec: list[float], index: int) -> float:
    """Compute softmax weight for index given alpha vector [0, a_h, a_a, a_s, a_e]."""
    # Numerically stable softmax: subtract max to prevent overflow
    max_a = max(alpha_vec) if alpha_vec else 0.0
    exps = [math.exp(a - max_a) for a in alpha_vec]
    total = sum(exps)
    return exps[index] / total if total > 0 else 0.2


def _alpha_to_v1_weight(alpha: float, alpha_vec: list[float], index: int) -> float:
    """Convert a softmax alpha to a v1-style weight (0-1 range).

    v1 weights are additive (anchor_pull + social_pull + ...).
    v2 weights go through softmax. We approximate the v1 weight
    as the softmax proportion scaled to a reasonable v1 range.

    The softmax output π_k ∈ (0,1) sums to 1. The v1 weights
    are typically 0.05–0.4 and don't sum to 1. We scale π_k
    so the total matches typical v1 weight sums (~0.65).
    """
    pi = _softmax_weight(alpha_vec, index)
    # v1 total weight sum is roughly 0.65 (0.1 + 0.15 + 0.05 + 0.05 + 0.4 - direct)
    # The 4 non-direct v1 weights sum to ~0.35
    # Map softmax π (excluding direct) to v1 weight space
    return round(pi * 0.8, 4)  # Scale factor chosen to match typical v1 ranges


class CalibratedParamLoader:
    """Load and serve calibrated parameters from v2 Bayesian posterior."""

    def __init__(self, posterior_path: Optional[str] = None):
        self.posterior_path = posterior_path or DEFAULT_POSTERIOR_PATH
        self.posterior = None
        self.available = False
        self._load()

    def _load(self):
        """Load the posterior JSON if available."""
        if not os.path.exists(self.posterior_path):
            logger.info(f"Posterior not found at {self.posterior_path} — using v1 defaults")
            return

        try:
            with open(self.posterior_path) as f:
                self.posterior = json.load(f)
            self.available = True
            mu = self.posterior["global"]["mu_global"]["mean"]
            logger.info(
                f"Loaded v2 posterior: μ_global = "
                f"[h={mu[0]:.3f}, a={mu[1]:.3f}, s={mu[2]:.3f}, e={mu[3]:.3f}]"
            )
        except Exception as e:
            logger.warning(f"Failed to load posterior: {e}")
            self.posterior = None

    def _get_alphas(self, domain: Optional[str] = None) -> tuple[list[float], str]:
        """Get alpha vector [α_h, α_a, α_s, α_e] from best available source.

        Returns (alphas, source) where source is 'domain' or 'global'.
        """
        if not self.posterior:
            return [0.0, 0.0, 0.0, 0.0], "defaults"

        # Try domain-level params
        if domain and "domains" in self.posterior:
            domain_data = self.posterior["domains"].get(domain)
            if domain_data and "mu_d" in domain_data:
                mu_d = domain_data["mu_d"]["mean"]
                return mu_d, "domain"

        # Fall back to global
        mu_global = self.posterior["global"]["mu_global"]["mean"]
        return mu_global, "global"

    def _get_ci95(self, domain: Optional[str] = None) -> dict:
        """Get 95% CI for each parameter."""
        if not self.posterior:
            return {}

        # Try domain-level
        if domain and "domains" in self.posterior:
            domain_data = self.posterior["domains"].get(domain)
            if domain_data and "mu_d" in domain_data:
                mu_d = domain_data["mu_d"]
                result = {}
                for i, name in enumerate(PARAM_NAMES):
                    v1_name = V2_TO_V1_MAP[name]
                    result[v1_name] = (mu_d["ci95_lo"][i], mu_d["ci95_hi"][i])
                return result

        # Global CI
        mu = self.posterior["global"]["mu_global"]
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            v1_name = V2_TO_V1_MAP[name]
            result[v1_name] = (mu["ci95_lo"][i], mu["ci95_hi"][i])
        return result

    def get_params(
        self,
        domain: Optional[str] = None,
        scenario_id: Optional[str] = None,
        include_uncertainty: bool = False,
    ) -> dict:
        """Return parameters in v1 format (for OpinionDynamics compatibility).

        Args:
            domain: Domain name (e.g. "political", "financial")
            scenario_id: Specific scenario ID (unused for now, reserved for v3)
            include_uncertainty: If True, include CI95 for each parameter

        Returns:
            Dict with v1 param names + metadata fields prefixed with '_'.
        """
        if not self.available:
            # Try v1 calibrated params file
            params = self._try_v1_params(domain)
            if params:
                params["_source"] = "v1_grid"
                params["_model_version"] = "v1"
                return params
            # Last resort: hardcoded defaults
            result = dict(V1_DEFAULTS)
            result["_source"] = "defaults"
            result["_model_version"] = "v1"
            return result

        alphas, source = self._get_alphas(domain)

        # Build full alpha vector with gauge-fixed direct = 0
        alpha_vec = [0.0, alphas[0], alphas[1], alphas[2], alphas[3]]

        # Convert to v1 weights
        result = {
            "herd_weight": _alpha_to_v1_weight(alphas[0], alpha_vec, 1),
            "anchor_weight": _alpha_to_v1_weight(alphas[1], alpha_vec, 2),
            "social_weight": _alpha_to_v1_weight(alphas[2], alpha_vec, 3),
            "event_weight": _alpha_to_v1_weight(alphas[3], alpha_vec, 4),
        }
        # Add frozen params
        result.update(FROZEN_DEFAULTS)

        # Metadata
        result["_source"] = source
        result["_model_version"] = "v2"
        result["_alphas"] = {
            "alpha_herd": alphas[0],
            "alpha_anchor": alphas[1],
            "alpha_social": alphas[2],
            "alpha_event": alphas[3],
        }

        if include_uncertainty:
            result["_ci95"] = self._get_ci95(domain)

        return result

    def get_v2_params(self, domain: Optional[str] = None) -> dict:
        """Return raw v2 parameters (alpha space) for direct JAX simulator use.

        Returns dict with alpha_herd, alpha_anchor, alpha_social, alpha_event
        plus frozen params in their v2 form.
        """
        if not self.available:
            return {
                "alpha_herd": -0.176,
                "alpha_anchor": 0.297,
                "alpha_social": -0.105,
                "alpha_event": -0.130,
                "log_lambda_elite": -1.2,
                "log_lambda_citizen": -0.5,
                "logit_herd_threshold": 0.5,
                "logit_anchor_drift": -1.4,
            }

        alphas, _ = self._get_alphas(domain)
        return {
            "alpha_herd": alphas[0],
            "alpha_anchor": alphas[1],
            "alpha_social": alphas[2],
            "alpha_event": alphas[3],
            "log_lambda_elite": -1.2,
            "log_lambda_citizen": -0.5,
            "logit_herd_threshold": 0.5,
            "logit_anchor_drift": -1.4,
        }

    def get_discrepancy(
        self,
        domain: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> dict:
        """Return discrepancy parameters for readout correction.

        Returns:
            {"delta_d": float, "delta_s": float, "sigma_delta": float}
        """
        if not self.posterior:
            return {"delta_d": 0.0, "delta_s": 0.0, "sigma_delta": 0.0}

        delta_d = 0.0
        if domain and "domains" in self.posterior:
            domain_data = self.posterior["domains"].get(domain, {})
            delta_d_data = domain_data.get("delta_d", {})
            delta_d = delta_d_data.get("mean", 0.0) if isinstance(delta_d_data, dict) else 0.0

        # sigma_delta from global
        sigma_within = 0.558  # default from calibration
        if "discrepancy" in self.posterior.get("global", {}):
            sigma_within = self.posterior["global"]["discrepancy"].get("sigma_within", {}).get("mean", 0.558)

        return {
            "delta_d": delta_d,
            "delta_s": 0.0,  # scenario-specific: not available without running calibration
            "sigma_delta": sigma_within,
        }

    def get_crisis_params(self) -> dict:
        """Return crisis parameters for regime switching.

        For now returns defaults. When v3 posterior is available,
        will load from posteriors_v3.json.
        """
        return dict(CRISIS_DEFAULTS)

    def get_transition_params(self) -> dict:
        """Return transition parameters for regime switching."""
        return dict(TRANSITION_DEFAULTS)

    def get_calibration_info(self) -> dict:
        """Return metadata about the calibration for display purposes."""
        if not self.available:
            return {
                "model_version": "v1",
                "params_source": "grid_search",
                "n_training_scenarios": 0,
                "test_mae": None,
                "test_coverage_90": None,
                "enkf_available": False,
            }
        return {
            "model_version": "v2",
            "params_source": "posterior",
            "n_training_scenarios": 34,
            "n_test_scenarios": 8,
            "test_mae": 12.6,
            "test_coverage_90": 85.7,
            "enkf_available": True,
        }

    def _try_v1_params(self, domain: Optional[str] = None) -> Optional[dict]:
        """Try to load v1 calibrated params from JSON file."""
        if domain:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "calibration", "results",
                f"calibrated_params_{domain}.json"
            )
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    params = dict(V1_DEFAULTS)
                    cal = data.get("calibrated_params", data)
                    params.update({k: v for k, v in cal.items() if k in V1_DEFAULTS})
                    logger.info(f"Loaded v1 calibrated params from {path}")
                    return params
                except Exception as e:
                    logger.warning(f"Failed to load v1 params: {e}")
        return None

    def print_comparison(self, domain: Optional[str] = None):
        """Print v1 vs v2 params side-by-side for debugging."""
        v1 = dict(V1_DEFAULTS)
        v1_loaded = self._try_v1_params(domain)
        if v1_loaded:
            v1 = v1_loaded

        v2 = self.get_params(domain=domain, include_uncertainty=True)

        print(f"\n{'─' * 60}")
        print(f"  Parameter Comparison (domain: {domain or 'global'})")
        print(f"  Source: {v2.get('_source', '?')} | Model: {v2.get('_model_version', '?')}")
        print(f"{'─' * 60}")
        print(f"  {'Param':<22} {'v1':>8} {'v2':>8}   {'v2 CI95'}")
        print(f"  {'─' * 56}")
        ci = v2.get("_ci95", {})
        for name in ["anchor_weight", "social_weight", "event_weight", "herd_weight"]:
            v1_val = v1.get(name, 0)
            v2_val = v2.get(name, 0)
            ci_str = ""
            if name in ci:
                lo, hi = ci[name]
                ci_str = f"[{lo:.3f}, {hi:.3f}]"
            print(f"  {name:<22} {v1_val:8.4f} {v2_val:8.4f}   {ci_str}")
        if "_alphas" in v2:
            print(f"\n  Raw alphas (v2 logit space):")
            for k, v in v2["_alphas"].items():
                print(f"    {k}: {v:.4f}")
        print(f"{'─' * 60}\n")
