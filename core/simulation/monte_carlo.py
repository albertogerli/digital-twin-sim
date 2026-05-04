"""Monte Carlo simulation runner — N parallel runs with parameter perturbation.

Runs the full LLM simulation multiple times with slightly different parameters
around the calibrated values, producing confidence intervals on outcomes.

Usage:
    engine = MonteCarloEngine(base_config, domain, n_runs=20)
    results = await engine.run(llm, progress_callback)
    # results.confidence_intervals, results.outcome_distribution, etc.
"""

import asyncio
import copy
import logging
import random
from dataclasses import dataclass, field
from statistics import mean, stdev

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloRoundResult:
    """Aggregated results for a single round across all runs."""
    round: int
    polarization_mean: float
    polarization_std: float
    polarization_ci_low: float
    polarization_ci_high: float
    avg_position_mean: float
    avg_position_std: float
    sentiment_positive_mean: float
    sentiment_neutral_mean: float
    sentiment_negative_mean: float
    coalition_sizes: list[dict]  # [{label, mean_size, std_size}]


@dataclass
class ScenarioCluster:
    """A distinct trajectory-shape cluster discovered across MC runs.

    Each cluster groups runs that share a similar trajectory (round-by-round
    avg_position + final polarization), letting the UI show probabilistic
    "what-if" scenarios instead of a single averaged outcome.
    """
    cluster_id: int
    n_runs: int
    pct: float  # share of all runs landing in this cluster
    label: str  # semantic label derived from the cluster centroid
    mean_final_position: float
    mean_final_polarization: float
    outcome_pro_pct: float  # pro share within this cluster only
    outcome_against_pct: float
    mean_trajectory: list[dict]  # [{round, avg_position, polarization}, ...]
    run_ids: list[int]


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo analysis result."""
    n_runs: int
    n_completed: int
    rounds: list[MonteCarloRoundResult]
    final_polarization_mean: float
    final_polarization_std: float
    final_polarization_ci: tuple[float, float]
    final_position_mean: float
    final_position_std: float
    final_position_ci: tuple[float, float]
    outcome_pro_pct: float  # % of runs where avg_position > 0
    outcome_against_pct: float
    parameter_sets: list[dict]
    per_run_summaries: list[dict]
    scenario_clusters: list[ScenarioCluster] = field(default_factory=list)


def perturb_params(
    base_params: dict,
    perturbation_pct: float = 0.15,
    seed: int = None,
    posteriors: dict | None = None,
) -> dict:
    """Sample a parameter set for one Monte Carlo run.

    If ``posteriors`` is provided as ``{param_name: (ci95_lo, ci95_hi)}``
    (e.g. from ``CalibratedParamLoader.get_params(include_uncertainty=True)``
    `_ci95` field), each posterior-covered parameter is drawn from
    ``Normal(mean=base, sigma=(hi-lo)/3.92)`` (the 1.96·σ → CI95 mapping)
    truncated to the safety range. This is the principled path: the
    spread reflects measured uncertainty from the NumPyro hierarchical
    fit, not an operator-tuned dial.

    Parameters NOT covered by ``posteriors`` (or when ``posteriors`` is
    None) fall back to a uniform ±perturbation_pct around the base value
    — same legacy behaviour as before this commit.

    Clamps every output to ``param_ranges`` so a wide CI tail can't push
    a weight outside what OpinionDynamics tolerates.
    """
    rng = random.Random(seed)
    perturbed = {}

    param_ranges = {
        "anchor_weight": (0.01, 0.30),
        "social_weight": (0.03, 0.35),
        "event_weight": (0.01, 0.20),
        "herd_weight": (0.01, 0.15),
        "herd_threshold": (0.05, 0.50),
        "direct_shift_weight": (0.1, 0.8),
        "anchor_drift_rate": (0.05, 0.50),
    }

    posteriors = posteriors or {}
    Z95 = 1.959964  # 95% normal quantile

    for key, value in base_params.items():
        if key not in param_ranges:
            perturbed[key] = value
            continue
        lo_safe, hi_safe = param_ranges[key]
        ci = posteriors.get(key)
        if ci is not None and ci[1] > ci[0]:
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
            sigma = max(1e-6, (ci_hi - ci_lo) / (2 * Z95))
            sampled = rng.gauss(value, sigma)
        else:
            delta = value * perturbation_pct
            sampled = rng.uniform(value - delta, value + delta)
        perturbed[key] = max(lo_safe, min(hi_safe, sampled))

    return perturbed


def _compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval using normal approximation."""
    if len(values) < 2:
        m = values[0] if values else 0
        return (m, m)

    m = mean(values)
    s = stdev(values)
    # z-score for 95% CI
    z = 1.96 if confidence == 0.95 else 1.645
    margin = z * s / (len(values) ** 0.5)
    return (m - margin, m + margin)


def _label_cluster(mean_final_position: float) -> str:
    """Human-readable label for a cluster from its mean final avg_position."""
    if mean_final_position >= 0.30:
        return "Vittoria netta (Pro)"
    if mean_final_position >= 0.10:
        return "Vittoria risicata (Pro)"
    if mean_final_position > -0.10:
        return "Testa a testa"
    if mean_final_position > -0.30:
        return "Vittoria risicata (Contro)"
    return "Vittoria netta (Contro)"


def _compute_scenario_clusters(
    all_runs: list[dict],
    max_rounds: int,
    random_state: int = 42,
) -> list[ScenarioCluster]:
    """Cluster MC runs by trajectory shape and emit per-cluster probabilities.

    Feature vector per run = [pos_r1, ..., pos_rN, final_polarization/10].
    K auto-selected in [2, 5] via silhouette; falls back to no clustering when
    sklearn is unavailable or we have too few runs for the technique to be
    meaningful (minimum 4 runs to support k=2 with ≥2 members per cluster).
    """
    if len(all_runs) < 4 or max_rounds < 1:
        return []

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as e:  # pragma: no cover - sklearn is a declared dep
        logger.warning("sklearn unavailable, skipping scenario clustering: %s", e)
        return []

    # Build feature matrix (pad shorter runs with their last observed value)
    features = []
    for run in all_runs:
        rounds = run.get("rounds", [])
        traj = [r.get("avg_position", 0.0) for r in rounds[:max_rounds]]
        if not traj:
            traj = [0.0]
        while len(traj) < max_rounds:
            traj.append(traj[-1])
        # Polarization is 0-10; normalize so it lives on a comparable scale.
        traj.append(run.get("final_polarization", 0.0) / 10.0)
        features.append(traj)

    X = np.array(features, dtype=float)
    # z-score normalize per feature; guard against zero-variance columns
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    X_norm = (X - mu) / sigma

    max_k = min(5, len(all_runs) - 1)
    best_k = None
    best_score = -1.0
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_norm)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X_norm, labels)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_k is None or best_labels is None:
        # Fallback to k=2 deterministically
        km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        best_labels = km.fit_predict(X_norm)
        best_k = 2

    # Assemble cluster summaries
    clusters: list[ScenarioCluster] = []
    for cid in range(best_k):
        member_idx = [i for i, lbl in enumerate(best_labels) if lbl == cid]
        if not member_idx:
            continue
        members = [all_runs[i] for i in member_idx]

        mean_traj = []
        for rd in range(max_rounds):
            pols = []
            poss = []
            for m in members:
                rounds = m.get("rounds", [])
                if rd < len(rounds):
                    pols.append(rounds[rd].get("polarization", 0.0))
                    poss.append(rounds[rd].get("avg_position", 0.0))
            mean_traj.append({
                "round": rd + 1,
                "avg_position": round(mean(poss), 3) if poss else 0.0,
                "polarization": round(mean(pols), 2) if pols else 0.0,
            })

        final_positions = [m.get("final_avg_position", 0.0) for m in members]
        final_pols = [m.get("final_polarization", 0.0) for m in members]
        mean_final_pos = mean(final_positions) if final_positions else 0.0
        mean_final_pol = mean(final_pols) if final_pols else 0.0

        pro = sum(1 for p in final_positions if p > 0.05)
        against = sum(1 for p in final_positions if p < -0.05)
        decided = pro + against or 1

        clusters.append(ScenarioCluster(
            cluster_id=cid,
            n_runs=len(members),
            pct=round(len(members) / len(all_runs) * 100, 1),
            label=_label_cluster(mean_final_pos),
            mean_final_position=round(mean_final_pos, 3),
            mean_final_polarization=round(mean_final_pol, 2),
            outcome_pro_pct=round(pro / decided * 100, 1),
            outcome_against_pct=round(against / decided * 100, 1),
            mean_trajectory=mean_traj,
            run_ids=member_idx,
        ))

    # Sort by probability share (desc) and reindex
    clusters.sort(key=lambda c: c.pct, reverse=True)
    for new_id, c in enumerate(clusters):
        c.cluster_id = new_id

    return clusters


class MonteCarloEngine:
    """Runs N simulations with perturbed parameters and aggregates results."""

    def __init__(
        self,
        n_runs: int = 20,
        perturbation_pct: float = 0.15,
        base_seed: int = 42,
        posteriors: dict | None = None,
    ):
        self.n_runs = n_runs
        self.perturbation_pct = perturbation_pct
        self.base_seed = base_seed
        # When provided, perturb_params will sample each covered weight
        # from its NumPyro posterior CI95 instead of uniform ±%.
        self.posteriors = posteriors or {}

    def generate_parameter_sets(self, base_params: dict) -> list[dict]:
        """Generate N parameter sets: first is base, rest are perturbed."""
        param_sets = [base_params.copy()]  # Run 0 = unperturbed baseline
        for i in range(1, self.n_runs):
            param_sets.append(perturb_params(
                base_params,
                perturbation_pct=self.perturbation_pct,
                seed=self.base_seed + i,
                posteriors=self.posteriors,
            ))
        return param_sets

    def aggregate_results(
        self,
        all_runs: list[dict],
        parameter_sets: list[dict],
    ) -> MonteCarloResult:
        """Aggregate results from N simulation runs.

        Each run is a dict with:
            - rounds: [{round, polarization, avg_position, sentiment, coalitions}]
            - final_polarization: float
            - final_avg_position: float
        """
        if not all_runs:
            return MonteCarloResult(
                n_runs=self.n_runs, n_completed=0, rounds=[],
                final_polarization_mean=0, final_polarization_std=0,
                final_polarization_ci=(0, 0),
                final_position_mean=0, final_position_std=0,
                final_position_ci=(0, 0),
                outcome_pro_pct=50, outcome_against_pct=50,
                parameter_sets=parameter_sets, per_run_summaries=[],
            )

        # Aggregate per-round
        max_rounds = max(len(r.get("rounds", [])) for r in all_runs)
        round_results = []

        for rd in range(max_rounds):
            pols = []
            positions = []
            sent_pos = []
            sent_neu = []
            sent_neg = []

            for run in all_runs:
                rounds = run.get("rounds", [])
                if rd < len(rounds):
                    r = rounds[rd]
                    pols.append(r.get("polarization", 0))
                    positions.append(r.get("avg_position", 0))
                    s = r.get("sentiment", {})
                    sent_pos.append(s.get("positive", 0))
                    sent_neu.append(s.get("neutral", 0))
                    sent_neg.append(s.get("negative", 0))

            pol_ci = _compute_ci(pols)
            round_results.append(MonteCarloRoundResult(
                round=rd + 1,
                polarization_mean=mean(pols) if pols else 0,
                polarization_std=stdev(pols) if len(pols) > 1 else 0,
                polarization_ci_low=pol_ci[0],
                polarization_ci_high=pol_ci[1],
                avg_position_mean=mean(positions) if positions else 0,
                avg_position_std=stdev(positions) if len(positions) > 1 else 0,
                sentiment_positive_mean=mean(sent_pos) if sent_pos else 0,
                sentiment_neutral_mean=mean(sent_neu) if sent_neu else 0,
                sentiment_negative_mean=mean(sent_neg) if sent_neg else 0,
                coalition_sizes=[],
            ))

        # Final metrics
        final_pols = [r.get("final_polarization", 0) for r in all_runs]
        final_positions = [r.get("final_avg_position", 0) for r in all_runs]

        pro_count = sum(1 for p in final_positions if p > 0.05)
        against_count = sum(1 for p in final_positions if p < -0.05)
        decided = pro_count + against_count or 1

        scenario_clusters = _compute_scenario_clusters(all_runs, max_rounds)

        return MonteCarloResult(
            n_runs=self.n_runs,
            n_completed=len(all_runs),
            rounds=round_results,
            final_polarization_mean=mean(final_pols) if final_pols else 0,
            final_polarization_std=stdev(final_pols) if len(final_pols) > 1 else 0,
            final_polarization_ci=_compute_ci(final_pols),
            final_position_mean=mean(final_positions) if final_positions else 0,
            final_position_std=stdev(final_positions) if len(final_positions) > 1 else 0,
            final_position_ci=_compute_ci(final_positions),
            outcome_pro_pct=round(pro_count / decided * 100, 1),
            outcome_against_pct=round(against_count / decided * 100, 1),
            parameter_sets=parameter_sets,
            per_run_summaries=[
                {
                    "run": i,
                    "final_polarization": r.get("final_polarization", 0),
                    "final_avg_position": r.get("final_avg_position", 0),
                    "params": parameter_sets[i] if i < len(parameter_sets) else {},
                }
                for i, r in enumerate(all_runs)
            ],
            scenario_clusters=scenario_clusters,
        )

    def result_to_dict(self, result: MonteCarloResult) -> dict:
        """Convert MonteCarloResult to JSON-serializable dict."""
        return {
            "n_runs": result.n_runs,
            "n_completed": result.n_completed,
            "final_polarization": {
                "mean": round(result.final_polarization_mean, 2),
                "std": round(result.final_polarization_std, 2),
                "ci_low": round(result.final_polarization_ci[0], 2),
                "ci_high": round(result.final_polarization_ci[1], 2),
            },
            "final_position": {
                "mean": round(result.final_position_mean, 3),
                "std": round(result.final_position_std, 3),
                "ci_low": round(result.final_position_ci[0], 3),
                "ci_high": round(result.final_position_ci[1], 3),
            },
            "outcome_probability": {
                "pro_pct": result.outcome_pro_pct,
                "against_pct": result.outcome_against_pct,
            },
            "rounds": [
                {
                    "round": r.round,
                    "polarization": {
                        "mean": round(r.polarization_mean, 2),
                        "std": round(r.polarization_std, 2),
                        "ci_low": round(r.polarization_ci_low, 2),
                        "ci_high": round(r.polarization_ci_high, 2),
                    },
                    "avg_position": {
                        "mean": round(r.avg_position_mean, 3),
                        "std": round(r.avg_position_std, 3),
                    },
                    "sentiment": {
                        "positive": round(r.sentiment_positive_mean, 2),
                        "neutral": round(r.sentiment_neutral_mean, 2),
                        "negative": round(r.sentiment_negative_mean, 2),
                    },
                }
                for r in result.rounds
            ],
            "per_run": result.per_run_summaries,
            "scenario_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "n_runs": c.n_runs,
                    "pct": c.pct,
                    "label": c.label,
                    "mean_final_position": c.mean_final_position,
                    "mean_final_polarization": c.mean_final_polarization,
                    "outcome_pro_pct": c.outcome_pro_pct,
                    "outcome_against_pct": c.outcome_against_pct,
                    "mean_trajectory": c.mean_trajectory,
                    "run_ids": c.run_ids,
                }
                for c in result.scenario_clusters
            ],
        }
