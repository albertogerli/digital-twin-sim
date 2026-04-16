"""Observation source adapters for EnKF data assimilation.

Each source converts raw data into (value, variance, obs_type) for the EnKF update.
Variance reflects measurement reliability: official results ≈ 0, polls ≈ p(1-p)/N,
social sentiment ≈ much higher (noisier signal).
"""

from dataclasses import dataclass


@dataclass
class ObservationSource:
    """Base class for observation sources."""

    def to_observation(self) -> tuple[float, float, str]:
        """Convert to (value_pct, variance, obs_type).

        Returns:
            value_pct: Observed pro% (0-100 scale).
            variance: Observation variance in percentage-point² units.
            obs_type: One of "polling", "sentiment", "outcome".
        """
        raise NotImplementedError


@dataclass
class PollingSurvey(ObservationSource):
    """Survey/poll with known sample size.

    Variance = p(1-p)/N × 100² to convert from proportion to pct² scale.
    Optional pollster_bias adds systematic bias correction.
    """

    pro_pct: float
    sample_size: int
    pollster_bias: float = 0.0  # additive bias in pp (e.g., +2 means poll overestimates by 2pp)

    def to_observation(self) -> tuple[float, float, str]:
        p = self.pro_pct / 100.0
        # Binomial variance in pct² units
        variance = p * (1.0 - p) / max(self.sample_size, 1) * 10000.0
        # Floor: even large polls have ~1pp² irreducible noise (design effects, etc.)
        variance = max(variance, 1.0)
        return self.pro_pct - self.pollster_bias, variance, "polling"


@dataclass
class SentimentSignal(ObservationSource):
    """Social media sentiment signal.

    Much noisier than polling. Confidence in [0, 1] maps to variance:
    high confidence (1.0) → variance ~25 pp²
    low confidence (0.1) → variance ~250 pp²
    """

    pro_pct: float
    confidence: float = 0.5  # 0-1, how reliable this signal is

    def to_observation(self) -> tuple[float, float, str]:
        # Base variance 50pp², scaled by 1/confidence
        variance = 50.0 / max(self.confidence, 0.01)
        return self.pro_pct, variance, "sentiment"


@dataclass
class OfficialResult(ObservationSource):
    """Official result (election, referendum).

    Near-zero variance — this is ground truth.
    """

    pro_pct: float

    def to_observation(self) -> tuple[float, float, str]:
        # Tiny variance (not exactly 0 for numerical stability)
        return self.pro_pct, 0.01, "outcome"
