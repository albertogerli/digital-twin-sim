# Predictive Intervals: Logit-Space Transform

## Problem
Adding model discrepancy noise (σ_δ=0.558 logit) in percentage space
can produce CI bounds outside [0, 100]. E.g., [95.1, 105.0] for extreme scenarios.

## Fix
Add discrepancy noise in **logit space**, then transform back via sigmoid:
```
logit(q/100) ~ Normal(logit(μ/100), σ_δ)
q = sigmoid(logit(q/100)) × 100  →  CI ∈ [0, 100] by construction
```

## Results

| Metric | Old (pct space) | New (logit space) |
|---|---|---|
| Coverage (90% CI) | 74.4% | 72.1% |
| Avg CI width | 44.2pp | 37.3pp |
| CIs outside [0,100] | 5 | 0 |
| Bounds guaranteed | No | Yes |

**43 scenarios evaluated.**
