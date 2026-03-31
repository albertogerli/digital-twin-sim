# SVI vs NUTS Posterior Comparison

**Scenarios:** 2 (political, technology, financial, corporate)
**SVI:** 1000 steps, Adam(lr=0.01), AutoLowRankMVN guide (891s)
**NUTS:** 200 warmup + 200 samples, max_tree_depth=8 (6s)

| Parameter | SVI mean +/- std | NUTS mean +/- std | |Δmean|/σ_NUTS | Compatible |
|---|---|---|---|---|
| alpha_herd | -0.5002 +/- 0.0849 | -0.3826 +/- 0.7961 | 0.148 | Yes |
| alpha_anchor | +0.9050 +/- 0.0651 | +0.6040 +/- 0.8560 | 0.352 | Yes |
| alpha_social | -0.9967 +/- 0.1054 | -0.3495 +/- 0.7632 | 0.848 | No |
| alpha_event | -1.7522 +/- 0.0730 | -0.1369 +/- 0.9464 | 1.707 | No |

**Verdict:** DISCREPANCY DETECTED

All |Δmean|/σ_NUTS values have at least one >= 0.5, indicating the SVI variational approximation is adequate for this model.
