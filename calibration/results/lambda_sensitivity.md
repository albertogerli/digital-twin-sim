# Lambda_citizen Sensitivity Analysis

**Sobol S_T for λ_citizen:** 0.121 (not negligible)
**Default λ_citizen:** 0.2500
**Perturbation range:** 0.150 to 0.350 (0.6x to 1.4x)

| Scenario | MAE(λ=0.15) | MAE(λ=0.20) | MAE(λ=0.25) | MAE(λ=0.30) | MAE(λ=0.35) | max Δ |
|---|---|---|---|---|---|---|
| COM-2017-IPHONE_X | 5.8 | 5.9 | 5.9 | 7.0 | 11.5 | 5.6 |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | 8.5 | 10.5 | 11.3 | 11.6 | 11.8 | 3.2 |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | 29.8 | 29.6 | 29.6 | 29.6 | 29.7 | 0.2 |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | 15.0 | 15.0 | 15.0 | 15.0 | 15.0 | 0.0 |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | 17.7 | 14.2 | 12.0 | 10.8 | 9.9 | 7.9 |

**Max |Δ(MAE)|:** 7.9pp
**Verdict:** SENSITIVE to λ perturbation
