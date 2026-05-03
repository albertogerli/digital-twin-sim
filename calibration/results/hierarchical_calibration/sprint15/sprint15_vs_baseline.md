# Sprint 15 — Re-calibration vs v2_discrepancy baseline

_Generated 2026-05-03T08:50:30+00:00_

## Setup
- **Baseline**: `calibration/results/hierarchical_calibration/v2_discrepancy/`
  (last run before Sprint 1-13 simulator fixes)
- **Sprint 15**: `calibration/results/hierarchical_calibration/sprint15/`
  (re-run after country-alias fix, realism gate fix, agent prompt + engine improvements)
- Same 42 empirical scenarios, same SVI hyperparameters (3000 steps, lr=0.005, seed=42)

## Final SVI loss
- baseline: 514.74
- sprint15: 493.79 (-20.95 ✓)

## Aggregate metrics
Format: `MAE` ± delta (✓ improved, ✗ regressed) | `cov` lift to 90%/50% | `CRPS`

- **OVERALL**  N= 42  MAE=14.65pp (-0.57 ✓)  cov90=83.3% (+4.8 ✓)  cov50=42.9% (+0.0)  CRPS=11.16 (-0.34 ✓)
- **TRAIN**  N= 34  MAE=13.97pp (-0.32 ✓)  cov90=82.4% (+2.9 ✓)  cov50=44.1% (+0.0)  CRPS=10.47 (-0.10 ✓)
- **TEST**  N=  8  MAE=17.56pp (-1.62 ✓)  cov90=87.5% (+12.5 ✓)  cov50=37.5% (+0.0)  CRPS=14.09 (-1.35 ✓)

## Per-domain MAE (full corpus)

| Domain         | N |  baseline MAE | sprint15 MAE | Δ            | cov90 base→new |
|----------------|---|---------------|--------------|--------------|----------------|
| commercial     | 2 |        7.10pp |      18.30pp | +11.20pp ✗ | 100.0% → 100.0% |
| corporate      | 7 |       13.31pp |      13.30pp | -0.01pp ✓ |  85.7% →  85.7% |
| energy         | 1 |        6.22pp |       4.15pp | -2.07pp ✓ | 100.0% → 100.0% |
| environmental  | 1 |        7.55pp |       0.01pp | -7.54pp ✓ | 100.0% → 100.0% |
| financial      | 7 |       32.59pp |      34.39pp | +1.79pp ✗ |  28.6% →  28.6% |
| labor          | 1 |       16.58pp |       0.51pp | -16.07pp ✓ | 100.0% → 100.0% |
| political      | 14 |       10.95pp |      10.22pp | -0.73pp ✓ |  85.7% →  92.9% |
| public_health  | 5 |       16.37pp |      10.08pp | -6.29pp ✓ |  80.0% → 100.0% |
| social         | 1 |        5.81pp |      10.61pp | +4.80pp ✗ | 100.0% → 100.0% |
| technology     | 3 |       10.83pp |      12.08pp | +1.25pp ✗ | 100.0% → 100.0% |

## Interpretation

**TEST MAE improved by 1.62pp** — Sprint 1-13 simulator changes meaningfully tightened predictions.

### Per-scenario TEST diff

| Scenario | gt | base pred | new pred | base |err| | new |err| | Δ |
|----------|----|-----------|----------|------------|-----------|---|
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACT |  62.0 |   54.09 |   44.96 |   7.91 |  17.04 | +9.13 ✗ |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018 |  56.0 |   63.27 |   67.27 |   7.27 |  11.27 | +3.99 ✗ |
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MAR |  35.0 |  100.00 |   99.92 |  65.00 |  64.92 | -0.07 ✓ |
| PH-2021-COVID_VAX_IT                   |  80.0 |   70.76 |   85.17 |   9.24 |   5.17 | -4.07 ✓ |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF |  38.7 |   65.33 |   54.54 |  26.63 |  15.84 | -10.78 ✓ |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCI |  66.1 |   51.60 |   58.93 |  14.50 |   7.17 | -7.32 ✓ |
| POL-2017-TURKISH_CONSTITUTIONAL_REFERE |  51.4 |   57.51 |   57.89 |   6.11 |   6.49 | +0.38 ✗ |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_201 |  83.0 |   66.26 |   70.47 |  16.74 |  12.53 | -4.21 ✓ |
