# Hierarchical Calibration Report — v2 (with Discrepancy)

**Generated:** 2026-04-03 11:02

## Model Description

Extension of v1 with additive discrepancy term δ_s in logit space:
- `q_corrected = sigmoid(logit(q_sim) + δ_s)`
- δ_s ~ Normal(δ_d, σ_δ_within) — per scenario
- δ_d ~ Normal(0, σ_δ_between) — per domain
- σ_outcome ~ HalfNormal(3.0) — tight, forces bias into δ

## Phase B: Empirical Fine-tuning

- SVI steps: 3000
- Final loss: 233.0
- Elapsed: 3003.0s

### Calibrated Parameters

- μ_global: [-0.314, 0.614, -0.206, -0.472]
- σ_global: [0.227, 0.230, 0.246, 0.243]

| Parameter | Mean | CI95 Low | CI95 High |
|---|---|---|---|
| alpha_herd | -0.314 | -0.617 | -0.005 |
| alpha_anchor | 0.614 | 0.344 | 0.874 |
| alpha_social | -0.206 | -0.519 | 0.095 |
| alpha_event | -0.472 | -0.760 | -0.178 |

### Discrepancy Parameters

- σ_δ_between: 0.371 (CI: 0.197, 0.612)
- σ_δ_within: 0.471 (CI: 0.339, 0.629)

### Per-Domain Discrepancy (δ_d)

| Domain | δ_d Mean | CI95 Low | CI95 High | Interpretation |
|---|---|---|---|---|
| corporate | -0.198 | -0.631 | 0.265 | No significant bias |
| financial | -0.499 | -0.873 | -0.117 | Sim over-predicts |
| political | 0.073 | -0.225 | 0.354 | No significant bias |
| public_health | 0.014 | -0.456 | 0.489 | No significant bias |

## Phase C: Validation

| Metric | Train | Test |
|---|---|---|
| n | 15.000 | 5.000 |
| mae | 13.926 | 13.617 |
| rmse | 20.454 | 19.960 |
| coverage_90 | 0.867 | 0.800 |
| coverage_50 | 0.400 | 0.400 |
| mean_crps | 10.710 | 11.065 |
| median_abs_error | 9.852 | 9.884 |

### Per-Scenario Results

| Scenario | Domain | Group | GT | Sim | δ_s | Error | CRPS | 90% CI |
|---|---|---|---|---|---|---|---|---|
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | financial | train | 35.0 | 98.0 | -1.357 | +63.0 | 60.17 | [85.1, 104.5] |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | financial | test | 62.0 | 20.9 |  | -41.1 | 35.51 | [8.3, 40.8] |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | financial | train | 38.0 | 69.3 | -0.965 | +31.3 | 24.39 | [47.6, 86.3] |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | public_health | train | 47.0 | 64.8 | -0.538 | +17.8 | 11.19 | [41.7, 85.6] |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | corporate | train | 41.0 | 26.7 | -0.194 | -14.3 | 10.40 | [11.4, 44.2] |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | political | test | 51.3 | 37.0 |  | -14.3 | 8.59 | [18.5, 58.5] |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | political | train | 50.9 | 37.1 | 0.097 | -13.8 | 9.18 | [19.1, 59.0] |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | political | train | 53.4 | 39.7 | 0.233 | -13.7 | 8.07 | [20.3, 63.5] |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | political | train | 40.0 | 29.0 | 0.040 | -11.0 | 6.46 | [11.6, 45.4] |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | public_health | test | 67.0 | 76.9 |  | +9.9 | 5.93 | [58.0, 90.9] |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | political | train | 66.4 | 56.5 | 0.009 | -9.9 | 5.49 | [33.7, 76.8] |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | political | train | 53.2 | 45.2 | 0.169 | -8.0 | 5.53 | [26.6, 66.9] |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | corporate | train | 28.0 | 35.8 | -0.828 | +7.8 | 3.77 | [18.3, 60.5] |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | political | train | 66.1 | 59.1 | 0.422 | -7.0 | 3.54 | [36.9, 78.1] |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | financial | train | 72.0 | 76.6 | 0.225 | +4.6 | 3.25 | [60.6, 90.4] |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | political | train | 40.9 | 44.6 | -0.169 | +3.7 | 3.90 | [19.9, 67.8] |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | political | test | 48.0 | 46.2 |  | -1.8 | 2.54 | [26.4, 65.7] |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | public_health | train | 63.0 | 61.3 | 0.495 | -1.7 | 2.98 | [37.8, 80.8] |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | financial | train | 22.0 | 20.6 | -0.993 | -1.4 | 2.33 | [9.0, 35.4] |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | corporate | test | 26.0 | 27.1 |  | +1.1 | 2.77 | [12.2, 44.1] |

## Diagnostic: Discrepancy Health Check

**NOTE**: Total discrepancy scale = 0.600 (0.5-1.0 in logit space ≈ 12-25pp). Moderate — check per-scenario δ values.

