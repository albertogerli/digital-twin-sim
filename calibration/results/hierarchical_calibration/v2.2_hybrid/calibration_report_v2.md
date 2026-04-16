# Hierarchical Calibration Report — v2 (with Discrepancy)

**Generated:** 2026-04-02 00:51

## Model Description

Extension of v1 with additive discrepancy term δ_s in logit space:
- `q_corrected = sigmoid(logit(q_sim) + δ_s)`
- δ_s ~ Normal(δ_d, σ_δ_within) — per scenario
- δ_d ~ Normal(0, σ_δ_between) — per domain
- σ_outcome ~ HalfNormal(3.0) — tight, forces bias into δ

## Phase B: Empirical Fine-tuning

- SVI steps: 3000
- Final loss: 230.6
- Elapsed: 5729.1s

### Calibrated Parameters

- μ_global: [-0.308, 0.581, -0.201, -0.446]
- σ_global: [0.228, 0.230, 0.248, 0.244]

| Parameter | Mean | CI95 Low | CI95 High |
|---|---|---|---|
| alpha_herd | -0.308 | -0.616 | -0.001 |
| alpha_anchor | 0.581 | 0.322 | 0.847 |
| alpha_social | -0.201 | -0.512 | 0.099 |
| alpha_event | -0.446 | -0.762 | -0.138 |

### Discrepancy Parameters

- σ_δ_between: 0.358 (CI: 0.185, 0.605)
- σ_δ_within: 0.527 (CI: 0.383, 0.692)

### Per-Domain Discrepancy (δ_d)

| Domain | δ_d Mean | CI95 Low | CI95 High | Interpretation |
|---|---|---|---|---|
| corporate | -0.207 | -0.662 | 0.274 | No significant bias |
| financial | -0.435 | -0.837 | -0.025 | Sim over-predicts |
| political | 0.041 | -0.286 | 0.345 | No significant bias |
| public_health | 0.002 | -0.485 | 0.491 | No significant bias |

## Phase C: Validation

| Metric | Train | Test |
|---|---|---|
| n | 15.000 | 5.000 |
| mae | 15.344 | 11.385 |
| rmse | 21.818 | 16.810 |
| coverage_90 | 0.733 | 0.800 |
| coverage_50 | 0.333 | 0.600 |
| mean_crps | 11.874 | 9.248 |
| median_abs_error | 10.157 | 7.226 |

### Per-Scenario Results

| Scenario | Domain | Group | GT | Sim | δ_s | Error | CRPS | 90% CI |
|---|---|---|---|---|---|---|---|---|
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | financial | train | 35.0 | 97.0 | -1.309 | +62.0 | 58.48 | [79.2, 104.4] |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | financial | test | 62.0 | 26.7 |  | -35.3 | 28.64 | [10.4, 48.8] |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | financial | train | 38.0 | 69.6 | -0.993 | +31.6 | 24.45 | [48.9, 88.4] |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | public_health | train | 47.0 | 75.2 | -0.749 | +28.2 | 20.07 | [49.9, 95.0] |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | corporate | train | 28.0 | 53.5 | -0.953 | +25.5 | 17.85 | [29.2, 75.0] |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | political | train | 66.4 | 51.6 | 0.030 | -14.8 | 8.44 | [29.2, 72.8] |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | political | train | 66.1 | 51.7 | 0.502 | -14.4 | 8.14 | [27.3, 74.2] |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | political | train | 40.9 | 51.1 | -0.262 | +10.2 | 6.25 | [26.4, 70.6] |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | political | train | 50.9 | 40.7 | 0.065 | -10.2 | 6.68 | [20.6, 64.8] |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | public_health | test | 67.0 | 76.9 |  | +9.9 | 6.39 | [59.1, 91.1] |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | corporate | train | 41.0 | 32.5 | -0.256 | -8.5 | 6.38 | [14.8, 54.6] |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | political | train | 40.0 | 31.7 | -0.055 | -8.3 | 4.70 | [11.1, 50.7] |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | corporate | test | 26.0 | 33.2 |  | +7.2 | 4.95 | [14.6, 53.2] |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | public_health | train | 63.0 | 55.9 | 0.629 | -7.1 | 4.54 | [30.2, 78.4] |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | financial | train | 22.0 | 25.8 | -1.079 | +3.8 | 2.80 | [11.9, 44.6] |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | political | test | 48.0 | 51.5 |  | +3.5 | 3.32 | [29.1, 72.8] |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | political | train | 53.4 | 50.7 | 0.174 | -2.7 | 2.74 | [28.8, 70.5] |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | financial | train | 72.0 | 74.1 | 0.350 | +2.1 | 2.68 | [55.5, 90.0] |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | political | test | 51.3 | 50.4 |  | -0.9 | 2.94 | [27.7, 71.2] |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | political | train | 53.2 | 52.3 | 0.113 | -0.9 | 3.91 | [29.8, 76.4] |

## Diagnostic: Discrepancy Health Check

**NOTE**: Total discrepancy scale = 0.637 (0.5-1.0 in logit space ≈ 12-25pp). Moderate — check per-scenario δ values.

