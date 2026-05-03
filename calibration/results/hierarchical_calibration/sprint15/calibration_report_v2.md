# Hierarchical Calibration Report — v2 (with Discrepancy)

**Generated:** 2026-05-03 10:49

## Model Description

Extension of v1 with additive discrepancy term δ_s in logit space:
- `q_corrected = sigmoid(logit(q_sim) + δ_s)`
- δ_s ~ Normal(δ_d, σ_δ_within) — per scenario
- δ_d ~ Normal(0, σ_δ_between) — per domain
- σ_outcome ~ HalfNormal(3.0) — tight, forces bias into δ

## Phase B: Empirical Fine-tuning

- SVI steps: 3000
- Final loss: 493.8
- Elapsed: 2436.3s

### Calibrated Parameters

- μ_global: [-0.250, 0.432, -0.206, -0.181]
- σ_global: [0.122, 0.120, 0.127, 0.127]

| Parameter | Mean | CI95 Low | CI95 High |
|---|---|---|---|
| alpha_herd | -0.250 | -0.377 | -0.123 |
| alpha_anchor | 0.432 | 0.291 | 0.572 |
| alpha_social | -0.206 | -0.339 | -0.076 |
| alpha_event | -0.181 | -0.305 | -0.050 |

### Discrepancy Parameters

- σ_δ_between: 0.166 (CI: 0.104, 0.250)
- σ_δ_within: 0.575 (CI: 0.455, 0.712)

### Per-Domain Discrepancy (δ_d)

| Domain | δ_d Mean | CI95 Low | CI95 High | Interpretation |
|---|---|---|---|---|
| commercial | -0.053 | -0.352 | 0.232 | No significant bias |
| corporate | -0.070 | -0.320 | 0.173 | No significant bias |
| energy | -0.030 | -0.334 | 0.282 | No significant bias |
| environmental | 0.031 | -0.265 | 0.322 | No significant bias |
| financial | -0.115 | -0.362 | 0.130 | No significant bias |
| labor | -0.008 | -0.299 | 0.289 | No significant bias |
| political | 0.053 | -0.148 | 0.262 | No significant bias |
| public_health | -0.047 | -0.304 | 0.227 | No significant bias |
| social | 0.027 | -0.302 | 0.350 | No significant bias |
| technology | -0.029 | -0.297 | 0.256 | No significant bias |

## Phase C: Validation

| Metric | Train | Test |
|---|---|---|
| n | 34.000 | 8.000 |
| mae | 13.969 | 17.555 |
| rmse | 18.829 | 25.398 |
| coverage_90 | 0.824 | 0.875 |
| coverage_50 | 0.441 | 0.375 |
| mean_crps | 10.470 | 14.093 |
| median_abs_error | 10.547 | 11.899 |

### Per-Scenario Results

| Scenario | Domain | Group | GT | Sim | δ_s | Error | CRPS | 90% CI |
|---|---|---|---|---|---|---|---|---|
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | financial | test | 35.0 | 99.9 |  | +64.9 | 63.35 | [95.0, 105.0] |
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | financial | train | 18.0 | 73.4 | -1.487 | +55.4 | 47.05 | [50.6, 92.8] |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | corporate | train | 32.0 | 68.8 | -0.447 | +36.8 | 29.32 | [46.3, 88.5] |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | financial | train | 38.0 | 74.5 | -1.080 | +36.5 | 29.88 | [51.1, 90.4] |
| FIN-2021-GAMESTOP | financial | train | 72.0 | 38.2 | 0.889 | -33.8 | 25.65 | [16.4, 62.0] |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | financial | train | 62.0 | 28.4 | 0.774 | -33.6 | 26.12 | [9.2, 50.6] |
| POL-2020-CHILE_CONSTITUTIONAL_REFERENDU | political | train | 78.3 | 52.4 | 0.776 | -25.9 | 17.65 | [29.7, 73.3] |
| TECH-2020-TIKTOK_US_BAN_DEBATE_2020 | technology | train | 60.0 | 38.5 | 0.260 | -21.5 | 14.40 | [16.0, 60.3] |
| COM-2017-IPHONE_X | commercial | train | 65.0 | 84.6 | -0.453 | +19.6 | 13.24 | [60.1, 102.4] |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | public_health | train | 62.0 | 80.5 | -0.433 | +18.5 | 13.38 | [58.2, 96.1] |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | commercial | test | 62.0 | 45.0 |  | -17.0 | 11.07 | [23.3, 68.9] |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | political | train | 53.4 | 36.6 | 0.314 | -16.8 | 9.64 | [12.7, 62.7] |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | political | train | 50.9 | 34.1 | 0.139 | -16.8 | 10.71 | [15.2, 56.7] |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | political | train | 51.3 | 35.1 | 0.272 | -16.2 | 10.28 | [13.7, 63.1] |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF | political | test | 38.7 | 54.5 |  | +15.8 | 9.71 | [32.0, 77.7] |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | public_health | train | 67.0 | 80.4 | -0.369 | +13.4 | 9.24 | [64.8, 93.8] |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | corporate | train | 26.0 | 39.0 | -0.557 | +13.0 | 6.38 | [15.2, 69.4] |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | corporate | train | 65.0 | 52.5 | 0.497 | -12.5 | 6.78 | [29.3, 74.9] |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_2017 | technology | test | 83.0 | 70.5 |  | -12.5 | 7.36 | [49.4, 87.8] |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | corporate | test | 56.0 | 67.3 |  | +11.3 | 7.52 | [43.8, 87.9] |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | public_health | train | 47.0 | 57.9 | -0.644 | +10.9 | 6.77 | [35.1, 78.9] |
| SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO | social | train | 61.6 | 72.2 | -0.191 | +10.6 | 7.06 | [50.5, 89.3] |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | financial | train | 72.0 | 82.5 | 0.090 | +10.5 | 6.56 | [69.7, 94.2] |
| CORP-2019-BOEING_MAX | corporate | train | 40.0 | 50.0 | 0.064 | +10.0 | 6.38 | [26.8, 72.7] |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | political | train | 66.4 | 56.4 | -0.028 | -10.0 | 5.28 | [31.3, 78.3] |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | political | train | 40.0 | 31.0 | 0.074 | -9.0 | 6.04 | [12.7, 54.1] |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | political | train | 53.2 | 44.2 | 0.162 | -9.0 | 5.08 | [21.7, 69.6] |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | political | test | 66.1 | 58.9 |  | -7.2 | 4.21 | [34.8, 79.2] |
| POL-2017-TURKISH_CONSTITUTIONAL_REFEREN | political | test | 51.4 | 57.9 |  | +6.5 | 5.01 | [34.3, 79.2] |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | financial | train | 22.0 | 28.1 | -0.968 | +6.1 | 4.34 | [10.6, 50.2] |
| PH-2021-COVID_VAX_IT | public_health | test | 80.0 | 85.2 |  | +5.2 | 4.53 | [63.0, 100.1] |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | corporate | train | 28.0 | 32.9 | -0.835 | +4.9 | 3.45 | [13.9, 53.9] |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | political | train | 40.9 | 45.5 | -0.155 | +4.6 | 3.69 | [25.6, 69.6] |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | corporate | train | 41.0 | 36.4 | -0.192 | -4.6 | 3.69 | [18.1, 58.8] |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | energy | train | 35.0 | 39.2 | -0.568 | +4.2 | 4.20 | [16.2, 67.0] |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | political | train | 48.0 | 45.2 | -0.207 | -2.8 | 2.89 | [23.5, 66.8] |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | public_health | train | 63.0 | 60.5 | 0.474 | -2.5 | 4.17 | [36.5, 83.0] |
| TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E | technology | train | 67.0 | 69.2 | -0.058 | +2.2 | 3.60 | [47.0, 87.2] |
| POL-2016-BREXIT | political | train | 51.9 | 53.5 | -0.001 | +1.6 | 3.76 | [29.2, 75.8] |
| POL-2014-SCOTTISH_INDEPENDENCE_REFEREND | political | train | 44.7 | 45.4 | -0.258 | +0.7 | 3.40 | [23.0, 70.4] |
| LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2 | labor | train | 45.0 | 45.5 | -0.384 | +0.5 | 3.55 | [21.1, 66.1] |
| ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES | environmental | train | 71.0 | 71.0 | 0.271 | -0.0 | 2.34 | [52.2, 88.2] |

## Diagnostic: Discrepancy Health Check

**NOTE**: Total discrepancy scale = 0.598 (0.5-1.0 in logit space ≈ 12-25pp). Moderate — check per-scenario δ values.

