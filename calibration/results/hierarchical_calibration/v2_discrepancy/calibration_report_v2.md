# Hierarchical Calibration Report — v2 (with Discrepancy)

**Generated:** 2026-03-31 02:03

## Model Description

Extension of v1 with additive discrepancy term δ_s in logit space:
- `q_corrected = sigmoid(logit(q_sim) + δ_s)`
- δ_s ~ Normal(δ_d, σ_δ_within) — per scenario
- δ_d ~ Normal(0, σ_δ_between) — per domain
- σ_outcome ~ HalfNormal(3.0) — tight, forces bias into δ

## Phase B: Empirical Fine-tuning

- SVI steps: 3000
- Final loss: 514.7
- Elapsed: 2411.2s

### Calibrated Parameters

- μ_global: [-0.176, 0.297, -0.105, -0.130]
- σ_global: [0.135, 0.147, 0.144, 0.143]

| Parameter | Mean | CI95 Low | CI95 High |
|---|---|---|---|
| alpha_herd | -0.176 | -0.265 | -0.079 |
| alpha_anchor | 0.297 | 0.199 | 0.401 |
| alpha_social | -0.105 | -0.202 | -0.005 |
| alpha_event | -0.130 | -0.227 | -0.033 |

### Discrepancy Parameters

- σ_δ_between: 0.115 (CI: 0.073, 0.173)
- σ_δ_within: 0.558 (CI: 0.436, 0.696)

### Per-Domain Discrepancy (δ_d)

| Domain | δ_d Mean | CI95 Low | CI95 High | Interpretation |
|---|---|---|---|---|
| commercial | -0.021 | -0.238 | 0.176 | No significant bias |
| corporate | -0.040 | -0.221 | 0.139 | No significant bias |
| energy | -0.026 | -0.237 | 0.186 | No significant bias |
| environmental | 0.010 | -0.197 | 0.211 | No significant bias |
| financial | -0.054 | -0.242 | 0.126 | No significant bias |
| labor | -0.011 | -0.219 | 0.195 | No significant bias |
| political | 0.019 | -0.145 | 0.183 | No significant bias |
| public_health | -0.032 | -0.223 | 0.164 | No significant bias |
| social | 0.023 | -0.186 | 0.235 | No significant bias |
| technology | -0.021 | -0.204 | 0.174 | No significant bias |

## Phase C: Validation

| Metric | Train | Test |
|---|---|---|
| n | 34.000 | 8.000 |
| mae | 14.293 | 19.175 |
| rmse | 18.769 | 26.604 |
| coverage_90 | 0.794 | 0.750 |
| coverage_50 | 0.441 | 0.375 |
| mean_crps | 10.571 | 15.445 |
| median_abs_error | 10.113 | 11.870 |

### Per-Scenario Results

| Scenario | Domain | Group | GT | Sim | δ_s | Error | CRPS | 90% CI |
|---|---|---|---|---|---|---|---|---|
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | financial | test | 35.0 | 100.0 |  | +65.0 | 63.59 | [95.1, 105.0] |
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | financial | train | 18.0 | 66.5 | -1.378 | +48.5 | 40.24 | [44.6, 87.7] |
| POL-2020-CHILE_CONSTITUTIONAL_REFERENDU | political | train | 78.3 | 36.8 | 0.816 | -41.5 | 33.83 | [17.2, 60.6] |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | financial | train | 38.0 | 76.1 | -0.835 | +38.1 | 31.02 | [53.9, 93.6] |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | public_health | train | 47.0 | 78.5 | -0.715 | +31.5 | 25.13 | [57.3, 97.2] |
| FIN-2021-GAMESTOP | financial | train | 72.0 | 43.5 | 0.608 | -28.5 | 20.50 | [20.3, 66.1] |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | financial | train | 62.0 | 33.8 | 0.600 | -28.2 | 20.73 | [13.3, 55.4] |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | corporate | train | 32.0 | 59.2 | -0.264 | +27.2 | 19.30 | [35.4, 82.3] |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF | political | test | 38.7 | 65.3 |  | +26.6 | 19.76 | [42.7, 84.5] |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | public_health | train | 62.0 | 82.5 | -0.532 | +20.5 | 14.95 | [61.7, 100.1] |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | corporate | train | 26.0 | 45.8 | -0.569 | +19.8 | 11.10 | [19.9, 75.7] |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | corporate | train | 65.0 | 48.1 | 0.545 | -16.9 | 9.90 | [25.9, 68.7] |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_2017 | technology | test | 83.0 | 66.3 |  | -16.7 | 10.60 | [45.2, 85.6] |
| LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2 | labor | train | 45.0 | 61.6 | -0.440 | +16.6 | 10.92 | [33.1, 86.4] |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | political | train | 66.4 | 50.1 | 0.032 | -16.3 | 9.48 | [26.1, 71.5] |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | corporate | train | 28.0 | 43.3 | -0.860 | +15.3 | 9.28 | [21.3, 67.4] |
| TECH-2020-TIKTOK_US_BAN_DEBATE_2020 | technology | train | 60.0 | 45.1 | 0.095 | -14.9 | 8.97 | [23.4, 67.7] |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | political | test | 66.1 | 51.6 |  | -14.5 | 8.33 | [28.2, 73.0] |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | political | train | 50.9 | 36.7 | 0.072 | -14.2 | 8.89 | [17.9, 58.7] |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | public_health | train | 67.0 | 80.5 | -0.253 | +13.5 | 9.31 | [64.9, 93.8] |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | financial | train | 22.0 | 33.0 | -0.727 | +11.0 | 6.65 | [14.4, 55.9] |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | political | train | 40.9 | 50.2 | -0.179 | +9.3 | 5.74 | [31.2, 71.7] |
| PH-2021-COVID_VAX_IT | public_health | test | 80.0 | 70.8 |  | -9.2 | 5.28 | [51.9, 88.3] |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | financial | train | 72.0 | 80.9 | 0.317 | +8.9 | 5.31 | [68.3, 93.1] |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | commercial | test | 62.0 | 54.1 |  | -7.9 | 6.01 | [32.5, 76.6] |
| ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES | environmental | train | 71.0 | 63.5 | 0.303 | -7.5 | 3.71 | [42.8, 83.3] |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | corporate | test | 56.0 | 63.3 |  | +7.3 | 5.34 | [39.6, 84.1] |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | public_health | train | 63.0 | 55.9 | 0.637 | -7.1 | 5.50 | [33.1, 79.0] |
| COM-2017-IPHONE_X | commercial | train | 65.0 | 71.3 | -0.286 | +6.3 | 4.13 | [49.1, 88.8] |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | energy | train | 35.0 | 41.2 | -0.709 | +6.2 | 4.55 | [18.9, 68.3] |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | political | train | 40.0 | 33.8 | -0.108 | -6.2 | 4.56 | [15.1, 57.4] |
| POL-2017-TURKISH_CONSTITUTIONAL_REFEREN | political | test | 51.4 | 57.5 |  | +6.1 | 4.66 | [33.6, 78.8] |
| SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO | social | train | 61.6 | 67.4 | -0.029 | +5.8 | 4.45 | [44.4, 85.3] |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | corporate | train | 41.0 | 45.9 | -0.220 | +4.9 | 3.63 | [27.1, 66.7] |
| POL-2014-SCOTTISH_INDEPENDENCE_REFEREND | political | train | 44.7 | 49.4 | -0.318 | +4.7 | 3.83 | [27.0, 73.8] |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | political | train | 53.4 | 48.7 | 0.171 | -4.7 | 3.02 | [24.4, 72.8] |
| POL-2016-BREXIT | political | train | 51.9 | 54.7 | -0.008 | +2.8 | 3.58 | [31.8, 75.8] |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | political | train | 48.0 | 50.7 | -0.263 | +2.7 | 2.71 | [28.8, 71.3] |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | political | train | 51.3 | 48.8 | 0.143 | -2.5 | 3.73 | [25.9, 74.0] |
| CORP-2019-BOEING_MAX | corporate | train | 40.0 | 41.9 | 0.134 | +1.9 | 4.29 | [13.2, 66.6] |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | political | train | 53.2 | 51.9 | 0.069 | -1.3 | 3.35 | [26.9, 74.0] |
| TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E | technology | train | 67.0 | 66.1 | 0.036 | -0.9 | 3.15 | [43.6, 84.4] |

## Diagnostic: Discrepancy Health Check

**NOTE**: Total discrepancy scale = 0.570 (0.5-1.0 in logit space ≈ 12-25pp). Moderate — check per-scenario δ values.

