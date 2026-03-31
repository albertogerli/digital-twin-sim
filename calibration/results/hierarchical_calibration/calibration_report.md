# Hierarchical Calibration Report

**Generated:** 2026-03-30 10:44

## Phase A: Synthetic Pre-training

- Scenarios: 50
- Domains: 11 (commercial, corporate, corporate/political, environmental, financial, labor, marketing, political, public_health, social, technology)
- SVI steps: 2000
- Final loss: 363.9
- Elapsed: 1624.7s
- μ_global (synthetic): [-0.320, 0.639, -0.219, -0.570]
- σ_global (synthetic): [0.108, 0.112, 0.121, 0.119]

## Phase B: Empirical Fine-tuning

- Train scenarios: 16
- Test scenarios: 6 (COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI, CORP-2019-BOEING_MAX, FIN-2021-GAMESTOP, PH-2021-COVID_VAX_IT, POL-2017-TURKISH_CONSTITUTIONAL_REFEREN, TECH-2017-NET_NEUTRALITY_REPEAL_US_2017)
- Domains: commercial, corporate, energy, environmental, financial, labor, political, public_health, social, technology
- SVI steps: 3000
- Final loss: 261.1
- Elapsed: 3037.1s

### Calibrated Parameters

- μ_global: [-0.258, 0.438, -0.178, -0.326]
- σ_global: [0.156, 0.151, 0.156, 0.154]

| Parameter | Mean | CI95 Low | CI95 High |
|---|---|---|---|
| alpha_herd | -0.258 | -0.346 | -0.169 |
| alpha_anchor | 0.438 | 0.321 | 0.565 |
| alpha_social | -0.178 | -0.292 | -0.063 |
| alpha_event | -0.326 | -0.433 | -0.220 |

### Observation Parameters

- τ_readout: 0.7365
- φ (BetaBinomial): 10.4

### Significant Covariates (95% CI excludes 0)

- `alpha_event` ← `institutional_trust`: -0.225 (CI: -0.421, -0.033)

## Phase C: Validation

| Metric | Train | Test |
|---|---|---|
| n | 16 | 6 |
| mae | 15.772 | 12.301 |
| rmse | 21.028 | 14.591 |
| coverage_90 | 0.438 | 0.667 |
| coverage_50 | 0.125 | 0.000 |
| mean_crps | 13.548 | 9.041 |
| median_abs_error | 10.661 | 9.543 |

### Per-Scenario Results

| Scenario | Domain | Group | GT | Sim | Error | CRPS | 90% CI |
|---|---|---|---|---|---|---|---|
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | financial | train | 18.0 | 68.6 | +50.6 | 46.20 | [52.7, 94.6] |
| POL-2020-CHILE_CONSTITUTIONAL_REFERENDU | political | train | 78.3 | 37.5 | -40.8 | 37.25 | [28.0, 47.8] |
| FIN-2021-GAMESTOP | financial | test | 72.0 | 44.4 | -27.6 | 23.99 | [38.5, 54.8] |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF | political | train | 38.7 | 65.9 | +27.2 | 24.26 | [56.6, 74.8] |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | corporate | train | 32.0 | 59.1 | +27.1 | 22.36 | [45.8, 69.6] |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | public_health | train | 62.0 | 83.3 | +21.3 | 19.20 | [75.1, 96.7] |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_2017 | technology | test | 83.0 | 66.7 | -16.3 | 13.38 | [58.7, 76.0] |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | corporate | train | 65.0 | 50.0 | -15.0 | 11.61 | [41.4, 60.2] |
| LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2 | labor | train | 45.0 | 59.8 | +14.8 | 11.07 | [47.8, 77.5] |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | commercial | test | 62.0 | 50.2 | -11.8 | 4.96 | [42.5, 68.7] |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | corporate | train | 56.0 | 66.7 | +10.7 | 8.01 | [58.0, 77.4] |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | energy | train | 35.0 | 45.7 | +10.7 | 6.44 | [33.5, 57.6] |
| TECH-2020-TIKTOK_US_BAN_DEBATE_2020 | technology | train | 60.0 | 49.5 | -10.5 | 10.10 | [32.9, 59.8] |
| PH-2021-COVID_VAX_IT | public_health | test | 80.0 | 72.7 | -7.3 | 4.96 | [64.2, 82.6] |
| COM-2017-IPHONE_X | commercial | train | 65.0 | 71.4 | +6.4 | 5.96 | [64.6, 85.4] |
| ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES | environmental | train | 71.0 | 65.1 | -5.9 | 4.50 | [51.0, 75.5] |
| POL-2017-TURKISH_CONSTITUTIONAL_REFEREN | political | test | 51.4 | 57.1 | +5.7 | 3.40 | [47.3, 66.4] |
| POL-2014-SCOTTISH_INDEPENDENCE_REFEREND | political | train | 44.7 | 50.0 | +5.3 | 3.71 | [40.1, 60.7] |
| CORP-2019-BOEING_MAX | corporate | test | 40.0 | 45.1 | +5.1 | 3.55 | [33.4, 57.5] |
| SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO | social | train | 61.6 | 66.7 | +5.1 | 3.90 | [59.1, 77.1] |
| POL-2016-BREXIT | political | train | 51.9 | 52.4 | +0.5 | 1.15 | [42.2, 62.1] |
| TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E | technology | train | 67.0 | 66.7 | -0.3 | 1.04 | [58.4, 81.2] |

### Domain Breakdown

| Domain | N | MAE | Coverage 90% |
|---|---|---|---|
| commercial | 2 | 9.1 | 100% |
| corporate | 4 | 14.5 | 25% |
| energy | 1 | 10.7 | 100% |
| environmental | 1 | 5.9 | 100% |
| financial | 2 | 39.1 | 0% |
| labor | 1 | 14.8 | 0% |
| political | 5 | 15.9 | 60% |
| public_health | 2 | 14.3 | 50% |
| social | 1 | 5.1 | 100% |
| technology | 3 | 9.1 | 33% |
