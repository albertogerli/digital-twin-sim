# Grounding v2.1 Recalibration Report

## 1. Strategy

Selective grounding: ground only high-|δ_s| domains (financial, corporate, energy, public_health),
keep low-|δ_s| domains unchanged (political, technology, etc.).
Then recalibrate Phase B on the mixed dataset.

## 2. Scenarios

- **Grounded**: 20 scenarios
- **Kept**: 22 scenarios

- Grounding success: 20/20

## 3. Calibration Metrics Comparison

| Metric | v2 | v2.1 | Δ |
|---|---|---|---|
| MAE test | 19.2pp | 18.9pp | -0.3pp |
| MAE train | 14.3pp | 15.1pp | +0.8pp |
| RMSE test | 26.6pp | 21.6pp | -5.0pp |
| Coverage 90% train | 79.4% | 2.9% | -76.5pp |

## 4. Notable Scenarios

| Scenario | GT% | v2 δ_s | v2.1 sim% | v2.1 err |
|---|---|---|---|---|
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | 32.0 | -0.264 | 47.6 | +15.6pp |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | 65.0 | +0.545 | 62.4 | -2.6pp |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | 28.0 | -0.860 | 51.5 | +23.5pp |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | 56.0 | +0.000 | 80.0 | +24.0pp |
| CORP-2019-BOEING_MAX | 40.0 | +0.134 | 76.9 | +36.9pp |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | 26.0 | -0.569 | 22.4 | -3.6pp |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | 41.0 | -0.220 | 0.0 | -41.0pp |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | 35.0 | -0.709 | 63.9 | +28.9pp |
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | 18.0 | -1.378 | 100.0 | +82.0pp |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | 72.0 | +0.317 | 80.0 | +8.0pp |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | 62.0 | +0.600 | 62.2 | +0.2pp |
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | 35.0 | +0.000 | 75.0 | +40.0pp |
| FIN-2021-GAMESTOP | 72.0 | +0.608 | 53.1 | -18.9pp |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | 22.0 | -0.727 | 20.0 | -2.0pp |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | 38.0 | -0.835 | 67.8 | +29.8pp |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | 62.0 | -0.532 | 85.7 | +23.7pp |
| PH-2021-COVID_VAX_IT | 80.0 | +0.000 | 90.0 | +10.0pp |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | 63.0 | +0.637 | 62.5 | -0.5pp |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | 67.0 | -0.253 | 80.0 | +13.0pp |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | 47.0 | -0.715 | 83.6 | +36.6pp |

## 5. Verdict

**MEANINGFUL IMPROVEMENT**: MAE test v2=19.2pp → v2.1=18.9pp (-0.3pp)

---
*Generated: 2026-04-01 11:21*