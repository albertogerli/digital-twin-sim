# v4 Multi-Modal Calibration (Polling + Financial Markets)

**Generated:** 2026-04-13 09:52

Scenarios with financial data: 14
SVI steps: 2500, Loss: 574.7, Time: 3235s

## Headline Metrics

| Metric | v2 (paper) | v4 (multimodal) Train | v4 Test | v4 Financial |
|---|---|---|---|---|
| MAE | 14.3 / 19.2pp | 13.8pp | 18.2pp | 24.0pp |
| RMSE | 18.8 / 26.6pp | 18.9pp | 25.6pp | 30.7pp |
| Cov90 | 79.4 / 75.0% | 76% | 88% | 57% |

## Financial Linkage Parameters

| Param | Mean | CI95 |
|---|---|---|
| w_opinion | 0.917 | [-1.106, 2.971] |
| w_event | -0.431 | [-2.092, 1.343] |
| w_polar | -0.662 | [-2.398, 1.153] |
| lambda_fin | 0.045 | [0.037, 0.055] |
| sigma_market | 13.033 | [11.084, 15.477] |

## Per-Scenario

| Scenario | Group | GT | Sim | Error | Fin? | 90%CI |
|---|---|---|---|---|---|---|
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | test | 35.0 | 99.7 | +64.7 | Y | no |
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | train | 18.0 | 73.7 | +55.7 | Y | no |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | train | 38.0 | 75.2 | +37.2 | Y | no |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | train | 32.0 | 68.8 | +36.8 | Y | no |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | train | 62.0 | 26.9 | -35.1 | Y | no |
| FIN-2021-GAMESTOP | train | 72.0 | 38.8 | -33.2 | Y | no |
| POL-2020-CHILE_CONSTITUTIONAL_REFERENDU | train | 78.3 | 52.7 | -25.6 |  | no |
| TECH-2020-TIKTOK_US_BAN_DEBATE_2020 | train | 60.0 | 35.6 | -24.4 |  | no |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | train | 62.0 | 80.8 | +18.8 |  | no |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | test | 62.0 | 43.2 | -18.8 |  | YES |
| COM-2017-IPHONE_X | train | 65.0 | 82.3 | +17.3 |  | YES |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | train | 53.4 | 36.6 | -16.8 |  | YES |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | train | 50.9 | 35.1 | -15.8 |  | YES |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF | test | 38.7 | 54.0 | +15.3 |  | YES |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | train | 51.3 | 36.5 | -14.8 |  | YES |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | train | 26.0 | 39.5 | +13.5 | Y | YES |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | train | 65.0 | 52.2 | -12.8 | Y | YES |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | test | 56.0 | 68.5 | +12.5 | Y | YES |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_2017 | test | 83.0 | 70.6 | -12.4 |  | YES |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | train | 72.0 | 83.5 | +11.5 | Y | YES |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | train | 67.0 | 78.4 | +11.4 |  | YES |
| CORP-2019-BOEING_MAX | train | 40.0 | 50.4 | +10.4 | Y | YES |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | train | 40.0 | 29.7 | -10.3 |  | YES |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | train | 66.4 | 56.2 | -10.2 |  | YES |
| SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO | train | 61.6 | 71.5 | +9.9 |  | YES |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | train | 47.0 | 55.9 | +8.9 |  | YES |
| POL-2017-TURKISH_CONSTITUTIONAL_REFEREN | test | 51.4 | 60.0 | +8.6 |  | YES |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | test | 66.1 | 58.4 | -7.7 |  | YES |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | train | 22.0 | 29.4 | +7.4 | Y | YES |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | train | 53.2 | 47.0 | -6.2 |  | YES |
| PH-2021-COVID_VAX_IT | test | 80.0 | 85.5 | +5.5 |  | YES |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | train | 40.9 | 44.9 | +4.0 |  | YES |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | train | 63.0 | 59.3 | -3.7 |  | YES |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | train | 41.0 | 37.5 | -3.5 | Y | YES |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | train | 48.0 | 44.8 | -3.2 |  | YES |
| TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E | train | 67.0 | 70.1 | +3.1 |  | YES |
| ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES | train | 71.0 | 69.0 | -2.0 |  | YES |
| POL-2016-BREXIT | train | 51.9 | 53.6 | +1.7 |  | YES |
| POL-2014-SCOTTISH_INDEPENDENCE_REFEREND | train | 44.7 | 46.2 | +1.5 |  | YES |
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | train | 28.0 | 29.4 | +1.4 | Y | YES |
| LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2 | train | 45.0 | 44.7 | -0.3 |  | YES |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | train | 35.0 | 34.8 | -0.2 |  | YES |
