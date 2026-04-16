# Grounding v2.3: Hybrid Events + Public Opinion Agent

## Strategy
- v2.2 hybrid dataset (original agents + grounded events)
- Added implicit Public Opinion agent from polling trajectory
  - Position: first polling pro_pct mapped to [-1, +1]
  - Type: citizen (rigidity=0.1, tolerance=0.9, influence=0.7)
  - Very event-reactive, broad social tolerance
- Discrepancy model with δ_s

## Headline Metrics

| Metric | v2 | v2.1 | v2.2 | v2.3 |
|---|---|---|---|---|
| MAE test | 19.2pp | 18.9pp | 11.4pp | 13.6pp |
| MAE train | 14.3pp | 15.1pp | 15.3pp | 13.9pp |
| RMSE test | 26.6pp | 21.6pp | 16.8pp | 20.0pp |
| Coverage 90% tr | 79.4pp | 2.9pp | 73.3pp | 86.7pp |

## Per-Scenario

| Scenario | GT | Sim | Error | in 90% CI |
|---|---|---|---|---|
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG | 28.0 | 35.8 | +7.8 | YES |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER | 26.0 | 27.1 | +1.1 | YES |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON | 41.0 | 26.7 | -14.3 | YES |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 | 72.0 | 76.6 | +4.6 | YES |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL | 62.0 | 20.9 | -41.1 | no |
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC | 35.0 | 98.0 | +63.0 | no |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 | 22.0 | 20.6 | -1.4 | YES |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI | 38.0 | 69.3 | +31.3 | no |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 | 63.0 | 61.3 | -1.7 | YES |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO | 67.0 | 76.9 | +9.9 | YES |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 | 47.0 | 64.8 | +17.8 | YES |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | 53.2 | 45.2 | -8.0 | YES |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | 40.9 | 44.6 | +3.7 | YES |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | 66.1 | 59.1 | -7.0 | YES |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | 48.0 | 46.2 | -1.8 | YES |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | 53.4 | 39.7 | -13.7 | YES |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | 66.4 | 56.5 | -9.9 | YES |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | 40.0 | 29.0 | -11.0 | YES |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | 51.3 | 37.0 | -14.3 | YES |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | 50.9 | 37.1 | -13.8 | YES |

---
*Generated: 2026-04-03 11:02*