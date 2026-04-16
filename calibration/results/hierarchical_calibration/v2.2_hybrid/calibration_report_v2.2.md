# Grounding v2.2: Hybrid Grounding + Discrepancy Model

## Strategy

- **Agents**: Keep original (human-calibrated positions, influence, rigidity)
- **Events**: Replace with grounded verified events from Google Search
- **Model**: hierarchical_model_v2 with discrepancy δ_s (not transfer model)

- Hybrid scenarios: 11
- Total scenarios: 20
- Train: 15, Test: 5

## Headline Metrics

| Metric | v2 | v2.1 | v2.2 | Best? |
|---|---|---|---|---|
| MAE test | 19.2pp | 18.9pp | 11.4pp | v2.2 |
| MAE train | 14.3pp | 15.1pp | 15.3pp | |
| RMSE test | 26.6pp | 21.6pp | 16.8pp | v2.2 |
| Coverage 90% train | 79.4% | 2.9% | 73.3% | v2 |

## Discrepancy

| Metric | v2 | v2.2 | Δ |
|---|---|---|---|
| σ_b,between | 0.115 | 0.358 | +0.243 |
| σ_b,within | 0.558 | 0.527 | -0.031 |

## Per-Scenario Results

| Scenario | Group | GT% | Sim% | Err | δ_s v2 | δ_s v2.2 | in 90%CI |
|---|---|---|---|---|---|---|---|
| CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG (H) | train | 28.0 | 53.5 | +25.5 | -0.860 | -0.953 | no |
| CORP-2021-FACEBOOK_META_REBRAND_OCTOBER (H) | test | 26.0 | 33.2 | +7.2 | -0.569 | +0.000 | YES |
| CORP-2022-TWITTER_X_ACQUISITION_BY_ELON (H) | train | 41.0 | 32.5 | -8.5 | -0.220 | -0.256 | YES |
| FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020 (H) | train | 72.0 | 74.1 | +2.1 | +0.317 | +0.350 | YES |
| FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL (H) | test | 62.0 | 26.7 | -35.3 | +0.600 | +0.000 | no |
| FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC (H) | train | 35.0 | 97.0 | +62.0 | +0.000 | -1.309 | no |
| FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202 (H) | train | 22.0 | 25.8 | +3.8 | -0.727 | -1.079 | YES |
| FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI (H) | train | 38.0 | 69.6 | +31.6 | -0.835 | -0.993 | no |
| PH-2021-MASKING_MANDATE_DEBATE_USA_202 (H) | train | 63.0 | 55.9 | -7.1 | +0.637 | +0.629 | YES |
| PH-2021-VACCINE_HESITANCY_USA_2021_CO (H) | test | 67.0 | 76.9 | +9.9 | -0.253 | +0.000 | YES |
| PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2 (H) | train | 47.0 | 75.2 | +28.2 | -0.715 | -0.749 | no |
| POL-2011-REFERENDUM_DIVORZIO_MALTA_2011 | train | 53.2 | 52.3 | -0.9 | +0.069 | +0.113 | YES |
| POL-2016-REFERENDUM_COSTITUZIONALE_ITAL | train | 40.9 | 51.1 | +10.2 | -0.179 | -0.262 | YES |
| POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA | train | 66.1 | 51.7 | -14.4 | +0.000 | +0.502 | YES |
| POL-2017-REFERENDUM_INDIPENDENZA_CATALU | test | 48.0 | 51.5 | +3.5 | -0.263 | +0.000 | YES |
| POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU | train | 53.4 | 50.7 | -2.7 | +0.171 | +0.174 | YES |
| POL-2018-REFERENDUM_ABORTO_IRLANDA_2018 | train | 66.4 | 51.6 | -14.8 | +0.032 | +0.030 | YES |
| POL-2019-ELEZIONI_EUROPEE_2019_ITALIA | train | 40.0 | 31.7 | -8.3 | -0.108 | -0.055 | YES |
| POL-2020-ELEZIONI_PRESIDENZIALI_USA_202 | test | 51.3 | 50.4 | -0.9 | +0.143 | +0.000 | YES |
| POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE | train | 50.9 | 40.7 | -10.2 | +0.072 | +0.065 | YES |

## Verdict

- MAE test improved: 19.2→11.4pp (better than v2.1's 18.9)
- Coverage 90% restored: 73.3% (v2=79.4%, v2.1=2.9%)
- σ_b,within reduced: 0.558→0.527
- Improved δ_s: 4/11 grounded scenarios
- Mean |δ_s| grounded: v2=0.521 → v2.2=0.574 (+0.053)

---
*Generated: 2026-04-02 08:28*