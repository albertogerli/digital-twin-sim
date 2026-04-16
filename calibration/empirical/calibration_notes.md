# Calibration Notes — Empirical Dataset

**Updated:** 2026-03-29 (post-review)

## Dataset Summary

The empirical dataset has 24 scenarios with verified outcomes but trajectories
per lo più LLM-estimated. This has a direct implication for Phase 2:

## Observation Model Implications

The observation model Beta-Binomial requires real `sample_size` per round.
Only 5-8 scenarios have polling datapoints with verified sample sizes.

### Recommendation: Dual Likelihood Configuration

In the hierarchical calibration (Phase 2.3-2.4), use two configurations:

**(A) FULL LIKELIHOOD** — only on the ~8 scenarios with verified per-round polling.
Use the Beta-Binomial observation model with real sample sizes:
- `POL-2016-BREXIT` (6/6 rounds with NatCen/ICM/Ipsos data)
- `PH-2021-COVID_VAX_IT` (7/7 rounds, all IPSOS N=1000)
- `COM-2017-IPHONE_X` (2/5 rounds with Fluent N=2117, Piper Jaffray N=1500)
- `FIN-2021-GAMESTOP` (1/7 rounds with Morning Consult N=2200)
- `CORP-2019-BOEING_MAX` (1/7 rounds with Reuters/Ipsos N=2000)
- `POL-2020-CHILE_CONSTITUTIONAL_REFERENDU` (4/6 rounds with Cadem N=1200)
- `SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO` (varies)
- `POL-2014-SCOTTISH_INDEPENDENCE_REFEREND` (varies)

**(B) OUTCOME-ONLY LIKELIHOOD** — on all 24 scenarios.
Use only the final datapoint (outcome) with a simplified observation model:
```
y_final ~ Normal(q_final, sigma_outcome)
```
Ignore the intermediate trajectory entirely.

### Validation Strategy

Compare the two resulting parameter sets. If they converge, the LLM-estimated
intermediate data is not distorting calibration. If they diverge, trust only
configuration (A).

## Improving Per-Round Verification

For a dataset with verified trajectories, the most promising domains are:

- **Political:** FiveThirtyEight polling averages, RealClearPolitics,
  Eurobarometer, UK Polling Report (What UK Thinks)
- **Financial:** Bloomberg sentiment indices, Morning Consult brand tracking,
  Harris Poll
- **Social:** Essential Research (Australia), Cadem (Chile), IPSOS Global
  Attitudes monthly tracker

### Action Item

Generate a second batch CSV (`events_batch2.csv`) focused on these domains
with the goal of having 15+ scenarios with per-round verified polling data.
Priority targets:
- US/UK elections with FiveThirtyEight averages (reliable N per round)
- EU referendums with Eurobarometer tracking
- Brand crises with Morning Consult BrandIntelligence data

## Non-Independence Warnings

Two scenario pairs share common events and are NOT independent:

1. **Boeing pair:** `CORP-2019-BOEING_MAX` + `CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI`
2. **Tech-2018 pair:** `TECH-2018-GDPR` + `TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA`

For calibration requiring IID samples, drop one from each pair.
Recommended: keep `CORP-2019-BOEING_MAX` and `TECH-2018-GDPR` (broader scope).
