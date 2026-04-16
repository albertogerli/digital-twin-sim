# Review Summary — Empirical Calibration Scenarios

**Review date:** 2026-03-29
**Total scenarios:** 24
**Verdicts:** 20 PASS, 4 CORRECTIONS, 0 REJECT

## Scenario Results

| Scenario | Verdict | Corrections | Quality Before | Quality After |
|----------|---------|-------------|---------------|--------------|
| COM-2017-IPHONE_X | PASS | - | 100/A | 100/A |
| COM-2019-TESLA_CYBERTRUCK_REVEAL_REACTI | PASS | - | 100/A | 100/A |
| CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW | PASS | - | 100/A | 100/A |
| CORP-2017-UBER_LONDON_LICENSE_BATTLE_201 | PASS | Meta: undecided_interpolated = true | 100/A | 97/A (undecided_interpolated: -3) |
| CORP-2018-AMAZON_HQ2_NYC_BACKLASH_2018_2 | PASS | - | 100/A | 100/A |
| CORP-2019-BOEING_MAX | PASS | - | 100/A | 90/A (overlap with another scenario: -10) |
| CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI | PASS | - | 100/A | 90/A (overlap with another scenario: -10) |
| ENE-2012-JAPANESE_NUCLEAR_RESTART_AFTER | PASS | - | 100/A | 100/A |
| ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES | PASS | - | 100/A | 100/A |
| FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC | PASS | - | 100/A | 100/A |
| FIN-2021-GAMESTOP | CORRECTIONS | Appended reviewer note to notes field; Meta: ground_truth... | 95/A | 90/A (partially_verified ground truth: -5) |
| LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2 | CORRECTIONS | Event R1: shock_direction -0.5 -> 0.3 | 100/A | 100/A |
| PH-2021-ASTRAZENECA_VACCINE_HESITANCY | PASS | - | 100/A | 100/A |
| PH-2021-COVID_VAX_IT | PASS | - | 100/A | 100/A |
| POL-2014-SCOTTISH_INDEPENDENCE_REFEREND | CORRECTIONS | Event R6: date 2014-08-25 -> 2014-09-06; Event R6: round ... | 100/A | 100/A |
| POL-2015-GREEK_BAILOUT_REFERENDUM_GREF | CORRECTIONS | Event R1: date 2015-06-26 -> 2015-06-27 | 100/A | 100/A |
| POL-2016-BREXIT | PASS | - | 100/A | 100/A |
| POL-2017-TURKISH_CONSTITUTIONAL_REFEREN | PASS | - | 100/A | 100/A |
| POL-2020-CHILE_CONSTITUTIONAL_REFERENDU | PASS | - | 100/A | 100/A |
| SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE_PO | PASS | - | 100/A | 100/A |
| TECH-2017-NET_NEUTRALITY_REPEAL_US_2017 | PASS | - | 100/A | 100/A |
| TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S | PASS | Meta: undecided_interpolated = true | 100/A | 87/A (undecided_interpolated: -3, overlap with another scenario: -10) |
| TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E | PASS | - | 100/A | 90/A (overlap with another scenario: -10) |
| TECH-2020-TIKTOK_US_BAN_DEBATE_2020 | PASS | - | 100/A | 100/A |

## Scenario Overlaps (Non-Independent Pairs)

- **CORP-2019-BOEING_MAX + CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI**
  Cover different phases of the same Boeing 737 MAX crisis. If used together in calibration, they are not independent. Options: (a) use only one, (b) treat as multi-phase scenario, (c) document dependency and use both consciously.

- **TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E + TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S**
  Share the Cambridge Analytica scandal (March 17, 2018) as a common shock event. Not independent for calibration purposes.

## Data Quality Warnings

- **Interpolated undecided:** CORP-2017-UBER_LONDON_LICENSE_BATTLE_201, TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S
- **Monotonic trajectory:** ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES

## Independence Recommendation

For calibration requiring independent scenarios, exclude one from each overlap pair:

- **Recommended exclusion set A** (conservative, 22 scenarios):
  - Drop `CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI` (keep original crisis)
  - Drop `TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S` (keep GDPR, broader scope)

- **Recommended exclusion set B** (aggressive, 20 scenarios):
  - Same as A, plus drop `ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES` (monotonic)
  - Plus drop `CORP-2017-UBER_LONDON_LICENSE_BATTLE_201` (flat undecided)
