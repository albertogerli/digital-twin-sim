# Grounding PoC Report

## 1. Setup

- **LLM**: Google Gemini (gemini-3.1-flash-lite-preview), temperature=0.2
- **Search**: DuckDuckGo HTML scraping (real web search)
- **Posteriors**: v2 domain-specific means (NOT recalibrated)
- **Scenarios**: 3 (Dieselgate, SVB, Brexit)

## 2. Comparison Table

| Scenario | GT% | Orig% | Ground% | Err_O | Err_G | Δ (pp) | Improvement |
|---|---|---|---|---|---|---|---|
| DIESELGATE PUBLIC TRUST IN VW | 32.0 | 61.5 | 55.5 | 29.5 | 23.5 | +6.0 | +20% |
| SVB COLLAPSE MARCH 2023 BANKI | 38.0 | 76.9 | 33.6 | 38.9 | 4.4 | +34.5 | +89% |
| BREXIT | 51.9 | 54.3 | 50.6 | 2.4 | 1.3 | +1.1 | +46% |

### Implicit discrepancy (b_s)

| Scenario | b_s original | b_s grounded | |Δb_s| |
|---|---|---|---|
| DIESELGATE PUBLIC TRUST IN VW | -1.222 | -0.973 | +0.249 |
| SVB COLLAPSE MARCH 2023 BANKI | -1.692 | +0.191 | +1.501 |
| BREXIT | -0.097 | +0.052 | +0.045 |

## 3. Verdict

**Average Δ error: +13.9 pp**

**STRONG SIGNAL — full recalibration justified**

## 4. Qualitative Notes

### Dieselgate
- Original elite: 2 agents (Winterkorn + generic 'Financial Analysts')
- Grounded elite: 4 agents (scenario-specific stakeholders)
- Grounded events: 5 (from 68/100 FactSheet)

## 5. Recommendation

Based on these results:

- **Recalibrate**: Run full SVI with grounded inputs on all 42 scenarios
- **Priority**: Financial and corporate domains (highest b_s reduction)
- **Expected outcome**: Lower σ_b,within as simulator needs less discrepancy correction

---
*Generated: 2026-04-01 02:01*