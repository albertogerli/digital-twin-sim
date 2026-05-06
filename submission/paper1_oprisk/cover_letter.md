# Cover Letter — Submission to Journal of Operational Risk

**Date:** 6 May 2026

**To:** Editor-in-Chief, *Journal of Operational Risk*

Dear Editor,

I am writing to submit the manuscript "**A Power-Law Cost Model for Operational-Risk Incident Reports: Calibration on 40 Historical Incidents and Implications for DORA Compliance**" for consideration by *Journal of Operational Risk*.

The manuscript addresses a quantitative gap in the operational-risk literature that has become acute since the Digital Operational Resilience Act (Regulation EU 2022/2554, "DORA") entered into application in January 2025. Article 19 requires regulated financial entities to report a euro figure or band for the economic impact of major ICT-related incidents within four hours of classification, even when actuals are not yet known and the institution must rely on a prospective estimate. Existing implementations rely on either hand-selected historical analogues — providing no formal uncertainty band — or qualitative tier mappings that lose precision at tier boundaries. To my knowledge, no calibrated quantitative cost model with a held-out validation has been published for this use case.

The submission makes six contributions:

1. A **curated open reference dataset** of N=40 historical operational-risk incidents (1998–2024) across seven categories (Italian banking, EU banking, US banking, sovereign, cyber, telco, energy), with shock-magnitude estimates, public-domain euro costs, ISO incident dates, and at least two source citations per row. The dataset is released under CC-BY-4.0 as the basis for an open benchmark.

2. **Empirical evidence that the cost-vs-shock relationship is super-linear** with $\hat{\gamma} \approx 3.36$ on the overall corpus ($R^2_{\log} = 0.72$, $N = 40$) and $\hat{\gamma}$ ranging from 1.65 (energy) to 3.92 (banking-US) across categories. The hypothesis $H_0: \gamma = 1$ (linear) is rejected at $p < 0.01$ on every category with $n \geq 4$.

3. A **per-category power-law cost model** $\hat{c}(s) = \beta_k \cdot s^{\gamma_k}$ fitted by log-log Huber regression, with a model selection rule promoting the power-law from diagnostic to primary headline when per-category $R^2_{\log} \geq 0.5$.

4. A **leave-one-out cross-validation** under three competing model specifications (overall linear, per-category linear, per-category power-law). Hit-rate within $\pm 100\%$ on the proposed power-law specification: **80%**, against **35%** for the linear pooled baseline and **40%** for the per-category linear baseline. Median absolute percent error drops from 394% to 57%.

5. A **six-layer diagnostic stack** alongside the headline estimate: pairs-bootstrap quantile band, Eicker–Huber–White–MacKinnon HC3 sandwich SE, 2-state Gaussian HMM regime mixture on monthly log(VIX) 1997–2025, Hill estimator for the Pareto tail index, 2SLS with the regime posterior as instrument for shock_units, and a log-log slope reported as a Taleb-style fragility exponent. Each diagnostic targets a falsifiable assumption underlying the headline estimate.

6. An **open-benchmark release** (dataset, code, reproducible test suite of 35 unit tests, refit script) at `https://github.com/albertogerli/digital-twin-sim` so future cost estimators can be evaluated on the same corpus under the same protocol.

The manuscript is organised as a single-claim paper: the reference dataset, the power-law model, the validation, and the open benchmark. Total length: 23 pages including references and appendix.

I confirm that:

- The work is original and has not been previously published.
- The work is not under consideration at any other venue.
- All authors (single author) have agreed to this submission.
- There are no conflicts of interest to declare.
- Funding sources are disclosed in the manuscript: Tourbillon Tech Srl provided computational resources; no external grant funding was used.

I would like to suggest the following candidate reviewers, none of whom have collaborated with the author in the last five years:

- **Marco Bardoscia** (Bank of England, Financial Stability) — operational-risk networks
- **Carsten Detken** (European Central Bank, DG Macroprudential Policy) — systemic risk modelling
- **Christoph Hess** (independent operational-risk researcher) — empirical operational losses
- **Silvia Magri** (Banca d'Italia, Risk Management) — DORA implementation in Italian banks

I would respectfully decline, on grounds of recent collaboration or institutional overlap, any reviewer affiliated with Università degli Studi di Milano (joint affiliation) or with Tourbillon Tech Srl (sole funding source).

I look forward to the editor's and reviewers' feedback.

Sincerely,

**Alberto Giovanni Gerli**
Tourbillon Tech Srl, Padova, Italy
Dipartimento di Scienze Cliniche e di Comunità, Università degli Studi di Milano, Italy
alberto@albertogerli.it
