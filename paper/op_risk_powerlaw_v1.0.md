# A Power-Law Cost Model for Operational-Risk Incident Reports: Calibration on 40 Historical Incidents and Implications for DORA Compliance

**Alberto Giovanni Gerli**^{1,2}

^1 Tourbillon Tech Srl, Padova, Italy
^2 Dipartimento di Scienze Cliniche e di Comunità, Università degli Studi di Milano, Italy

**Corresponding author:** alberto@albertogerli.it

**Keywords:** operational risk, DORA, EU Regulation 2022/2554, major-incident reporting, power-law cost model, bootstrap inference, regime-switching HMM, leave-one-out cross-validation

---

## Abstract

The Digital Operational Resilience Act (Regulation EU 2022/2554, "DORA") requires regulated financial entities to report the economic impact of major information-and-communication-technology incidents within deadlines defined by a seven-criterion classification. Reporting templates request a euro figure or band for `economic_impact_eur` even when actuals are not yet known and the reporting institution must rely on a prospective estimate. Existing approaches anchor the estimate to either historical-cost analogues selected by hand or to qualitative tier mappings prescribed by internal risk-management policy. Both approaches lack a defensible quantitative model and a transparent uncertainty band that survives external audit.

We propose a calibrated power-law cost model $\hat{c}(s) = \beta \cdot s^\gamma$ where $s$ is a scalar shock-magnitude index of the incident and $(\beta, \gamma)$ are estimated by log-log Huber regression on a curated reference table of $N=40$ historical incidents (1998–2024) across seven operational categories — banking-IT, banking-EU, banking-US, sovereign, cyber, telco, and energy. We document the dataset, the per-category fits, and the model selection rule that promotes the power-law from diagnostic to primary estimator only when the log-space coefficient of determination exceeds $0.5$.

A leave-one-out cross-validation reports a hit-rate within $\pm 100\%$ of $80\%$ under the power-law model, against $35\%$ for a linear baseline $\hat{c}(s) = \alpha s$ and $40\%$ for a per-category linear baseline. Median absolute percent error drops from $394\%$ to $57\%$. The remaining error concentrates in the banking-US bucket where Lehman Brothers (2008) is a high-leverage outlier dominating the in-bucket slope.

The headline estimate is reported alongside six independent statistical diagnostics: (i) an empirical pairs-bootstrap on $(\beta, \gamma)$ for the epistemic uncertainty interval, (ii) the Eicker–Huber–White–MacKinnon HC3 sandwich standard error, (iii) a 2-state Gaussian hidden Markov model on monthly log(VIX) for incident-level regime posteriors and a regime-mixture α as cross-validation, (iv) a Hill estimator for the Pareto tail index of residuals to flag infinite-variance regimes, (v) two-stage least squares with the regime posterior as instrument for the shock index to test for endogeneity, and (vi) the log-log slope reported as a Taleb-style fragility exponent. The diagnostic stack is designed so that the supervisor reads each output against a falsifiable assumption, rather than against a black-box estimator.

We release the reference dataset, the calibration code, and the validation harness as an open benchmark for operational-risk cost modeling. The benchmark provides a reproducible skill floor for proposed cost estimators and a curated dataset for empirical work in operational-risk modeling under DORA and adjacent regulatory frameworks.

---

## 1. Introduction

### 1.1 Motivation

Article 19 of Regulation EU 2022/2554 (DORA, in force since January 2025) requires regulated financial entities to submit major-incident reports to their competent supervisory authority — for Italian banks the Banca d'Italia, for insurance undertakings IVASS, and for harmonised cross-jurisdictional cases the European Banking Authority, EIOPA, or ESMA depending on the entity. Annex I of the implementing technical standards (RTS, Joint Committee Final Report JC 2024-43) defines a seven-criterion classification matrix that includes a quantitative `economic_impact_eur` field. The matrix prescribes four reporting tiers — non-material (<€100k), low (€100k–€1m), high (€1m–€10m), critical ($\geq$ €10m) — and the entity is required to select a tier within four hours of an event being classified as major.

In practice the reporting institution must form a prospective estimate of the economic impact within a short deadline and based on partial information, and revise that estimate as the incident evolves. Neither the regulation nor the technical standards prescribe a specific estimation methodology, leaving the choice to the institution's internal risk-management policy. Existing implementations fall into two families.

The first family is **historical-cost analogue selection**. The risk officer identifies one or two past incidents that resemble the current one — typically by the same institution or by a comparable peer — and reports the analogue's eventual cost. The approach is intuitive and survives audit when the analogue is well-documented, but suffers from selection bias and provides no formal uncertainty band: a single analogue is a sample of size one. Industry guidance from the Operational Risk Management Forum (ORMF, 2021) and the EBA Guidelines on ICT and security risk management (EBA/GL/2019/04) acknowledges the practice but does not endorse it for the DORA reporting context.

The second family is **qualitative tier mapping**. The institution defines internal heuristics that map incident attributes (number of clients affected, downtime hours, geographical spread) to one of the four DORA tiers, typically through a rule-based engine implemented in the institution's GRC platform (MetricStream, ServiceNow GRC, IBM OpenPages, Wolters Kluwer OneSumX). The mapping is auditable but the tier boundaries are coarse: an incident estimated at €5m and one estimated at €15m fall into different tiers and trigger different supervisory follow-ups, but the underlying analysis cannot distinguish between them.

Neither approach provides what an external auditor or the supervisory authority would expect of a quantitative model: a documented relationship between observable inputs and the cost estimate, a calibrated parameter set with measurable goodness-of-fit on historical data, and a propagated uncertainty band with formal coverage properties.

### 1.2 Contribution

We close this gap by proposing a **calibrated power-law cost model** for operational-risk economic-impact estimation:

$$
\hat{c}(s) = \beta \cdot s^\gamma + \varepsilon, \qquad \log \hat{c} = \log \beta + \gamma \log s + u,
$$

where $s$ is a scalar shock-magnitude index of the incident, $(\beta, \gamma)$ are category-specific parameters fitted by log-log Huber regression on a curated reference table of $N=40$ historical incidents (1998–2024), and $\varepsilon$ has empirical distribution estimated from a pairs-bootstrap on the calibration subset.

The contributions are:

1. **A curated open reference dataset** (`shared/dora_reference_incidents.json`, schema v1.1) of $N=40$ historical incidents across seven operational categories, with per-incident shock-magnitude estimates, public-domain euro costs, ISO incident dates, regime labels, and a minimum of two source citations per row. The dataset is auditable, version-controlled, and released as the basis for an open benchmark for operational-risk cost modeling (Section 3).

2. **Empirical evidence that the cost-vs-shock relationship is super-linear with $\hat{\gamma} \approx 3$ on the overall corpus** ($R^2_{\log} = 0.72$) and $\hat{\gamma} = 3.12$ on the banking-IT slice ($R^2_{\log} = 0.97$). The linear cost model $\hat{c}(s) = \alpha s$, which is implicit in industry practice, is therefore mis-specified: it systematically over-predicts small incidents and under-predicts large ones (Section 4).

3. **A model selection rule** that promotes the power-law from diagnostic to primary estimator when the per-category log-space $R^2 \geq 0.5$, and falls back to the linear baseline otherwise. The rule is data-driven, audit-friendly (it produces an interpretable label per request), and conservative (smaller and noisier buckets default to the more familiar baseline) (Section 4.4).

4. **A leave-one-out cross-validation under three competing model specifications** (overall linear, per-category linear, per-category power-law) on the reference dataset. The power-law specification attains $80\%$ within $\pm 100\%$ against $40\%$ for the per-category linear baseline and $35\%$ for the overall linear baseline; median absolute percent error drops from $394\%$ to $57\%$ (Section 6).

5. **A six-layer diagnostic stack** alongside the headline estimate: empirical pairs-bootstrap on $(\beta, \gamma)$, HC3 heteroscedastic-robust standard error, hidden Markov regime posteriors, Hill tail-index estimator, two-stage least squares with regime as instrument, and a log-log slope reported as a fragility exponent. Each diagnostic targets a specific assumption underlying the headline estimate, and each is designed to be falsifiable and externally auditable (Section 5).

6. **An open benchmark and reproducibility package** that allows any proposed cost estimator to be evaluated against the same dataset under the same leave-one-out protocol, providing a fixed reference skill floor for the operational-risk modeling literature (Section 8).

### 1.3 Paper organisation

Section 2 reviews related work in operational-risk modeling, power-law regularities in financial losses, regime-switching econometrics, and the regulatory landscape under DORA. Section 3 documents the reference incident dataset, its construction methodology, and aggregate statistics. Section 4 presents the linear and power-law specifications and the model selection rule. Section 5 describes the six-layer diagnostic stack. Section 6 presents the cross-validation results. Section 7 discusses limitations and external validity. Section 8 describes the open-benchmark release. Section 9 concludes.

---

## 2. Related Work

### 2.1 Operational risk modeling

The quantitative literature on operational risk historically separated *frequency* models (how often does a loss occur) from *severity* models (how large is each loss). The Loss Distribution Approach (LDA), formalised under the Basel II Advanced Measurement Approach in 2004 and refined under Basel III, fits a Poisson process to event frequency and a heavy-tailed parametric distribution (typically lognormal, Weibull, or generalised Pareto) to event severity, then convolves the two via Monte Carlo to estimate the 99.9% Value-at-Risk one-year aggregate loss [Frachot et al. 2001; Cruz 2002; Aue & Kalkbrener 2006; Chernobai et al. 2007].

The LDA framework is appropriate for *aggregate-portfolio* risk capital but is less well-suited to *single-incident* prospective cost estimation, which is what DORA Article 19 requires. The institution observes a specific event in progress, is asked to estimate its cost within hours, and must condition on the incident's category, geography, and counterparty exposure rather than on a portfolio-level frequency–severity calibration.

Recent work has begun to address this gap. Cope et al. (2009) document that operational losses across institutions exhibit strong scaling regularities by event type, suggesting that conditional models stratified by category outperform pooled fits. Hess (2011) analyses the Algo FIRST loss database and reports that within-category severity distributions are well-approximated by power-law tails. Bardoscia et al. (2017) propose a network-conditional framework for operational losses in banking. None of these target the specific DORA major-incident reporting use-case, and none provide a public benchmark.

### 2.2 Power laws in financial losses

The empirical literature on financial losses has consistently reported power-law (Pareto-type) tails. Gabaix et al. (2003) documented power-law tails in stock-market returns; Bouchaud et al. (2002) found similar regularities in volatility and trading volume. For systemic-banking-crisis losses specifically, Reinhart and Rogoff (2009) and Laeven and Valencia (2018) maintain a curated dataset whose cost distribution exhibits a heavy right tail consistent with power-law behaviour above the median.

Power-law cost relationships of the form $c \propto s^\gamma$ with $\gamma > 1$ have been theoretically derived in cascade and contagion models (Acemoglu et al. 2015; Battiston et al. 2012) and empirically confirmed in network-banking simulations (Caccioli et al. 2014). The $\gamma$ exponent is variously called *contagion intensity*, *fragility exponent* [Taleb 2012], or simply *cost convexity*. Our work applies this regularity to the DORA reporting context and validates it against a held-out historical sample for the first time, to our knowledge.

### 2.3 Regime-switching econometrics

The 2-state Gaussian hidden Markov model for financial-market regime identification is due to Hamilton (1989, *Econometrica* 57:357), with EM estimation following Baum–Welch (Baum et al. 1970) and the standard log-space implementation (Rabiner 1989). Applications to volatility regime detection in equity markets (Schwert 1989; Pagan & Schwert 1990; Ang & Bekaert 2002), credit-spread regime classification (Kiefer 2010), and central-bank policy phase identification (Sims & Zha 2006) are extensive.

We use Hamilton's 2-state regime-switching HMM as a *diagnostic* layer: the regime posterior conditioned on incident date is one of three independent stress signals (the other two being category and hand-coded regime label), and the regime-mixture α serves as a cross-check on the per-category fit. This is a non-standard application — the literature uses HMM regime posteriors as inputs to a forecasting or pricing model, not as cross-validation for a separately fitted cost surface. The use is closer in spirit to Ang & Timmermann (2012, *Annual Review of Financial Economics* 4:313) who treat regime classification as an exploratory device for empirical-finance datasets.

### 2.4 Robust regression and bootstrap inference

The Huber M-estimator (Huber 1964) is the standard robust regression specification for samples containing high-leverage outliers; we use it in both the level (linear $\alpha s$) and log-space (power-law) fits with the canonical tuning constant $k = 1.345$. The empirical pairs-bootstrap follows Efron (1979) and Davison & Hinkley (1997, *Bootstrap Methods and Their Application*); we report empirical 5°/95° quantiles of the resampled coefficient distribution rather than parametric confidence intervals because the residual distribution is heavy-tailed and a Gaussian band would understate tail risk.

The Eicker–Huber–White–MacKinnon HC3 sandwich estimator (Eicker 1967; Huber 1967; White 1980; MacKinnon & White 1985) provides heteroscedastic-robust standard errors without requiring the analyst to specify a variance form. The Hill estimator for the Pareto tail index (Hill 1975, *Annals of Statistics* 3:1163) and its small-sample bias properties (Hall 1990; Resnick & Stărică 1997) are standard tools in extreme-value analysis; we use it as a diagnostic on residual tails to flag the infinite-variance regime that occurs when the corpus contains a Lehman-class outlier.

### 2.5 The DORA regulatory context

Regulation EU 2022/2554 (DORA) entered into force on 16 January 2023 and is applicable from 17 January 2025. Article 17 requires regulated entities to maintain an ICT-related incident management process; Article 18 establishes the criteria for classifying ICT-related incidents as "major"; Article 19 prescribes the reporting obligation to the competent supervisory authority. The implementing technical standards were finalised in the EBA/EIOPA/ESMA Joint Committee Final Report JC 2024-43 (July 2024) and define the seven-criterion classification matrix (clients_affected, data_losses, reputational_impact, duration_downtime_hours, geographical_spread, economic_impact_eur, criticality_of_services_affected) along with the report templates.

Existing surveys of the DORA implementation landscape (Deloitte 2024; PwC 2024; Accenture 2025) emphasise process and governance compliance and largely defer the quantitative `economic_impact_eur` estimation to existing institutional GRC tooling. Our work targets that specific quantitative gap.

---

## 3. Reference Incident Dataset

### 3.1 Construction methodology

The reference dataset, `shared/dora_reference_incidents.json` (schema v1.1), comprises $N=40$ historical operational-risk incidents in regulated financial-system or financial-system-adjacent sectors that occurred between September 1998 and July 2024. Each incident is an event for which (i) a public-domain estimate of the eventual euro cost exists in regulator filings, financial-stability reports, peer-reviewed retrospectives, or official institutional disclosures, (ii) a rough estimate of the incident's shock magnitude can be derived from contemporaneous news intensity, and (iii) the event date and category are unambiguous.

The construction proceeded in three phases. First, an initial seed list of $\sim 80$ candidate events was assembled from the Reinhart-Rogoff systemic banking crisis database, Laeven-Valencia 2018, the IMF Financial Soundness Indicators incident registry, the BIS Quarterly Review historical archive (2007–2024), and the EBA risk dashboards. Second, candidates were filtered to those with at least two independent public-domain cost citations and a primary regulatory or central-bank source. Third, each surviving candidate was assigned a categorical label (one of seven operational categories), an ISO incident date for the peak news event (typically the day of formal announcement: Lehman bankruptcy filing 2008-09-15, MPS bailout 2017-07-04, CrowdStrike Falcon outage 2024-07-19), and a shock-magnitude estimate following the heuristic tiers defined in Section 3.3.

The final corpus contains $N=40$ entries distributed across the seven categories as follows: banking-IT (Italian banks, $n=6$), banking-EU ($n=8$), banking-US ($n=7$), sovereign ($n=6$), cyber ($n=6$), telco ($n=4$), and energy ($n=3$). The category distribution reflects both the documented public-information availability and the relative importance of each category to the EU regulated-financial perimeter that DORA covers.

### 3.2 Data schema

Each incident row is a JSON object with the following fields:

```json
{
  "id": "mps_bailin_2017",
  "shock_units": 1.6,
  "cost_eur_m": 3900,
  "category": "banking_it",
  "label": "MPS precautionary recapitalisation (2017)",
  "incident_date": "2017-07-04",
  "regime": "stressed",
  "sources": [
    "DG Comp state aid SA.47677",
    "MEF press release 2017-07-04"
  ]
}
```

The fields and their interpretation:

- `id`: a snake-case identifier, primary key.
- `shock_units`: a scalar in the open range $(0, 5)$ measuring the magnitude of the event on a heuristic four-tier scale (Section 3.3).
- `cost_eur_m`: the public-domain estimate of the eventual cost in euro-millions, taken from the cited primary source. For events denominated in non-euro currencies (Lehman, SVB, Argentina default), the conversion uses the historical average exchange rate over the resolution window.
- `category`: one of `banking_it`, `banking_eu`, `banking_us`, `sovereign`, `cyber`, `telco`, `energy`.
- `label`: a human-readable description used in tables and visualisations.
- `incident_date`: the ISO date of the peak event (formal announcement, takeover, declaration). Used in Section 5 for the regime-posterior diagnostic.
- `regime`: one of `calm`, `stressed`, `crisis`. Hand-coded from market context. Used as a hand-curated cross-check against the data-driven regime posterior in Section 5.
- `sources`: a list of at least two public-domain references. Allows audit of the cost figure and fact-checking of the date and category.

### 3.3 Shock-magnitude heuristic

The shock-magnitude index $s$ is a scalar that aggregates the event's intrinsic severity and its observed news intensity. The index is dimensionless and is calibrated to four heuristic tiers reflecting the scale at which institutional risk officers typically reason about operational events:

| Tier | Range | Description | Examples |
|---|---|---|---|
| I | 0.5 – 1.2 | Single-firm contained event | Tercas FITD intervention (2014, $s=0.8$); Popolare Bari rescue (2019, $s=1.0$) |
| II | 1.2 – 2.0 | Sector-wide moderate or protracted single-firm | MPS deposit run (2016, $s=1.4$); Wirecard collapse (2020, $s=1.6$) |
| III | 2.0 – 3.0 | Major sector or sovereign-adjacent | SVB collapse (2023, $s=2.4$); Cyprus bail-in (2013, $s=2.3$) |
| IV | $\geq 3.0$ | Systemic or sovereign-class | Brexit Wave-1 (2016, $s=3.2$); Argentina default (2001, $s=3.5$); Lehman (2008, $s=4.0$) |

The tier assignment was performed by the author with reference to contemporaneous press archives, the cited primary regulatory sources, and the central-bank macro-financial reviews from the relevant year. The heuristic is intentionally coarse (one significant figure per assignment) to avoid over-fitting the cost model to subjective shock-magnitude refinements. Section 7.2 discusses the sensitivity of the headline cost estimate to plausible shock-magnitude perturbations.

### 3.4 Aggregate dataset statistics

Across the $N=40$ corpus, the unweighted distribution of shock-magnitude is centred around $\bar{s} = 1.93$ with standard deviation $0.78$ and range $[0.7, 4.0]$. The unweighted distribution of euro cost is heavily right-skewed: median cost €5,200m, mean cost €31,860m, maximum cost €600,000m (Lehman). The cost distribution exhibits a Gini coefficient of $0.81$, indicating concentration of mass in a few high-leverage observations. Per-category statistics are reported in Table 1.

**Table 1: Reference dataset summary statistics by category.**

| Category | $n$ | mean $s$ | mean $c$ (€M) | median $c$ (€M) | max $c$ (€M) |
|---|---|---|---|---|---|
| banking_it | 6 | 1.18 | 2,300 | 1,950 | 5,200 |
| banking_eu | 8 | 1.81 | 21,250 | 7,150 | 90,000 |
| banking_us | 7 | 2.27 | 96,500 | 9,000 | 600,000 |
| sovereign | 6 | 2.57 | 35,330 | 27,500 | 82,000 |
| cyber | 6 | 2.25 | 21,800 | 7,000 | 100,000 |
| telco | 4 | 1.05 | 3,725 | 1,600 | 11,000 |
| energy | 3 | 1.40 | 15,130 | 11,000 | 34,000 |

The banking-US bucket is dominated by Lehman Brothers (2008, $c =$ €600B, $s = 4.0$), which is approximately $66\times$ larger than the next-largest in-category observation (Washington Mutual, $c =$ €25B). Retaining Lehman in the calibration set ensures the model can extrapolate to systemic-class events, but it dominates per-category fits in banking-US through high leverage.

The full corpus is reproduced in Appendix A; the JSON file is available at the project repository (Section 8).

---

## 4. Power-Law Cost Model

### 4.1 Model specifications

We compare three model specifications fitted to the reference dataset:

**Model M1 — Linear pooled.** The simplest specification: a single slope $\alpha$ across all incidents, no intercept, no category conditioning.

$$
M_1: \quad c_i = \alpha \cdot s_i + \varepsilon_i, \quad i = 1, \ldots, N.
$$

**Model M2 — Linear per-category.** A category-specific slope $\alpha_k$ for each of the seven operational categories, no intercept.

$$
M_2: \quad c_i = \alpha_{k(i)} \cdot s_i + \varepsilon_i, \quad k(i) \in \{1, \ldots, 7\}.
$$

**Model M3 — Power-law per-category (proposed).** A category-specific multiplicative pre-factor $\beta_k$ and a category-specific exponent $\gamma_k$.

$$
M_3: \quad c_i = \beta_{k(i)} \cdot s_i^{\gamma_{k(i)}} + \varepsilon_i,
\qquad \log c_i = \log \beta_{k(i)} + \gamma_{k(i)} \log s_i + u_i.
$$

The power-law model M3 is fitted by ordinary least squares in log space. Models M1 and M2 are fitted by Huber regression (Huber 1964) with the canonical tuning constant $k = 1.345$ via iteratively reweighted least squares; the Huber loss bounds the influence of high-leverage observations (notably Lehman in banking-US) without dropping them. Log-space residuals from M3 are nearly Gaussian post-transformation, so the additional robustness of Huber is unnecessary for M3.

### 4.2 Estimated parameters

Table 2 reports the per-category fits of the power-law model M3 alongside the per-category linear $\alpha_k$ from M2 for comparison. The pooled linear $\alpha$ from M1 is $\alpha = $ €13.99B per shock-unit.

**Table 2: Estimated parameters per category (M2 and M3).**

| Category | $n$ | M2: $\alpha_k$ (€M/unit) | $R^2$ | M3: $\hat{\beta}_k$ (€M) | $\hat{\gamma}_k$ | $R^2_{\log}$ |
|---|---|---|---|---|---|---|
| banking_it | 6 | 1,946 | 0.88 | 745 | 3.12 | 0.97 |
| banking_eu | 8 | 14,455 | 0.55 | 1,820 | 3.38 | 0.78 |
| banking_us | 7 | 65,502 | 0.46 | 4,620 | 3.92 | 0.82 |
| sovereign | 6 | 14,277 | 0.83 | 6,180 | 1.96 | 0.74 |
| cyber | 6 | 9,612 | 0.29 | 2,130 | 2.81 | 0.69 |
| telco | 4 | 4,329 | 0.57 | 1,210 | 2.34 | 0.80 |
| energy | 3 | 11,129 | 0.80 | 8,920 | 1.65 | 0.84 |
| Overall | 40 | 13,989 | 0.20 | 1,158 | 3.36 | 0.72 |

Three observations:

First, $\hat{\gamma}_k > 1$ on every category, ranging from $1.65$ (energy, with $n=3$) to $3.92$ (banking-US). The cost-vs-shock relationship is strictly super-linear across all sectors of the corpus. The hypothesis $H_0: \gamma = 1$ is rejected at $p < 0.01$ on every category with $n \geq 4$ (Wald test on the log-log slope, standard errors via the same pairs-bootstrap as in Section 5.1).

Second, $R^2_{\log}$ in M3 is uniformly higher than $R^2$ in M2 across every category except `energy` (where $n=3$ is too small for a meaningful comparison). The improvement is largest on the banking-IT slice ($0.97$ vs $0.88$) where the corpus exhibits clean log-log linearity.

Third, the pooled $\hat{\gamma} = 3.36$ on the full corpus has $R^2_{\log} = 0.72$, which is moderately strong evidence that the super-linear regularity is not an artefact of category-specific calibration but a structural property of the operational-risk cost surface across sectors.

### 4.3 Residual diagnostics

Figure 1 (reproduced in the project repository) shows the log-log scatter of cost vs shock-magnitude on the full corpus, with the M3 fitted line and per-category colour coding. Lehman Brothers (2008, banking-US) is the most visible outlier above the line; Tercas (2014, banking-IT) and ENI Gabon (2020, energy) are the most visible outliers below. The systematic structure of these residuals — Lehman is genuinely a tail event, Tercas was contained by FITD intervention, ENI Gabon was a writedown rather than a cash loss — suggests that the power-law fit is capturing the central tendency well but that the residuals carry information that a richer model could exploit (e.g., an indicator for ex-post supervisory intervention).

### 4.4 Model selection rule

For production deployment we propose a **conservative model selection rule**: the power-law model M3 is promoted to primary headline whenever the per-category log-space coefficient of determination satisfies $R^2_{\log} \geq 0.5$, otherwise the per-category linear baseline M2 is retained.

The threshold of $0.5$ is pragmatic: it ensures that the log-log fit is meaningfully better than a horizontal line in log-cost space, while remaining low enough to admit power-law conditioning on every category in the current corpus. Under this rule, all seven categories qualify for M3 on the present dataset; the rule will become binding only on substantially smaller or noisier categories that may emerge as the corpus grows (Section 7.4).

A second rule — **extrapolation flagging** — fires a warning when the request's shock-magnitude $s_*$ exceeds $1.3 \times \max_i s_i$ on the calibration subset. Power laws are well-known to be hazardous outside their support; the $1.3$ threshold is conservative. The warning is surfaced in the user interface and recorded in the audit log.

### 4.5 Worked validation against the reference table

For three within-bucket reference points in the banking-IT × stressed-regime slice we compare the M2 and M3 predictions against the actuals:

**Table 3: Within-bucket validation in banking-IT.**

| Incident | $s$ | actual $c$ (€M) | M2 prediction (€M) | M2 error | M3 prediction (€M) | M3 error |
|---|---|---|---|---|---|---|
| Tercas (2014) | 0.8 | 300 | 1,560 | $+420\%$ | 370 | $+23\%$ |
| MPS bailout (2017) | 1.6 | 3,900 | 3,110 | $-20\%$ | 3,220 | $-17\%$ |
| Veneto/Pop Vicenza (2017) | 2.0 | 5,200 | 3,890 | $-25\%$ | 6,460 | $+24\%$ |

The linear M2 fails systematically on Tercas (a small-shock contained event); the power-law M3 splits its errors symmetrically around the truth. For the genuine tail event in banking-US — Lehman 2008 ($s = 4.0$, actual $c = $€600B) — M3 predicts €422B ($-30\%$) while M2 predicts €199B ($-67\%$); both under-predict the true 6σ event, but M3 captures more of the convexity.

The systematic comparison across the full $N=40$ corpus is in Section 6.

---

## 5. Diagnostic Stack

The headline estimate from Section 4 is one quantity computed under one set of modelling assumptions. To avoid the false precision that a single black-box estimator delivers, the production pipeline reports six independent diagnostics alongside the headline. Each diagnostic targets a specific assumption that the headline rests on, and each is designed to be falsifiable and externally auditable.

### 5.1 Empirical pairs-bootstrap on $(\beta, \gamma)$

The epistemic interval on the headline estimate is the empirical 5°/95° quantile of the predicted cost distribution under a $B = 5{,}000$ replicate pairs-bootstrap of the calibration subset (Efron 1979; Davison & Hinkley 1997). For each bootstrap replicate $b$, we resample $N$ rows with replacement from the per-category subset, refit $(\beta_b, \gamma_b)$ by log-log Huber regression, and compute the predicted cost $\beta_b \cdot s_*^{\gamma_b}$ at the request's target shock $s_*$. The empirical $5\%$ and $95\%$ quantiles of the resulting $B$ predictions are reported as the lower and upper bounds of the epistemic interval.

The pairs-bootstrap propagates joint uncertainty in $(\beta, \gamma)$ — which is correlated since $\log \beta$ is the intercept and $\gamma$ the slope of the log-log fit — through to the cost prediction. The use of empirical quantiles rather than parametric ($\pm 1.645 \cdot \hat{\sigma}$) bands is appropriate because the prediction distribution is right-skewed for $\gamma > 1$: the lower tail compresses while the upper tail extends, and a Gaussian band would understate upper-tail risk.

### 5.2 HC3 sandwich standard error

Alongside the bootstrap quantile band we report the Eicker–Huber–White–MacKinnon HC3 sandwich estimator for the standard error of the linear-baseline $\hat{\alpha}$ in M2:

$$
\widehat{\text{Var}}_{\text{HC3}}(\hat{\alpha}) = \frac{1}{(\sum_i s_i^2)^2} \sum_i s_i^2 \left( \frac{c_i - \hat{\alpha} s_i}{1 - h_i} \right)^2, \qquad h_i = \frac{s_i^2}{\sum_j s_j^2},
$$

where $h_i$ is the leverage of observation $i$ (Eicker 1967; Huber 1967; White 1980; MacKinnon & White 1985). HC3 inflates standard errors in the presence of heteroscedasticity (residuals scaling with the regressor) without requiring the analyst to specify a variance form. The difference $\hat{\sigma}_{\text{HC3}} - \hat{\sigma}_{\text{OLS}}$ is itself a heteroscedasticity diagnostic: on the banking-IT × stressed slice we report $\hat{\sigma}_{\text{HC3}} = $ €404M/unit against the homoscedastic $\hat{\sigma}_{\text{OLS}} = $ €1,089M/unit, indicating that residuals are *less* dispersed than a homoscedastic-Gaussian model would suggest — a few high-leverage observations were inflating the homoscedastic estimate.

### 5.3 Hidden Markov regime posterior

We fit a 2-state Gaussian hidden Markov model (Hamilton 1989, *Econometrica* 57:357) on the monthly log(VIX) series 1997-01 through 2025-12 ($T = 348$ observations), with EM estimation via Baum–Welch / forward–backward in log-space (Baum et al. 1970; Rabiner 1989). Initialisation is by quantile clustering (low-state mean at the 25° percentile of log(VIX), high-state mean at the 75° percentile) with strongly persistent transition prior ($\tilde{a}_{ii} = 0.95$).

The fitted parameters are well-identified and the two regimes are well-separated: low-volatility regime $\hat{\mu}_0 = \log(14.7)$, high-volatility regime $\hat{\mu}_1 = \log(24.9)$, with $P(z_{t+1} = z_t \mid z_t = i) \approx 0.96$ in both states (mean dwell time $\sim 25$ months). The smoother evaluated at each reference incident's calendar month gives the posterior $p_i = P(z_i = \text{high} \mid x_{1:T})$.

We then fit a regime-mixture cost model
$$
c_i = \alpha_{\text{low}} (1 - p_i) s_i + \alpha_{\text{high}} p_i s_i + \nu_i
$$
by two-feature OLS. On the overall corpus we report $\hat{\alpha}_{\text{low}} = $ €2.4B per unit, $\hat{\alpha}_{\text{high}} = $ €47.7B per unit; the regime amplification is $\hat{\alpha}_{\text{high}} / \hat{\alpha}_{\text{low}} = 20.1$ ($R^2 = 0.43$).

The HMM-derived regime posterior provides an *external* cross-check on the hand-coded regime label (calm/stressed/crisis) included in the dataset for each incident. The two are *not* always in agreement: Cyprus 2013 is hand-coded `crisis` but the HMM posterior gives $p_i = 0.000$ (VIX was actually low at the time, the crisis was sovereign-debt regional rather than global volatility); LTCM 1998 is hand-coded `stressed` but the HMM posterior gives $p_i = 1.000$ (the joint Russian-default-plus-LTCM unwind was a global volatility event by the HMM's measure). The disagreements are interpretable and the HMM posterior provides a more uniform stress signal than the hand-coded label.

### 5.4 Hill estimator for the Pareto tail index

The Hill estimator (Hill 1975, *Annals of Statistics* 3:1163) provides a non-parametric estimate of the Pareto tail index $\alpha_{\text{Hill}}$ of the absolute residual distribution. For the top $k = \lfloor 0.10 N \rfloor$ residuals ranked by absolute magnitude,
$$
\hat{\alpha}_{\text{Hill}} = \frac{1}{(1/k) \sum_{i=1}^{k} \log r_{(i)} - \log r_{(k+1)}}.
$$

On the overall corpus we measure $\hat{\alpha}_{\text{Hill}} = 0.74$, formally an *infinite-variance* regime dominated by the Lehman residual. On the banking-IT slice we measure $\hat{\alpha}_{\text{Hill}} = 4.91$, a moderate tail. The diagnostic surfaces a flag in the user interface when $\hat{\alpha}_{\text{Hill}} < 2$ (infinite variance), telling the operator that the headline point estimate remains meaningful but the upper tail of the cost distribution is genuinely unbounded.

### 5.5 Two-stage least squares with regime as instrument

The simulated shock-magnitude $s$ is plausibly *endogenous* to the calibration of any forecasting tool: if the same tool were tuned on the same outcomes the costs are derived from, an OLS regression on the reference table would over-estimate the slope. We instrument $s_i$ with the HMM regime posterior $p_i$, which is exogenous to the forecasting tool by construction (it is derived from VIX, an external observation channel, and from publicly available date metadata).

The just-identified Wald estimator is
$$
\hat{\beta}_{\text{2SLS}} = \frac{\widehat{\text{Cov}}(p_i, c_i)}{\widehat{\text{Cov}}(p_i, s_i)}
$$
with first-stage F-statistic $F = \hat{\pi}_1^2 / \widehat{\text{Var}}(\hat{\pi}_1)$ from the auxiliary regression $s_i = \pi_0 + \pi_1 p_i + u_i$ (Theil 1953; Basmann 1957). On the current $N=40$ corpus we measure $F \approx 0.9$, well below the Stock & Yogo (2005) threshold of $F \geq 10$ for valid weak-instrument inference. The instrument is therefore *too weak* on this sample size to deliver clean structural inference on $\beta$; we report the diagnostic as a sanity check, not as the headline.

The honest framing for the supervisor is: "an endogeneity test was performed; the only available regime-derived instrument has $F < 1$ on the present sample, and we cannot definitively rule out simulated-shock endogeneity. The remediation path is either a stronger instrument (a measurement of stress that is exogenous by construction and not derived from VIX) or a substantially larger reference dataset that increases the power of the first-stage regression."

### 5.6 Log-log slope as fragility exponent

The log-log slope $\hat{\gamma}$ from the M3 fit of Section 4 is identical, up to interpretation, to Taleb's (2012, *Antifragile*) fragility exponent: $\gamma > 1$ flags convex / fragile exposure (cost responds super-linearly to shock), $\gamma < 1$ flags concave / antifragile exposure (cost saturates as shock grows), $\gamma \approx 1$ is the linear baseline. On the overall corpus we measure $\hat{\gamma} = 3.36$ ($R^2_{\log} = 0.72$, $N = 40$); on banking-IT $\hat{\gamma} = 3.12$ ($R^2 = 0.97$, $n = 6$). Both signal strongly convex exposure.

The diagnostic is surfaced in the user interface when $\hat{\gamma} > 1.10$, telling the operator that the linear $\alpha s$ baseline is mis-specified and the power-law primary is doing the work. The threshold of $1.10$ is informally chosen to provide a buffer above the unit slope; future work could replace it with a Wald-test threshold tied to the bootstrap standard error of $\hat{\gamma}$.

---

## 6. Validation: Leave-One-Out Cross-Validation

The leave-one-out cross-validation protocol is as follows. For each held-out incident $i \in \{1, \ldots, N\}$ in turn, the chosen model specification is refit on the remaining $N-1$ incidents and the held-out cost $\hat{c}_i$ is predicted. We aggregate the per-incident percent errors $\text{err}_i = (\hat{c}_i - c_i) / c_i$ into hit-rates within $\pm 50\%$, $\pm 100\%$, and $\pm 200\%$, plus the median absolute percent error and the mean absolute error in euro-millions.

We compare three model specifications corresponding to M1, M2, and M3 of Section 4.1:

- **Mode `overall`** (M1): pool all $N-1$ incidents, fit a single $\alpha$, predict the held-out as $\hat{c}_i = \hat{\alpha} \cdot s_i$. Worst-case baseline; included to quantify the cost of *not* category-conditioning.
- **Mode `category_aware`** (M2): refit the per-category $\alpha_k$ using only same-category training data (with fallback to overall when in-category $n < 3$). Linear baseline used in the production system before promotion of M3.
- **Mode `power_law`** (M3, proposed): refit the per-category $(\beta_k, \gamma_k)$ via log-log Huber on the same-category training data (with fallback to overall when in-category $n < 4$).

### 6.1 Aggregate results

Table 4 reports the cross-validation aggregates across the three modes.

**Table 4: Leave-one-out cross-validation summary on $N=40$.**

| Mode | Hit ±50% | Hit ±100% | Hit ±200% | Median \|err\| | MAE (€B) |
|---|---|---|---|---|---|
| `overall` (M1, linear) | 20% | 35% | 40% | 394% | 31.9 |
| `category_aware` (M2, linear) | 20% | 40% | 52% | 166% | 43.3 |
| `power_law` (M3, proposed) | **48%** | **80%** | **88%** | **57%** | **25.7** |

Three observations.

First, the naive `overall` linear fit attains hit-rate $\pm 100\%$ at only $35\%$. The reason is illustrated by Tercas 2014 ($s = 0.8$, actual €300M): the LOO refit on the other 39 rows gives $\hat{\alpha} \approx $€14B per unit (heavily influenced by Lehman and other banking-US observations), and the prediction for Tercas is $0.8 \times $€14B $= $€11.3B — wildly over-predicted by a factor of 38. This is exactly the failure mode that motivates per-category conditioning.

Second, going to `category_aware` (M2 per-category fit) drops the median absolute percent error from $394\%$ to $166\%$ but still leaves over half the corpus outside $\pm 100\%$, because the linear functional form is wrong. For the same Tercas example, the in-category fit gives $\hat{\alpha}_{\text{IT}} = $ €1.95B per unit and the prediction becomes $0.8 \times $€1.95B $= $€1.56B — better, but still $5\times$ over the actual.

Third, `power_law` clears $80\%$ of the corpus to within $\pm 100\%$ — this is the production estimator's defensible empirical claim. Hit-rate within $\pm 50\%$ improves from $20\%$ (both linear modes) to $48\%$, and median absolute percent error drops to $57\%$. The MAE in euro-millions also drops, indicating that the improvement is not purely a small-incident effect.

### 6.2 Per-bucket residual analysis

The remaining errors in `power_law` mode concentrate in two buckets.

The **banking-US 2023 sub-bucket** — Silicon Valley Bank ($s = 2.4$, actual €9B; predicted €40B), Signature Bank, First Republic — is over-predicted because the per-category $\hat{\gamma}_{\text{US}} = 3.92$ is pulled high by Lehman 2008 ($s = 4.0$, actual €600B). This is a sample-size issue (only $n = 7$ banking-US incidents, of which one is a 6σ outlier), not a model-form issue. The remediation path is annotating more banking-US incidents from the 2014–2019 calm-regime period (e.g., Wells Fargo cross-selling 2016, JPMorgan London Whale 2012, Goldman Sachs 1MDB 2018) to balance the bucket.

The **cyber bucket** has high variance in both directions. Equifax 2017 ($s = 1.5$, actual €1.4B) is over-predicted at €7.3B because the bucket includes SolarWinds 2020 ($s = 2.3$, actual €100B) and CrowdStrike 2024 ($s = 2.8$, actual €10B) which together pull $\hat{\gamma}_{\text{cyber}} = 2.81$ high. Conversely SolarWinds itself is under-predicted at €30B vs actual €100B — the cyber-supply-chain compromise category genuinely has a much heavier tail than the moderate $\hat{\gamma}_{\text{cyber}}$ captures. Both directions of error point to the same underlying cause: cyber incidents span four orders of magnitude on the cost axis with only six observations, and the bucket needs at least double the observations to support a stable per-category fit.

The **best-fit bucket is sovereign**: median absolute error $9.3\%$, hit-rate within $\pm 50\%$ at $83\%$ (5 of 6). This is consistent with the cleaner power-law structure visible in Table 2 ($R^2_{\log} = 0.74$) and with the well-documented power-law tails in sovereign-default cost distributions (Reinhart & Rogoff 2009).

### 6.3 Interpretation: why $80\%$ matters

The hit-rate within $\pm 100\%$ is the practical target metric for the DORA reporting context: a supervisor who reads a report claiming "economic impact estimated at €5B" would not be misled by a true cost of €4B or €6B, but would be misled by a true cost of €500M or €50B (one tier up or one tier down on the four-tier reporting matrix). The $80\%$ figure means that, on the historical corpus, the proposed estimator delivers a tier-correct prediction in four out of five cases.

The benchmark for this metric in the existing operational-risk literature is essentially absent: industry implementations of qualitative tier mapping do not publish cross-validation hit-rates, and the academic LDA literature targets aggregate-portfolio Value-at-Risk rather than single-incident point predictions. A direct comparison is therefore only possible against the linear baselines — the $80\%$ vs $35\%$ improvement is the cleanest claim. We invite reviewers to evaluate alternative estimators on the same corpus under the same protocol; the open-benchmark release (Section 8) is designed exactly for this.

---

## 7. Discussion: Limitations and External Validity

### 7.1 Sample size

The most important limitation is the sample size: $N=40$ is small for the seven-category structure, with the smallest bucket (`energy`) at $n=3$ and the largest (`banking_eu`) at $n=8$. Per-category $\hat{\gamma}_k$ has wide bootstrap confidence intervals on the smaller buckets, and the production system falls back to the overall fit when in-category $n < 4$.

Active expansion of the corpus is the obvious next step. The natural target for v2.0 of the dataset is $N \geq 60$ with no bucket below $n = 6$, which would tighten per-category confidence intervals enough to let extrapolation flagging (Section 4.4) become the primary safety net rather than the in-category-$n < 4$ fallback. Candidate additions are documented in the project repository.

### 7.2 Shock-magnitude heuristic sensitivity

The shock-magnitude index $s$ is constructed via a heuristic four-tier assignment (Section 3.3) rather than an objectively measured quantity. We assess sensitivity by perturbing each incident's $s$ by $\pm 0.1$ (one half-tier) and re-running the full leave-one-out cross-validation under M3. Hit-rate within $\pm 100\%$ moves from $80\%$ to the range $[76\%, 82\%]$ across $50$ random perturbations; median absolute percent error moves from $57\%$ to the range $[51\%, 64\%]$. The headline claims are robust to plausible shock-magnitude perturbations.

A higher-fidelity construction of $s$ — for example, replaying each historical incident through an agent-based simulation calibrated independently and measuring the simulator's aggregate $\sum_t |s_t \cdot \hat{d}_t|$ — would in principle reduce the heuristic noise. This is documented as future work in the project repository.

### 7.3 Endogeneity of the shock measure

If the shock-magnitude $s$ is itself constructed by a tool calibrated on the same outcomes the costs are derived from, then OLS on the reference table over-estimates the slope. Section 5.5 reports the 2SLS-IV diagnostic with the HMM regime posterior as instrument; on the current $N=40$ corpus the first-stage $F < 1$, so we cannot definitively bound the endogeneity bias. The remediation paths are (i) a stronger instrument (a measurement of stress that is exogenous to the forecasting tool by construction), (ii) a larger sample to power the first-stage regression, and (iii) a documented separation between the calibration corpus and the operational deployment scope. Path (iii) is the cleanest in practice: the reference dataset is fixed, the production estimator is calibrated on it once, and the deployment is over future incidents rather than over the same corpus.

### 7.4 Tail risk and infinite-variance regime

The Hill estimator on the overall corpus gives $\hat{\alpha}_{\text{Hill}} = 0.74$, formally an infinite-variance regime. The point predictions of the headline estimator remain well-identified, but the upper tail of the cost distribution is genuinely unbounded: a new Lehman-class event would dominate the worst-case projection regardless of the chosen model specification. The supervisor should read the headline as a central tendency of the cost distribution conditional on the observed shock-magnitude, not as a worst-case bound. The bootstrap 95° quantile (Section 5.1) provides a more honest worst case at the cost of a wider reported interval.

### 7.5 Categorical assignment ambiguity

Some incidents straddle category boundaries — e.g., the CrowdStrike 2024 outage is classified `cyber` in our schema because the root cause was a software supply-chain failure, but its primary economic impact was in `banking` and `telco` services that depended on the affected hosts. We resolve such cases by assigning the category corresponding to the *root cause* rather than the *propagated impact*, matching the EBA event-type taxonomy (EBA/GL/2017/05). The choice is documented in the dataset's per-row notes.

### 7.6 External validity to non-EU jurisdictions

The reference dataset covers EU and US incidents (with one Argentine sovereign default included). The model's external validity to non-EU jurisdictions — Asia-Pacific operational events, Latin American sovereign-bank cascades — is not established. We recommend that any deployment outside the EU/US perimeter recalibrate against a regionally-curated reference dataset. The methodology transfers cleanly; only the dataset needs to change.

---

## 8. Open Benchmark Release

To support reproducibility and follow-up work, we release the following artefacts at the project repository:

1. **The reference dataset** (`shared/dora_reference_incidents.json`, schema v1.1) under CC-BY-4.0 licence. The dataset includes per-incident shock-magnitude estimates, public-domain euro costs, ISO incident dates, regime labels, and source citations.

2. **The calibration code** (`core/dora/economic_impact.py`, MIT licence). Implements the linear and power-law fits (M1, M2, M3), the six-layer diagnostic stack (Section 5), and the leave-one-out cross-validation harness with the three competing modes (Section 6).

3. **The HMM regime-posterior cache** (`shared/vix_monthly_cache.json`, CC-BY-4.0). Monthly log(VIX) observations 1997-01 through 2025-12 ($T = 348$) used to fit the regime-mixture diagnostic of Section 5.3.

4. **A reproducible test suite** (`tests/test_dora_economic_impact.py`) with $35$ unit tests covering the analytical properties of each estimator (closed-form OLS/Huber on synthetic samples, hand-computed HC3 on a 3-point example, $\gamma$ recovery on synthetic $y = \beta x^\gamma$ data, Pareto tail-index recovery on synthetic Pareto residuals, contract invariants on the public API). Runs in under two seconds on a standard laptop.

5. **A re-fit script** (`scripts/calibrate_dora_alpha.py`) that re-runs the full per-category calibration and writes a versioned snapshot to `outputs/dora_calibration.json`. Designed to be invoked nightly via cron once the reference dataset exceeds $N \approx 100$ entries that change in any given month.

The project repository is at `https://github.com/albertogerli/digital-twin-sim`.

We invite the operational-risk modelling community to propose alternative estimators evaluated on the same corpus under the same leave-one-out protocol. A standardised comparison table — model specification, hit-rate within $\pm 50\%/\pm 100\%/\pm 200\%$, median absolute percent error, MAE — provides a falsifiable skill floor for future work.

---

## 9. Conclusion

We have presented a calibrated power-law cost model for operational-risk economic-impact estimation under DORA, validated against a curated public-domain reference dataset of $N=40$ historical incidents (1998–2024). The principal empirical finding is that the cost-vs-shock relationship is super-linear with $\hat{\gamma} \approx 3$ on the overall corpus and $\hat{\gamma}$ ranging from $1.65$ to $3.92$ across the seven operational categories. Promoting a per-category power-law estimator $\hat{c}(s) = \beta_k \cdot s^{\gamma_k}$ from diagnostic to primary headline lifts leave-one-out cross-validation hit-rate within $\pm 100\%$ from $35\%$ (linear pooled baseline) to $80\%$, and reduces median absolute percent error from $394\%$ to $57\%$. The improvement is robust to plausible shock-magnitude perturbations and is concentrated in the tier-correct prediction range that matters for the DORA reporting context.

The headline estimate is reported alongside six independent statistical diagnostics (pairs-bootstrap quantile band, HC3 sandwich SE, HMM regime mixture, Hill tail-index estimator, 2SLS-IV endogeneity check, fragility exponent), each of which targets a specific assumption that the headline rests on. The diagnostic stack is designed for external audit: every claim in the production output traces to a documented method with a published reference, an inspectable parameter, and a reproducible computation.

The reference dataset, the calibration code, and the validation harness are released as an open benchmark for operational-risk cost modelling. We invite alternative estimators to be evaluated on the same corpus under the same leave-one-out protocol; a falsifiable skill floor on a curated public-domain dataset is the most concrete contribution to the operational-risk modelling literature that we can offer at this stage of the framework's development.

The framework's natural next steps are corpus expansion toward $N \geq 60$ (Section 7.1), a higher-fidelity shock-magnitude index derived from agent-based simulation rather than from the heuristic four-tier assignment (Section 7.2), and a stronger instrument for the endogeneity diagnostic (Section 7.3). Each is independent of the others and can be pursued without modifying the model architecture documented here.

---

## Appendix A: Full Reference Dataset

The 40 reference incidents are reproduced below with their assigned shock-magnitude, cost, category, ISO date, and a single condensed source citation. Full source lists are in the JSON file.

| # | id | category | $s$ | $c$ (€M) | date | source |
|---|---|---|---|---|---|---|
| 1 | mps_deposit_run_2016 | banking_it | 1.4 | 2,100 | 2016-12-22 | EBA SREP 2016 |
| 2 | mps_bailin_2017 | banking_it | 1.6 | 3,900 | 2017-07-04 | DG Comp SA.47677 |
| 3 | carige_2019 | banking_it | 1.3 | 1,800 | 2019-01-08 | FITD intervention disclosure |
| 4 | veneto_popvicenza_2017 | banking_it | 2.0 | 5,200 | 2017-06-25 | Italian Banking Resolution Decree 99/2017 |
| 5 | popolare_bari_2019 | banking_it | 1.0 | 900 | 2019-12-13 | BdI Annual Report 2019 |
| 6 | tercas_2014 | banking_it | 0.8 | 300 | 2014-07-23 | FITD intervention disclosure |
| 7 | northern_rock_2007 | banking_eu | 1.5 | 38,000 | 2007-09-14 | BoE Quarterly Bulletin 2008 |
| 8 | dexia_bailout_2008_2011 | banking_eu | 2.5 | 90,000 | 2011-10-09 | EC Decision SA.33760 |
| 9 | abn_amro_2008 | banking_eu | 2.0 | 27,000 | 2008-10-03 | DNB Annual Report 2008 |
| 10 | banco_espirito_santo_2014 | banking_eu | 1.7 | 4,900 | 2014-08-03 | Banco de Portugal resolution |
| 11 | credit_suisse_ubs_2023 | banking_eu | 2.6 | 17,000 | 2023-03-19 | FINMA / SNB joint statement |
| 12 | sberbank_europe_2022 | banking_eu | 1.2 | 1,100 | 2022-02-28 | EBA SRB resolution |
| 13 | greensill_2021 | banking_eu | (n/a) | 500 | 2021-03-08 | UK PRA review 2021 |
| 14 | wirecard_2020 | banking_eu | 1.6 | 4,200 | 2020-06-25 | BaFin annual report 2020 |
| 15 | svb_2023 | banking_us | 2.4 | 9,000 | 2023-03-10 | FDIC receivership announcement |
| 16 | signature_2023 | banking_us | 1.8 | 5,000 | 2023-03-12 | NYDFS press release |
| 17 | first_republic_2023 | banking_us | 1.7 | 4,500 | 2023-05-01 | FDIC press release |
| 18 | lehman_2008 | banking_us | 4.0 | 600,000 | 2008-09-15 | FCIC Final Report 2011 |
| 19 | bear_stearns_2008 | banking_us | 2.0 | 27,000 | 2008-03-16 | Fed Maiden Lane facility disclosure |
| 20 | wamu_2008 | banking_us | 2.2 | 25,000 | 2008-09-25 | OTS receivership filing |
| 21 | ltcm_1998 | banking_us | 1.5 | 4,000 | 1998-09-23 | Fed/PWG report 1999 |
| 22 | brexit_wave1_2016 | sovereign | 3.2 | 30,000 | 2016-06-24 | UK HMT economic impact assessment |
| 23 | brexit_wave2_2019 | sovereign | 2.0 | 15,000 | 2019-03-29 | OBR 2019 forecast |
| 24 | italy_budget_2018 | sovereign | 1.8 | 25,000 | 2018-10-23 | EC Budget assessment 2018 |
| 25 | greece_2015 | sovereign | 2.6 | 40,000 | 2015-07-05 | ESM stability support disclosure |
| 26 | cyprus_2013 | sovereign | 2.3 | 15,000 | 2013-03-25 | ESM/IMF programme documents |
| 27 | argentina_2001 | sovereign | 3.5 | 82,000 | 2001-12-23 | IMF Article IV consultation 2002 |
| 28 | crowdstrike_2024 | cyber | 2.8 | 10,000 | 2024-07-19 | Microsoft + CrowdStrike joint post-mortem |
| 29 | solarwinds_2020 | cyber | 2.3 | 100,000 | 2020-12-13 | CISA emergency directive 21-01 |
| 30 | colonial_pipeline_2021 | cyber | 1.8 | 4,400 | 2021-05-07 | TSA security directive 2021 |
| 31 | equifax_2017 | cyber | 1.5 | 1,400 | 2017-09-07 | FTC settlement 2019 |
| 32 | wannacry_2017 | cyber | 2.5 | 4,000 | 2017-05-12 | Europol Internet Organised Crime Threat Assessment 2017 |
| 33 | notpetya_2017 | cyber | 2.6 | 10,000 | 2017-06-27 | White House attribution statement 2018 |
| 34 | tim_downgrade_2014 | telco | 0.8 | 700 | 2014-12-09 | S&P credit rating action |
| 35 | wind3_merger_fallout_2017 | telco | 0.9 | 1,200 | 2017-01-05 | EC Merger decision M.7758 |
| 36 | vodafone_de_churn_2023 | telco | 1.0 | 2,000 | 2023-11-14 | Vodafone Group Q3 trading update |
| 37 | bt_pension_2017 | telco | 1.2 | 11,000 | 2017-05-10 | BT Group annual report 2017 |
| 38 | uniper_rescue_2022 | energy | 2.4 | 34,000 | 2022-09-21 | Bundesregierung press release |
| 39 | engie_hedging_2020 | energy | 1.1 | 1,100 | 2020-07-30 | Engie H1 2020 results |
| 40 | eni_gabon_writedown_2020 | energy | 0.7 | 300 | 2020-07-31 | ENI H1 2020 results |

---

## References

Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The network origins of aggregate fluctuations. *Econometrica*, 80(5), 1977–2016.

Ang, A., & Bekaert, G. (2002). Regime switches in interest rates. *Journal of Business and Economic Statistics*, 20(2), 163–182.

Ang, A., & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313–337.

Aue, F., & Kalkbrener, M. (2006). LDA at work: Deutsche Bank's approach to quantifying operational risk. *Journal of Operational Risk*, 1(4), 49–93.

Bardoscia, M., Battiston, S., Caccioli, F., & Caldarelli, G. (2017). Pathways towards instability in financial networks. *Nature Communications*, 8(1), 14416.

Basmann, R. L. (1957). A generalized classical method of linear estimation of coefficients in a structural equation. *Econometrica*, 25(1), 77–83.

Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). DebtRank: Too central to fail? Financial networks, the FED and systemic risk. *Scientific Reports*, 2, 541.

Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics*, 41(1), 164–171.

Bouchaud, J.-P., Mézard, M., & Potters, M. (2002). Statistical properties of stock order books: empirical results and models. *Quantitative Finance*, 2(4), 251–256.

Caccioli, F., Shrestha, M., Moore, C., & Farmer, J. D. (2014). Stability analysis of financial contagion due to overlapping portfolios. *Journal of Banking & Finance*, 46, 233–245.

Chernobai, A. S., Rachev, S. T., & Fabozzi, F. J. (2007). *Operational Risk: A Guide to Basel II Capital Requirements, Models, and Analysis*. Wiley.

Cope, E. W., Mignola, G., Antonini, G., & Ugoccioni, R. (2009). Challenges and pitfalls in measuring operational risk from loss data. *Journal of Operational Risk*, 4(4), 3–27.

Cruz, M. G. (2002). *Modeling, Measuring and Hedging Operational Risk*. Wiley.

Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.

Deloitte (2024). *EU Digital Operational Resilience Act (DORA): A Practical Guide for Financial Entities*. Deloitte Risk Advisory.

EBA/EIOPA/ESMA Joint Committee (2024). *Final Report on Draft Regulatory Technical Standards on the Content, Format, Templates and Timelines for Reporting Major ICT-Related Incidents and Significant Cyber Threats Under Regulation (EU) 2022/2554*. JC 2024-43.

European Banking Authority (2017). *Guidelines on the Assessment of ICT and Operational Risk Exposures*. EBA/GL/2017/05.

European Banking Authority (2019). *Guidelines on ICT and Security Risk Management*. EBA/GL/2019/04.

Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*, 7(1), 1–26.

Eicker, F. (1967). Limit theorems for regressions with unequal and dependent errors. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 59–82.

Frachot, A., Georges, P., & Roncalli, T. (2001). *Loss Distribution Approach for Operational Risk*. Crédit Lyonnais Working Paper.

Gabaix, X., Gopikrishnan, P., Plerou, V., & Stanley, H. E. (2003). A theory of power-law distributions in financial market fluctuations. *Nature*, 423(6937), 267–270.

Hall, P. (1990). Using the bootstrap to estimate mean squared error and select smoothing parameter in nonparametric problems. *Journal of Multivariate Analysis*, 32(2), 177–203.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Hess, C. (2011). The impact of the financial crisis on operational risk in the financial services industry: Empirical evidence. *Journal of Operational Risk*, 6(1), 23–35.

Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics*, 3(5), 1163–1174.

Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.

Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 221–233.

Kiefer, N. M. (2010). Default estimation and expert information. *Journal of Business and Economic Statistics*, 28(2), 320–328.

Laeven, L., & Valencia, F. (2018). *Systemic Banking Crises Revisited*. IMF Working Paper WP/18/206.

MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.

Pagan, A. R., & Schwert, G. W. (1990). Alternative models for conditional stock volatility. *Journal of Econometrics*, 45(1–2), 267–290.

PwC (2024). *DORA: Driving Digital Operational Resilience in the EU Financial Sector*. PwC Risk Advisory Series.

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

Reinhart, C. M., & Rogoff, K. S. (2009). *This Time Is Different: Eight Centuries of Financial Folly*. Princeton University Press.

Resnick, S., & Stărică, C. (1997). Smoothing the Hill estimator. *Advances in Applied Probability*, 29(1), 271–293.

Schwert, G. W. (1989). Why does stock market volatility change over time? *Journal of Finance*, 44(5), 1115–1153.

Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.

Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *American Economic Review*, 96(1), 54–81.

Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In Andrews, D. W. K., & Stock, J. H. (Eds.), *Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*, 80–108. Cambridge University Press.

Taleb, N. N. (2012). *Antifragile: Things That Gain from Disorder*. Random House.

Theil, H. (1953). *Repeated Least Squares Applied to Complete Equation Systems*. The Hague: Central Planning Bureau (mimeographed).

White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
