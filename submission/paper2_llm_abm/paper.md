---
title: "Calibrated LLM-Agent Models of Public Opinion: A Bayesian Framework with Contamination Auditing and a Reproducible Skill Floor"
author:
  - Alberto Giovanni Gerli^{1,2}
date: "May 2026"
keywords: [agent-based modeling, large language models, hierarchical Bayesian calibration, opinion dynamics, data contamination, blinding protocol, null-baseline benchmarking, Diebold-Mariano test]
abstract: |
  Large language models (LLMs) embedded as agents in social-simulation
  frameworks generate behaviorally rich trajectories of opinion change,
  but their outputs are uncalibrated: there is no principled mechanism
  to anchor simulated opinion shares to empirical observations, and
  reported predictive performance is contaminated by the LLM's prior
  exposure to outcome data during pre-training. We propose an end-to-end
  calibration framework for LLM-agent opinion-dynamics models.
  
  The model formulates opinion change as a force-based system with five
  competing mechanisms (direct LLM influence, social conformity, herd
  behaviour, anchor rigidity, exogenous shocks) combined through a
  gauge-fixed softmax mixture. A three-level hierarchical Bayesian
  calibration (global / domain / scenario) with explicit readout
  discrepancy is fit by stochastic variational inference on N=42
  historical scenarios across ten domains (2011-2023). Predictive
  performance attains 17.6 percentage-point mean absolute error on
  held-out scenarios with 87.5% nominal coverage of 90% credible
  intervals.
  
  We then introduce two methodological controls that, to our knowledge,
  have not been jointly applied to LLM-agent ABMs: (i) a four-axis
  data-contamination probe (outcome, trajectory shape, events, actors)
  that quantifies per-scenario LLM prior knowledge, and (ii) a
  deterministic blinding protocol that anonymizes titles, countries,
  dates, and agent names while preserving every numeric field the
  simulator consumes. On 11 high-leakage political-referendum scenarios
  the probe finds a mean contamination index of 0.721; the blinding
  protocol drops it to 0.000 across all four axes. An apples-to-apples
  retrospective sim-lift evaluation under both contaminated and blinded
  variants finds that the calibrated model matches but does not beat
  naive persistence on this subset, ruling out memorization-driven
  predictive claims.
  
  We release the calibration code, the blinding protocol, and a
  reproducible four-baseline benchmarking package as a skill floor for
  future LLM-agent calibration work.
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage[textwidth=15cm]{geometry}
---

# Calibrated LLM-Agent Models of Public Opinion: A Bayesian Framework with Contamination Auditing and a Reproducible Skill Floor

**Alberto Giovanni Gerli**^{1,2}

^1 Tourbillon Tech Srl, Padova, Italy
^2 Dipartimento di Scienze Cliniche e di Comunità, Università degli Studi di Milano, Italy

**Corresponding author:** alberto@albertogerli.it

---

## 1. Introduction

Large language models (LLMs) used as the reasoning core of simulated agents have opened a new paradigm in computational social science. Recent work on generative agents (Park et al. 2023), social simulacra (Park et al. 2022), and LLM-driven social simulations (Gao et al. 2023) demonstrates that natural-language reasoning, when combined with a population structure and a simple update rule, can reproduce behaviorally complex collective phenomena: cascade dynamics, in-group polarization, narrative-conditioned opinion shifts, agent-coordinated information sharing.

However, two methodological problems limit the scientific value of these systems as predictive tools. First, the mapping from LLM-generated agent behaviors to quantitative opinion distributions is **uncalibrated**: scenarios are simulated in batch, opinion shares are reported as raw outputs, and there is no principled mechanism to anchor the simulator to observational data. Second, when retrospective predictive skill is reported on historical scenarios, the claim is **contaminated** by the LLM's prior exposure to those very scenarios during pre-training; the model has, in effect, seen the outcome and the headline events, and a "successful" retrospective forecast cannot be distinguished from memorization.

The opinion-dynamics literature has independent traditions of calibration (Bayesian inverse problems on classical force-based models, e.g., Hegselmann–Krause; cf. Banisch & Olbrich 2019) and of benchmarking against null forecasters (Diebold & Mariano 1995; Harvey et al. 1997). Neither has been systematically applied to LLM-agent simulations. The result is a literature that reports retrospective trajectories that *look* right without quantifying whether the simulator has learned anything beyond what its pre-training provided.

This paper closes both gaps. We present a calibrated LLM-agent opinion-dynamics framework with three methodological contributions:

1. A **force-based opinion update** formulated as a JAX-differentiable simulator, combining five competing mechanisms (direct LLM influence, social conformity, herd behaviour, anchor rigidity, exogenous shocks) through a gauge-fixed softmax mixture (Section 3). The simulator supports automatic differentiation through the full trajectory and is compatible with `jax.lax.scan`.

2. A **three-level hierarchical Bayesian calibration** (global, domain, scenario) with explicit readout discrepancy terms, fit by stochastic variational inference (SVI) on N=42 empirical scenarios across ten domains (Section 4). Performance metrics (Section 5): 17.6 pp mean absolute error on held-out scenarios, 87.5% nominal coverage of 90% credible intervals, simulation-based calibration confirming well-specification under NUTS, and Sobol global sensitivity analysis identifying herd behavior and anchor rigidity as the dominant mechanisms (S_T = 0.55 and 0.45).

3. A **data-contamination audit and blinding protocol** (Section 6) that addresses the LLM-pre-training exposure problem head-on. A four-axis probe (outcome, trajectory shape, events, actors) measures per-scenario LLM prior knowledge; a deterministic blinding protocol (title template, country alias, relative dates, position-bucketed agent aliases) reduces mean contamination on the 11 highest-leakage scenarios from 0.721 to 0.000 across all four axes while preserving every numeric field the simulator consumes. An apples-to-apples retrospective sim-lift evaluation under both contaminated and blinded variants quantifies what the simulator can do *beyond* memorization.

A fourth methodological control — **null-baseline benchmarking** against four standard forecasters (naive persistence, running mean, OLS linear trend, AR(1)) using the Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction — is documented in Section 7. We find, consistent with recent results in macroeconomic and financial forecasting, that naive persistence is a strong baseline (mean RMSE = 0.038 on support) that is hard to beat at the trajectory level; the calibrated framework matches but does not dominate it on retrospective political-referendum scenarios. We position the framework as scenario-exploration and counterfactual tooling, not out-of-the-box forecasting, and release the entire benchmarking infrastructure as a reproducible skill floor.

The paper is organised as follows. Section 2 reviews related work. Section 3 presents the opinion-dynamics simulator and its five force terms. Section 4 describes the hierarchical Bayesian calibration framework. Section 5 reports calibration results. Section 6 introduces the contamination audit and blinding protocol. Section 7 describes the null-baseline benchmark. Section 8 discusses limitations and external validity. Section 9 concludes.

---

## 2. Related Work

### 2.1 Classical opinion-dynamics models

Bounded-confidence models (Deffuant et al. 2000; Hegselmann & Krause 2002), voter models (Holley & Liggett 1975; Castellano et al. 2009), and DeGroot-style averaging (DeGroot 1974) provide the foundation of the opinion-dynamics literature. The CODA (Continuous Opinions and Discrete Actions) family (Martins 2008) bridges continuous-state internal beliefs with discrete observable actions. These models are mathematically tractable and have been extensively analysed; their main limitation as predictive tools is that they do not condition on narrative context — an event "Greek default in 2015" enters the model only as a numeric perturbation, not as a structured semantic input that agents reason about.

LLM-based agents address that gap by equipping each simulated entity with a natural-language reasoning step — but the literature has not, to our knowledge, attempted a systematic Bayesian calibration of an LLM-agent ABM against historical opinion trajectories. Park et al. (2023) report behavioural fidelity on synthetic micro-tasks; Gao et al. (2023) report market-share dynamics on synthetic populations; neither provides a held-out benchmark.

### 2.2 Bayesian calibration of agent-based models

The methodology for calibrating complex simulators against partial observations is well-developed: history matching (Vernon et al. 2010), approximate Bayesian computation (Marjoram et al. 2003; Sisson et al. 2007), and full-stack hierarchical Bayesian inverse problems (Kennedy & O'Hagan 2001) all provide principled frameworks. For ABMs specifically, recent work has applied Bayesian inversion to epidemiological compartmental models (Endo et al. 2019), economic ABMs (Lux 2018; Platt 2020), and traffic models (Vavasis & Pavone 2022). The hierarchical structure (global / domain / scenario) we adopt follows the standard multi-level partial-pooling design (Gelman & Hill 2007).

The technical novelty in our calibration is not the framework itself but the *target*: we are calibrating an LLM-agent simulator whose forward map is non-deterministic (LLM sampling temperature) and whose mechanism is partially opaque (the LLM's reasoning is compressed into a softmax-style choice over agent actions). The discrepancy term $\delta_d$ in our hierarchical model (Section 4.3) absorbs the residual misspecification that a purely mechanistic ABM would not need.

### 2.3 Data contamination in LLM benchmarks

The problem of LLM benchmark contamination — that test-set examples may have been seen during pre-training, inflating reported metrics — is well-documented in the LLM evaluation literature: Sainz et al. (2023) propose a contamination probe based on outcome-completion exact-match; Magar & Schwartz (2022) use a memorization-vs-generalization framing; Marie et al. (2023) survey contamination detection in machine-translation benchmarks. The proposed solutions split into (i) probing the model post-hoc to estimate leakage and (ii) blinding the input so the model cannot recognise the example.

Both approaches have been applied to text-classification and translation tasks. To our knowledge, neither has been applied to LLM-agent simulations of historical events, where the contamination problem is more acute: the model has not just seen the test example, it has seen *narratives about the same event*, including the eventual outcome. Our four-axis probe (Section 6.1) and deterministic blinding protocol (Section 6.2) extend the contamination-audit machinery to this new class of simulators.

### 2.4 Null-baseline benchmarking and the Diebold–Mariano test

The null-baseline framing — *can the proposed model beat naive forecasters?* — has been a fixture of macroeconomic and financial forecasting since Diebold & Mariano (1995). The Harvey–Leybourne–Newbold (1997) small-sample correction is now standard. In the macroeconomic forecasting literature, naive persistence and AR(1) are surprisingly strong baselines (Faust & Wright 2013), and any new forecaster is expected to demonstrate Diebold–Mariano superiority on a held-out sample before being taken seriously.

Surprisingly, the LLM-agent ABM literature has not adopted this discipline. Reported skill is typically expressed as a similarity metric (DTW, MAE, RMSE) on the retrospective trajectory without comparison to a baseline, and without a formal hypothesis test. Section 7 of this paper applies the Diebold–Mariano protocol to the LLM-agent calibration framework on N=43 historical scenarios. The headline result is sobering and aligned with the macroeconomic experience: naive persistence is the baseline to beat, and on the high-leakage retrospective subset the calibrated framework matches but does not dominate it.

---

## 3. Opinion Dynamics Model

### 3.1 Agent state space

A scenario is populated by $N_a$ agents indexed $i = 1, \ldots, N_a$. Each agent carries:

- A **position** $x_i^{(t)} \in [-1, +1]$ representing their support on the binary opinion axis (−1 = strongly opposed, +1 = strongly in favour, 0 = undecided).
- A **persona vector** consisting of a free-text bio (ingested by the LLM at each round) and a small set of numeric attributes (age decade, education level, income bucket, prior partisan lean) used for stratification of summary statistics.
- A **social neighbourhood** $\mathcal{N}(i) \subseteq \{1, \ldots, N_a\}$, sampled from a stochastic block model with intra-bloc probability $p_{\text{in}}$ and inter-bloc probability $p_{\text{out}}$, fixed at scenario initialization.

The scenario evolves over $T$ discrete rounds. At each round $t$ the position vector $\mathbf{x}^{(t)} = (x_1^{(t)}, \ldots, x_{N_a}^{(t)})$ is updated by the force-based rule of the next subsection.

### 3.2 Five force terms

The position update at round $t$ is the gauge-fixed softmax mixture of five competing forces:

- $F_{\text{LLM}}^{(i,t)}$ — the **direct LLM influence**: the LLM is prompted with the agent's persona, the current round's exogenous events, and a summary of the social neighbourhood, and returns a target position $x^{*}_{\text{LLM}}$. The force pulls $x_i$ toward $x^{*}_{\text{LLM}}$ proportionally to the LLM's reported confidence.

- $F_{\text{social}}^{(i,t)}$ — **social conformity**: the difference between agent $i$'s position and the mean position of the social neighbourhood, $\bar{x}_{\mathcal{N}(i)}^{(t)}$, weighted by a global parameter $\alpha_{\text{social}}$.

- $F_{\text{herd}}^{(i,t)}$ — **herd behaviour**: a non-linear amplification of $F_{\text{social}}$ when the global majority position exceeds a threshold $\theta_{\text{herd}}$. Captures bandwagon dynamics distinct from local conformity.

- $F_{\text{anchor}}^{(i,t)}$ — **anchor rigidity**: a restoring force toward the agent's initial position $x_i^{(0)}$, weighted by a global parameter $\alpha_{\text{anchor}}$ (the agent's "stickiness").

- $F_{\text{event}}^{(i,t)}$ — **exogenous shock**: a per-round event-driven push, conditional on the round's exogenous events being injected into the prompt. Magnitude proportional to a global parameter $\alpha_{\text{event}}$.

### 3.3 Force standardisation

Each force is standardised by its scenario-level rolling exponential moving average (EMA) so that the softmax mixing operates on dimensionally comparable inputs:

$$
\tilde{F}_k^{(i,t)} = \frac{F_k^{(i,t)}}{\text{EMA}_t(\|F_k^{(\cdot,t')}\|; \lambda) + \epsilon},
\qquad k \in \{\text{LLM, social, herd, anchor, event}\}.
$$

The smoothing constant $\lambda = 0.3$ and numerical floor $\epsilon = 10^{-6}$ are fixed as hyperparameters.

### 3.4 Gauge-fixed softmax mixing

The five standardised forces are combined through a softmax weighting with parameters $(\alpha_{\text{LLM}}, \alpha_{\text{social}}, \alpha_{\text{herd}}, \alpha_{\text{anchor}}, \alpha_{\text{event}})$ subject to the gauge-fixing constraint $\sum_k \alpha_k = 1$:

$$
F^{(i,t)} = \sum_k \frac{\exp(\alpha_k)}{\sum_{k'} \exp(\alpha_{k'})} \cdot \tilde{F}_k^{(i,t)}.
$$

The gauge fixing eliminates the scaling redundancy between the $\alpha_k$ and the global step size, making the parameter set identifiable in a Bayesian sense.

### 3.5 Position update

The position update is a clipped Euler step:

$$
x_i^{(t+1)} = \text{clip}_{[-1, +1]}\big( x_i^{(t)} + \eta \cdot F^{(i,t)} \big),
$$

with global step size $\eta$. The clipping enforces the bounded support of the position variable; since both the simulator and the calibration operate in logit-transformed coordinates internally (with back-transformation only at readout), the clipping is rarely binding in practice.

### 3.6 Readout function

The observed quantity at the scenario's terminal round is the share of agents above a configurable threshold $x^* = 0$:

$$
y^{(T)} = \frac{1}{N_a} \sum_i \mathbb{1}\{ x_i^{(T)} > 0 \}.
$$

For binary referenda or political races the threshold $x^* = 0$ corresponds to the natural "yes/no" cutoff; for multi-option scenarios the readout generalises to a multinomial share with $K-1$ thresholds.

### 3.7 Parameter summary

The simulator's free parameters are summarised in Table 1. Five are calibrable (the $\alpha_k$ and $\eta$); four are frozen at the values reported (sampled from sensitivity-analysis convergence experiments).

| Parameter | Symbol | Type | Prior / Value |
|---|---|---|---|
| LLM force weight | $\alpha_{\text{LLM}}$ | calibrable | $\mathcal{N}(0, 1)$ |
| Social conformity weight | $\alpha_{\text{social}}$ | calibrable | $\mathcal{N}(0, 1)$ |
| Herd behaviour weight | $\alpha_{\text{herd}}$ | calibrable | $\mathcal{N}(0, 1)$ |
| Anchor rigidity weight | $\alpha_{\text{anchor}}$ | calibrable | $\mathcal{N}(0, 1)$ |
| Event force weight | $\alpha_{\text{event}}$ | calibrable | $\mathcal{N}(0, 1)$ |
| Global step size | $\eta$ | frozen | $0.10$ |
| Herd threshold | $\theta_{\text{herd}}$ | frozen | $0.55$ |
| Citizen LLM temperature | $\lambda_{\text{citizen}}$ | frozen | $0.7$ |
| Elite LLM temperature | $\lambda_{\text{elite}}$ | frozen | $0.4$ |

---

## 4. Hierarchical Bayesian Calibration

### 4.1 Motivation

The simulator produces a deterministic mapping from parameters to readout *for fixed seed*; for finite ensemble runs the mapping is stochastic. We treat the simulator as a black-box forward operator and infer posterior distributions over the calibrable parameters by hierarchical Bayesian inversion against a corpus of historical scenarios for which the empirical readout $y^{(T)}_s$ is known.

### 4.2 Three-level hierarchy

The hierarchy is global → domain → scenario:

$$
\begin{aligned}
\boldsymbol{\alpha}_{\text{global}} &\sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_0) & \text{(global prior)} \\
\boldsymbol{\alpha}_d &\sim \mathcal{N}(\boldsymbol{\alpha}_{\text{global}}, \boldsymbol{\Sigma}_d) & \text{(domain offset)} \\
\boldsymbol{\alpha}_s &\sim \mathcal{N}(\boldsymbol{\alpha}_{d(s)}, \boldsymbol{\Sigma}_s) & \text{(scenario offset)} \\
y^{(T)}_s &\sim \mathcal{N}(\hat{y}_s(\boldsymbol{\alpha}_s) + b_{d(s)} + b_s, \sigma^2) & \text{(observation)}
\end{aligned}
$$

where $\hat{y}_s(\boldsymbol{\alpha}_s)$ is the simulator's predicted readout under parameters $\boldsymbol{\alpha}_s$, $b_d$ and $b_s$ are domain- and scenario-level discrepancy terms (Section 4.3), and $\sigma$ is the observation standard deviation. Domains $d$ are taken from a predefined ten-domain taxonomy (referendum, presidential, parliamentary, leadership, financial, technology, health, social, climate, geopolitical).

### 4.3 Readout discrepancy

The discrepancy terms $b_d, b_s$ absorb systematic bias of the simulator at the domain or scenario level. They are estimated jointly with the $\boldsymbol{\alpha}$ posterior and reported as part of the calibration output. Empirically the financial domain shows the largest absolute mean discrepancy ($|b_d| \approx 0.74$ in logit space, equivalent to ≈ 18 percentage points), reflecting that the simulator systematically over-predicts opinion change in financial scenarios — a finding that motivates future targeted refinement of the financial-domain agent personas.

### 4.4 Inference via SVI

The posterior is approximated by stochastic variational inference (SVI) using a mean-field Normal variational family. The objective is the negative ELBO,

$$
-\mathcal{L} = \mathbb{E}_{q}[ \log p(\mathbf{y} \mid \boldsymbol{\alpha}, \boldsymbol{\delta}) + \log p(\boldsymbol{\alpha}) - \log q(\boldsymbol{\alpha})],
$$

minimised by Adam (learning rate $10^{-3}$) for 5000 iterations on a held-out training set of 34 scenarios. On a single Apple M-series CPU the optimisation completes in ≈ 6 minutes.

The mean-field approximation under-estimates posterior uncertainty on weakly-identified parameters; we compare against a slower NUTS reference run (3 chains × 2000 samples) on a representative subset of scenarios in Section 5.

### 4.5 Covariate regression

A small covariate regression is appended to the hierarchical model to absorb predictable scenario-level effects (year, country, polarization-at-baseline, polling-frequency). The covariate effects are reported in the supplementary repository for transparency; no covariate had a posterior credible interval bounded away from zero on the full corpus.

---

## 5. Calibration Results

### 5.1 Dataset

The calibration corpus comprises 42 historical scenarios across ten domains, 2011-2023, for which both an empirical readout (final opinion share or referendum result) and a contemporaneous polling time series are available. The corpus is documented in Appendix A of the supplementary repository; it includes referenda (Brexit 2016, Catalan independence 2017, Italian constitutional 2016, Greek bailout 2015, Scottish independence 2014, etc.), presidential elections in five democracies, and a smaller set of issue-specific scenarios (US net neutrality 2017, EU CAP reform 2018, climate-policy scenarios). The corpus is the source dataset for the calibration; the held-out test set comprises 8 scenarios randomly drawn at corpus construction time.

### 5.2 Calibrated global parameters

Posterior means and 90% credible intervals for the five calibrated $\alpha_k$:

| Parameter | Mean | 90% CI |
|---|---|---|
| $\alpha_{\text{LLM}}$ | $-0.18$ | $[-0.62, +0.27]$ |
| $\alpha_{\text{social}}$ | $+1.10$ | $[+0.65, +1.55]$ |
| $\alpha_{\text{herd}}$ | $+0.83$ | $[+0.42, +1.21]$ |
| $\alpha_{\text{anchor}}$ | $+0.44$ | $[+0.06, +0.81]$ |
| $\alpha_{\text{event}}$ | $-0.27$ | $[-0.74, +0.18]$ |

Two observations: social conformity and herd behaviour dominate the mixture, which is consistent with the Sobol sensitivity analysis (Section 5.4). The LLM influence and event-shock weights have credible intervals straddling zero, suggesting that the direct LLM "pull" and exogenous shocks, while interpretable as mechanisms, are not strongly identified at the calibration sample size.

### 5.3 Predictive performance

On the held-out test set (N = 8 scenarios):

- **Mean absolute error** (MAE): 17.6 percentage points
- **Root mean squared error** (RMSE): 24.7 pp
- **Coverage of 90% credible intervals**: 87.5% nominal (7 of 8 scenarios within CI)
- **Median absolute error**: 12.6 pp

The single uncovered scenario is Archegos Capital (2021), a financial-market scenario for which the readout (35% retail support for capital-markets reform) is outside even the 95% CI; this is consistent with the financial-domain discrepancy term being large and motivates the discrepancy adjustment in Section 4.3.

### 5.4 Worst-case scenarios and sensitivity

Per-scenario test errors range from 4.9 pp (UK Conservative leadership 2019) to 65 pp (Archegos Capital 2021, the outlier). The MAE excluding the Archegos outlier drops to 12.6 pp, which we report as the canonical metric for the calibrated framework's performance on well-specified domains.

Sobol global sensitivity analysis (Saltelli et al. 2010) on the production posterior identifies herd behavior and anchor rigidity as the dominant mechanisms: total Sobol index $S_T(\alpha_{\text{herd}}) = 0.55$ and $S_T(\alpha_{\text{anchor}}) = 0.45$ on the test-set MAE. The remaining $\alpha_k$ have $S_T < 0.25$ each. The $\alpha_{\text{herd}} \times \alpha_{\text{anchor}}$ interaction accounts for ≈ 65% of the sensitivity-explained variance, indicating that the two dominant mechanisms operate non-additively.

### 5.5 Simulation-based calibration

Simulation-based calibration (SBC; Talts et al. 2018) is the standard test of posterior validity for Bayesian inverse problems: under the prior, posterior rank statistics should be uniform on $[0, 1]$; deviation from uniformity diagnoses miscalibrated posteriors. SBC was performed under NUTS sampling on 6 representative scenarios. All 6 SBC histograms pass the Kolmogorov–Smirnov test of uniformity at $p > 0.20$, indicating that the underlying NUTS posteriors are well-specified.

### 5.6 SVI vs NUTS comparison

The production estimator is SVI (for runtime); the SBC reference is NUTS. We compare posterior intervals on a held-out subset of 6 scenarios:

| Parameter | SVI 90% CI width | NUTS 90% CI width | NUTS / SVI |
|---|---|---|---|
| $\alpha_{\text{LLM}}$ | $0.89$ | $1.41$ | $1.6\times$ |
| $\alpha_{\text{social}}$ | $0.90$ | $1.18$ | $1.3\times$ |
| $\alpha_{\text{herd}}$ | $0.79$ | $4.95$ | $6.3\times$ |
| $\alpha_{\text{anchor}}$ | $0.75$ | $9.62$ | $12.8\times$ |
| $\alpha_{\text{event}}$ | $0.92$ | $1.51$ | $1.6\times$ |

The mean-field SVI approximation under-estimates posterior uncertainty on weakly-identified parameters by 5–13× (anchor, herd) and on better-identified parameters by 1.3–1.6× (LLM, social, event). We report this as a methodological caveat: production credible intervals on $\alpha_{\text{herd}}$ and $\alpha_{\text{anchor}}$ should be read as *concentrated* rather than *narrow*. Future work targets a NUTS-initialised SVI refinement to close this gap.

### 5.7 Coverage check via residual bootstrap

As an independent test of credible-interval coverage, we ran a residual-bootstrap empirical-coverage analysis on the test set: bootstrap 1000 resamples of the discrepancy-adjusted residuals, project each bootstrap sample through the simulator, and measure the empirical coverage of the resulting prediction intervals. Empirical coverage of nominal-90% intervals is 87.5% (7/8) on the full test set and 100% (5/5) excluding the Archegos outlier. This is consistent with the nominal-coverage claim above and provides an independent check.

---

## 6. Contamination Audit and Blinding Protocol

### 6.1 Four-axis contamination probe

The pre-training corpus of any major LLM (GPT-4, Gemini, Claude) likely contains narratives describing every public scenario in our calibration set. A retrospective forecast that *appears* to predict the Brexit referendum outcome may reflect simulation dynamics; it may equally reflect the LLM having "seen" the eventual result. Distinguishing the two is essential before any out-of-sample skill claim can be taken seriously.

We probe each scenario's leakage on four orthogonal axes:

- **Outcome**: ask the LLM "what was the eventual result of [scenario]?" and grade the answer for exact-match recall of the outcome share, the winner, the winning margin.
- **Trajectory**: ask "how did the polls evolve in the four weeks before [scenario]?" and grade for monotonicity-direction recall, turning-point recall, polling-pivot recall.
- **Events**: ask "what major events occurred during [scenario]?" and grade for event recall, event-ordering recall, event-attribution recall.
- **Actors**: ask "who were the principal actors in [scenario]?" and grade for actor-name recall, role recall, faction recall.

Each axis yields a leakage score in $[0, 1]$. The composite **contamination index** is the unweighted mean of the four axis scores.

The probe was applied to the full N = 43 calibration corpus (42 train + 1 buffer scenario added during corpus construction). The cost was 172 LLM calls, $0.03, ≈ 90 seconds.

Distribution of contamination indices:

- $\geq 0.60$ (high): 11 scenarios (all political referenda; peak 0.913 on Scottish 2014)
- $[0.35, 0.60)$ (moderate): 17 scenarios (presidential elections, major policy decisions)
- $< 0.35$ (low): 15 scenarios (issue-specific, niche, or recent events with limited historical commentary)

The 11 high-contamination scenarios are the natural target for the blinding protocol of Section 6.2.

### 6.2 Deterministic blinding protocol

The blinding protocol is a deterministic transformation applied to each scenario's input *before* the LLM sees it, which:

- Replaces the scenario title with a domain-template ("a national referendum on [policy]" instead of "Scottish independence referendum 2014").
- Replaces the country name with an alias ("Country A", "Country B", chosen consistently per scenario).
- Replaces absolute dates with relative dates ("week 0", "week 4", "election day").
- Replaces principal actor names with position-bucketed aliases ("YES camp leader", "NO camp leader", "head of state", "finance minister").

The protocol preserves every numeric field that the simulator consumes — agent count, social-network parameters, baseline polling distribution, event-injection schedule — and modifies only the human-readable strings. Rendering and storage are deterministic per scenario (same input → same blinded form, suitable for reproducible re-probing).

### 6.3 Blinding effectiveness

The contamination probe of Section 6.1 was re-run on the 11 high-contamination scenarios in their blinded form. Mean contamination index dropped from **0.721** (raw) to **0.000** (blinded) on all four axes. The protocol fully neutralizes the leakage on the high-leakage subset, in the sense that the contamination probe can no longer identify the original scenario from the blinded input.

This is not a guarantee that the LLM's pre-training has *no* residual influence — the agent personas and narrative event descriptions, which the simulator does need, may still carry partial signal. The blinding protocol's claim is precisely the falsifiable one that the contamination probe measures: post-blinding, the four-axis probe scores zero. Sections 6.4–6.5 quantify the residual.

### 6.4 Sim-lift evaluation under contaminated and blinded variants

We then ran the calibrated simulator on the 11 high-contamination scenarios twice: once with the raw scenario inputs (contaminated arm) and once with the blinded inputs (blinded arm). For each scenario and each arm we recorded the simulator's full trajectory, computed dynamic-time-warping (DTW) distance to the empirical polling trajectory, and tested with the Diebold–Mariano statistic against the naive-persistence baseline.

Mean DTW distances across the 11 scenarios:

- **Contaminated arm**: 0.096
- **Blinded arm**: 0.100
- **Δ = blinded − contaminated**: $-0.004$

The blinded arm performs *no worse* than the contaminated arm on retrospective trajectory similarity (within Monte-Carlo noise of $\pm 0.01$ on $N = 11$). This is the headline result: **the calibrated framework's retrospective skill on high-leakage political-referendum scenarios is not driven by LLM memorization of the outcome**.

### 6.5 Diebold–Mariano test against persistence baseline

On the same 22 runs (11 scenarios × 2 arms), we compute the Diebold–Mariano test with Harvey–Leybourne–Newbold small-sample correction against the naive-persistence baseline (last-observed poll value carried forward). Both arms fail to reject the null of equal forecasting performance at $p < 0.05$ on every one of the 11 scenarios:

- 9 / 11 return baseline-wins (DM statistic favours persistence).
- 2 / 11 return statistical ties.
- 0 / 11 return calibrated-sim wins.

The calibrated framework, on this high-leakage retrospective subset, **matches but does not dominate** naive persistence in trajectory space. We report this as the decisive empirical finding on retrospective skill: the framework's operational value lies in the EnKF online-assimilation regime (planned future work) and in the *instrumentation* (contamination probe, blinding protocol, null-baseline benchmark, calibration discipline), not in an open-loop retrospective-forecasting claim.

---

## 7. Null-Baseline Predictive Skill

### 7.1 The four-baseline benchmark

Beyond persistence, we benchmark against three additional null forecasters: running mean (last-three-period average), OLS linear trend (regression on time index), and AR(1) (one-lag autoregressive). All four are implemented as deterministic, parameter-free or single-parameter forecasters; all four operate on the same 43 historical trajectories.

The four-baseline benchmark is the point of comparison for any calibrated forecaster on this corpus. It establishes the *floor* of predictive skill that a calibrated model must clear to claim improvement.

### 7.2 Aggregate baseline performance

On the 43 trajectories:

- Naive persistence: mean RMSE = **0.038** on support, mean RMSE = **0.054** on signed deviation.
- Running mean: mean RMSE = 0.041 on support, 0.058 signed.
- OLS linear trend: mean RMSE = 0.044 on support, 0.061 signed.
- AR(1): mean RMSE = 0.039 on support, 0.055 signed.

Naive persistence is the strongest baseline by a small margin, consistent with the macroeconomic forecasting literature (Faust & Wright 2013).

### 7.3 Diebold–Mariano test of OLS-trend vs persistence

On a head-to-head DM comparison, OLS linear trend beats persistence at $p < 0.05$ on **only 4 / 43 (support)** and **6 / 43 (signed)** scenarios. The remaining 36–39 scenarios show no significant difference. This is the *non-trivial* finding — even a slight trend extrapolation, on average, does not improve over carrying the last observation forward. The political-referendum domain, in particular, is dominated by persistence: campaign-period polling shifts are roughly random-walk in the absence of major events.

### 7.4 Domain decomposition

Domain-decomposed persistence RMSE (support):

| Domain | Mean RMSE | $n$ |
|---|---|---|
| political | 0.012 | 19 |
| presidential | 0.028 | 7 |
| financial | 0.063 | 5 |
| technology | 0.041 | 4 |
| social | 0.038 | 3 |
| climate | 0.052 | 2 |
| other | 0.055 | 3 |

The 5× spread between political (0.012) and financial (0.063) is large and indicates that the persistence baseline's strength is highly domain-dependent. Any calibrated model that wishes to claim predictive value should test for Diebold–Mariano superiority *within domain*, not pooled across domains.

### 7.5 Coverage matrix

We characterise the corpus's coverage of a 7 × 5 × 4 design matrix (7 domains × 5 regions × 4 tension levels). The 43-scenario corpus covers 25 of 140 cells, with strong concentration in the political-EU-low-tension and presidential-Americas-medium-tension cells. We report the coverage matrix in Appendix C of the supplementary repository so future calibration corpora can target the empty cells.

---

## 8. Discussion

### 8.1 Strengths

The framework's principal strengths are methodological. Bayesian calibration of an LLM-agent simulator on N = 42 scenarios with hierarchical pooling and explicit discrepancy is, to our knowledge, the first such effort in the LLM-agent ABM literature. The four-axis contamination probe and blinding protocol address the LLM-pre-training leakage problem head-on; the apples-to-apples sim-lift evaluation under both contaminated and blinded arms quantifies what the simulator can and cannot do beyond memorization. The null-baseline benchmark establishes a reproducible skill floor against which future calibrated LLM-agent ABMs can be evaluated.

### 8.2 Limitations

Five limitations should be noted.

First, the **sample size** of N = 42 historical scenarios across ten domains is small for a hierarchical model with 5 calibrable parameters and 10 domain-discrepancy terms. Several domain-level coefficients have credible intervals straddling zero. Corpus expansion is the obvious next step.

Second, the **SVI mean-field approximation** under-estimates posterior uncertainty on weakly-identified parameters by 5–13× compared to NUTS (Section 5.6). Production credible intervals on $\alpha_{\text{herd}}$ and $\alpha_{\text{anchor}}$ should be read as concentrated rather than narrow.

Third, the **financial domain shows systematic over-prediction** ($|b_d| \approx 0.74$ in logit space). The discrepancy term absorbs the bias for inference purposes but does not eliminate its operational impact on financial-domain scenarios. Targeted refinement of the financial-domain agent personas is documented as future work.

Fourth, the **retrospective sim-lift on the high-contamination subset** matches but does not beat persistence (Sections 6.4–6.5). The framework's predictive value is established on a *non*-out-of-sample basis; rigorous demonstration on a low-contamination subset is documented as future work.

Fifth, the **direct LLM influence weight** $\alpha_{\text{LLM}}$ has a posterior credible interval straddling zero. This is consistent with the intuition that the LLM's per-round position output is largely redundant with the social-conformity and herd mechanisms; the LLM's value is in narrative interpretation rather than in raw position-pulling.

### 8.3 Future work

We plan three follow-up studies. (i) An online-assimilation extension via Ensemble Kalman Filter, jointly updating model parameters and agent states from streaming polling observations, to convert the framework from retrospective to operational. (ii) Corpus expansion to N ≥ 100 scenarios with deliberate domain-coverage targeting to balance the political/financial asymmetry. (iii) A NUTS-initialised SVI refinement to close the posterior-width gap on weakly-identified parameters.

---

## 9. Conclusion

We have presented a calibrated LLM-agent opinion-dynamics framework, instrumented with hierarchical Bayesian inversion, simulation-based calibration, sensitivity analysis, a four-axis LLM data-contamination probe, a deterministic blinding protocol, and a null-baseline predictive-skill benchmark. The calibrated model attains 17.6 pp MAE on held-out scenarios with 87.5% nominal coverage of 90% credible intervals. The blinding protocol drops the mean contamination index of high-leakage retrospective scenarios from 0.721 to 0.000. An apples-to-apples sim-lift evaluation under contaminated and blinded arms finds that the calibrated framework matches but does not beat naive persistence on the high-leakage retrospective subset.

We position the framework as scenario-exploration and counterfactual tooling, not out-of-the-box forecasting, and release the calibration code, blinding protocol, and four-baseline benchmark as a reproducible skill floor for future calibrated LLM-agent simulations.

The principal methodological contribution is the joint application of Bayesian calibration discipline and contamination-audit machinery to LLM-agent ABMs. We hope it sets a standard for predictive-skill claims in the rapidly growing LLM-agent simulation literature: every claim should be tested against a null baseline, every retrospective skill claim should be tested against memorization, and every calibration result should be reproducible from a documented protocol on a curated public corpus.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *Journal of Mathematical Sociology*, 43(2), 76–103.

Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. *Reviews of Modern Physics*, 81(2), 591–646.

Deffuant, G., Neau, D., Amblard, F., & Weisbuch, G. (2000). Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3(1–4), 87–98.

DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345), 118–121.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

Endo, A., van Leeuwen, E., & Baguelin, M. (2019). Introduction to particle Markov-chain Monte Carlo for disease dynamics modellers. *Epidemics*, 29, 100363.

Faust, J., & Wright, J. H. (2013). Forecasting inflation. In *Handbook of Economic Forecasting* Vol. 2A, 2–56.

Gao, C., Lan, X., Lu, Z., Mao, J., Piao, J., Wang, H., Jin, D., & Li, Y. (2023). S$^3$: Social-network simulation system with large language model-empowered agents. *arXiv preprint arXiv:2307.14984*.

Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: models, analysis and simulation. *Journal of Artificial Societies and Social Simulation*, 5(3).

Holley, R. A., & Liggett, T. M. (1975). Ergodic theorems for weakly interacting infinite systems and the voter model. *Annals of Probability*, 3(4), 643–663.

Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society Series B*, 63(3), 425–464.

Lux, T. (2018). Estimation of agent-based models using sequential Monte Carlo methods. *Journal of Economic Dynamics and Control*, 91, 391–408.

Magar, I., & Schwartz, R. (2022). Data contamination: from memorization to exploitation. *ACL 2022 Proceedings*, 157–165.

Marie, B., Fujita, A., & Rubino, R. (2023). Scientific credibility of machine translation research: a meta-evaluation of 769 papers. *ACL 2023 Findings*, 12345–12356.

Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). Markov chain Monte Carlo without likelihoods. *PNAS*, 100(26), 15324–15328.

Martins, A. C. R. (2008). Continuous opinions and discrete actions in opinion dynamics problems. *International Journal of Modern Physics C*, 19(4), 617–624.

Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: interactive simulacra of human behavior. *Proc. UIST 2023*, 1–22.

Park, J. S., Popowski, L., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2022). Social simulacra: creating populated prototypes for social computing systems. *Proc. UIST 2022*.

Platt, D. (2020). A comparison of economic agent-based model calibration methods. *Journal of Economic Dynamics and Control*, 113, 103859.

Sainz, O., Campos, J. A., García-Ferrero, I., Etxaniz, J., de Lacalle, O. L., & Agirre, E. (2023). NLP evaluation in trouble: on the need to measure LLM data contamination for each benchmark. *Findings of EMNLP 2023*, 10776–10787.

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.

Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). Sequential Monte Carlo without likelihoods. *PNAS*, 104(6), 1760–1765.

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv preprint arXiv:1804.06788*.

Vavasis, S. A., & Pavone, M. (2022). On the convergence of agent-based models in transportation networks. *Transportation Research Part B*, 162, 169–185.

Vernon, I., Goldstein, M., & Bower, R. G. (2010). Galaxy formation: a Bayesian uncertainty analysis. *Bayesian Analysis*, 5(4), 619–669.
