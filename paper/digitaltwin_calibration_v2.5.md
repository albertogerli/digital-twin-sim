# DigitalTwinSim: Bayesian Calibration, Null-Baseline Benchmarking, and Online Data Assimilation for LLM-Agent Opinion Dynamics Simulation

**Version 2.5** — adds a null-baseline benchmarking layer (Diebold–Mariano with Harvey–Leybourne–Newbold correction, residual-bootstrap coverage, and a 7×5×4 scenario-diversity matrix) to the v2.4 calibration and EnKF framework. The v2.4 calibration numbers are unchanged; Section 6.6–6.8 and Appendix D are new.

**Alberto Giovanni Gerli**^{1,2}

^1 Tourbillon Tech Srl, Padova, Italy
^2 Dipartimento di Scienze Cliniche e di Comunità, Università degli Studi di Milano, Italy

**Keywords:** opinion dynamics, agent-based modeling, large language models, Bayesian calibration, data assimilation, ensemble Kalman filter, digital twin, computational social science, multi-modal calibration, financial market linkage, predictive skill, Diebold–Mariano test, null baselines

---

## Abstract

We present DigitalTwinSim, a computational framework that combines large language model (LLM)-driven agent-based simulation with Bayesian calibration and online data assimilation for modeling public opinion dynamics. The system addresses a fundamental limitation of LLM-agent simulations: their outputs are stochastic, structurally misspecified, and uncalibrated against empirical data. We formulate opinion dynamics as a force-based system with five competing mechanisms—direct LLM influence, social conformity, herd behavior, anchor rigidity, and exogenous shocks—combined through a gauge-fixed softmax mixture. A three-level hierarchical Bayesian model (global, domain, scenario) with explicit readout discrepancy is calibrated via stochastic variational inference (SVI) on 42 empirical scenarios spanning 10 domains. The calibrated model achieves 12.6 pp mean absolute error on verified held-out scenarios (19.2 pp on the full test set including one data-quality-flagged scenario) with 85.7% coverage of 90% credible intervals. Credible intervals are computed in logit space and back-transformed via sigmoid, guaranteeing bounds within [0, 100] by construction. Simulation-based calibration confirms that the generative model is well-specified under a gold-standard NUTS backend (6/6 parameters pass KS uniformity, all p > 0.20); a separate comparison shows the SVI approximation agrees with NUTS on the dominant parameters but underestimates uncertainty on weaker ones. Variance-based global sensitivity analysis identifies herd behavior (S_T = 0.55) and anchor rigidity (S_T = 0.45) as the dominant mechanisms, with their interaction (S_2 = 0.094) accounting for most nonlinear output variance—justifying the decision to freeze four mechanistic parameters. We extend the framework with an Ensemble Kalman Filter (EnKF) for online data assimilation, enabling live updating of both model parameters and agent states as streaming observations arrive. On the Brexit referendum scenario, the EnKF reduces prediction error to 1.8 pp with six polling observations—a 77% improvement over the last-available-poll baseline—while substantially outperforming polling baselines. We further develop a multi-modal calibration extension (v4) that jointly fits the model on polling outcomes and historical financial market returns, linking opinion dynamics to observed stock prices via sector-specific political beta coefficients. On 14 scenarios enriched with market data from Yahoo Finance, the multi-modal model improves test set coverage from 75.0% to 87.5% while maintaining comparable MAE (18.2 pp). However, the learned modality weight λ_fin = 0.045 indicates that financial returns are too noisy to substantially constrain the opinion dynamics parameters at the current dataset scale—a principled negative result that quantifies the information content of market signals for social simulation calibration. We discuss limitations including systematic over-prediction in financial domains (mean |b_s| = 0.74 in logit space), sensitivity of the frozen citizen step size parameter (up to 7.9 pp MAE variation under ±40% perturbation), and the irreducible gap between LLM-generated narratives and real-world opinion formation. **Version 2.5 adds a null-baseline benchmarking layer.** Four standard forecasters (naive persistence, running mean, OLS linear trend, AR(1)) are evaluated against the same 43 empirical trajectories using the Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction. On normalized support, naive persistence is a surprisingly strong baseline (mean RMSE = 0.038, i.e. 3.8 pp), and no single baseline reliably dominates it — OLS linear trend beats persistence significantly (p < 0.05) on only 4/43 scenarios for support and 6/43 for signed position. Domain-stratified skill is heterogeneous: political polling is the easiest to forecast (persistence RMSE = 0.012), while financial scenarios are the hardest (0.063), a 5× spread that any calibrated model must internalize. A residual-bootstrap coverage calculator provides a model-free complement to the logit-space credible intervals of v2.4. The full corpus covers 25/140 cells of a 7 (domain) × 5 (region) × 4 (tension) diversity matrix with no axis empty. The benchmarking package is released as a standalone module (`benchmarks/`) with 103 automated tests and a command-line runner, establishing a reproducible skill floor that any future calibrated sim must clear to justify its complexity.

---

## 1. Introduction

Agent-based models (ABMs) have long been used to study opinion dynamics, social influence, and collective decision-making. Classical approaches—bounded confidence models (Hegselmann & Krause, 2002), voter models (Holley & Liggett, 1975), and DeGroot-style averaging (DeGroot, 1974)—provide theoretical insight but struggle to capture the narrative complexity, heterogeneous reasoning, and contextual sensitivity that characterize real-world opinion formation.

The emergence of large language models (LLMs) as agent engines has opened a new approach: equipping simulated agents with natural language reasoning capabilities, allowing them to process news events, generate social media posts, form coalitions, and shift positions in response to contextual narratives. Recent work on generative agents (Park et al., 2023), social simulacra (Park et al., 2022), and LLM-driven social simulations (Gao et al., 2023) demonstrates the potential of this paradigm. However, these systems share a critical limitation: their outputs are uncalibrated. The mapping from LLM-generated agent behaviors to quantitative opinion distributions is ad hoc, and there is no principled mechanism to anchor simulations to empirical observations.

This paper addresses this gap by developing a complete calibration and assimilation pipeline for LLM-agent opinion dynamics. Our contributions are:

1. **A differentiable opinion dynamics simulator** formulated as a force-based system with five competing mechanisms, combined through gauge-fixed softmax mixing. The simulator is implemented in JAX and compatible with `jax.lax.scan`, enabling automatic differentiation through the full simulation trajectory.

2. **A hierarchical Bayesian calibration framework** with three levels (global, domain, scenario) and explicit readout discrepancy terms (b_d, b_s), fitted via stochastic variational inference (SVI) on 42 empirical scenarios across 10 domains.

3. **Simulation-based calibration (SBC)** and **variance-based global sensitivity analysis (Sobol indices)** providing rigorous validation of posterior quality and identification of dominant mechanisms.

4. **An Ensemble Kalman Filter (EnKF)** for online data assimilation, bridging the offline calibrated posterior to live streaming observations. The EnKF jointly updates model parameters and agent states, producing calibrated probabilistic forecasts at each round.

5. **A preliminary regime-switching extension** for crisis dynamics (Appendix B), using soft sigmoid-based switching between normal and crisis regimes to model discontinuous trust collapses while maintaining JAX differentiability.

6. **A multi-modal calibration extension** (Section 5.8) that jointly fits the hierarchical model on polling outcomes and historical financial market returns, connecting opinion dynamics to observable financial signals through a learnable linkage function with sector-specific political beta coefficients.

7. **A null-baseline predictive-skill benchmark** (Section 6.6) built on four standard forecasters, the Diebold–Mariano test with Harvey–Leybourne–Newbold small-sample correction, residual-bootstrap coverage (Section 6.7), and a seven-by-five-by-four scenario-diversity matrix (Section 6.8). The benchmark operates on the same 43 empirical trajectories used for calibration, provides pooled skill estimates by domain, region, and tension level, and is released as a reusable module (Appendix D) so any modification to the sim can be compared against a fixed reference.

The framework transforms LLM-agent simulations from narrative-generation tools into quantitative digital twins: calibrated probabilistic models that can be validated, updated with data, benchmarked against null baselines, and used for counterfactual analysis.

### 1.1 Paper Organization

Section 2 reviews related work. Section 3 presents the opinion dynamics model and its five force terms. Section 4 describes the hierarchical Bayesian calibration framework. Section 5 presents calibration results on 42 empirical scenarios, including a multi-modal extension that incorporates financial market data (Section 5.8). Section 6 covers validation through SBC, sensitivity analysis, robustness checks, and — new in v2.5 — predictive-skill benchmarking against null forecasters (Section 6.6), residual-bootstrap coverage (Section 6.7), and a scenario-diversity matrix (Section 6.8). Section 7 introduces the EnKF online assimilation module. Section 8 discusses limitations and future work. Section 9 concludes. Appendix A provides implementation details. Appendix B describes the preliminary regime-switching extension. Appendix C lists the full scenario dataset. Appendix D documents the reproducibility package for the v2.5 benchmarks.

---

## 2. Related Work

### 2.1 Classical Opinion Dynamics Models

The mathematical study of opinion formation has a rich history. DeGroot (1974) introduced iterative weighted averaging, where agents update beliefs as weighted means of their neighbors', converging to consensus under connectivity conditions. Hegselmann & Krause (2002) proposed the bounded confidence model, where agents interact only with those holding sufficiently similar opinions, producing clustering rather than global consensus. Deffuant et al. (2000) introduced a pairwise variant of bounded confidence with convergence at the dyadic level. These models provide foundational mechanisms but operate on homogeneous populations with idealized interaction rules.

DigitalTwinSim builds on the bounded confidence tradition but introduces three departures: (i) the tolerance threshold is smooth (sigmoid-based) rather than a hard cutoff, preserving differentiability; (ii) five distinct social mechanisms are combined through a learnable softmax mixture rather than studied in isolation; and (iii) agent heterogeneity (type, rigidity, tolerance) is derived from LLM-generated character profiles, providing scenario-specific structure rather than distributional assumptions.

### 2.2 Bayesian Calibration of Agent-Based Models

The systematic calibration of computer models against observational data was formalized by Kennedy & O'Hagan (2001), who introduced the framework of model discrepancy—an explicit term capturing the gap between the simulator output and reality, even at the best parameter settings. This framework provides the theoretical foundation for our readout discrepancy terms b_d and b_s.

Calibrating ABMs is notoriously difficult because the likelihood is typically intractable. Grazzini et al. (2017) developed Bayesian estimation methods for ABMs using indirect inference, matching simulated and empirical summary statistics. Platt (2020) provided a systematic comparison of ABM calibration methods—approximate Bayesian computation (ABC), synthetic likelihood, and sequential Monte Carlo—finding that method choice depends strongly on the model's dimensionality and summary statistic informativeness. Blei et al. (2017) reviewed variational inference as a scalable alternative to MCMC for complex probabilistic models.

DigitalTwinSim is, to our knowledge, the first system to combine Kennedy-O'Hagan-style discrepancy modeling with an LLM-agent ABM and variational inference. The differentiable JAX implementation enables direct gradient computation through the simulator, avoiding the summary-statistic bottleneck that plagues ABC-based approaches.

### 2.3 LLM-Agent Simulations

Park et al. (2023) introduced Generative Agents—LLM-powered characters that plan, reflect, and interact in a sandbox environment, demonstrating emergent social behaviors. Gao et al. (2023) surveyed the rapidly growing field of LLM-empowered ABM, cataloging applications from epidemic modeling to market simulation. Argyle et al. (2023) proposed "silicon sampling"—using LLMs as proxies for human survey respondents—and showed that GPT-3 can reproduce demographic opinion patterns on political topics with surprising fidelity. Horton (2023) introduced "homo silicus," demonstrating that LLMs can serve as simulated economic agents that replicate classic behavioral economics results.

These works establish that LLMs can generate plausible social behaviors, but none calibrate the emergent dynamics against empirical outcome data or provide uncertainty quantification. DigitalTwinSim fills this gap by treating LLM outputs as inputs to a mechanistic model that is then calibrated and validated against real-world observations.

### 2.4 Data Assimilation in Social Systems

The Ensemble Kalman Filter (Evensen, 2003) is the standard tool for sequential data assimilation in geosciences, combining model forecasts with observations in a computationally efficient ensemble framework. Reich & Cotter (2015) provide a comprehensive treatment of Bayesian data assimilation methods. In epidemiology, EnKF and related methods have been used to assimilate surveillance data into disease transmission models (Mistry et al., 2021).

Despite its success in physical and biological systems, data assimilation remains rare in computational social science. DigitalTwinSim applies the EnKF to opinion dynamics, using it to bridge offline Bayesian calibration with streaming polling observations—a novel application that enables real-time digital twin operation.

### 2.5 Sensitivity Analysis for ABMs

Saltelli et al. (2008) developed the theory and practice of variance-based global sensitivity analysis using Sobol indices, which decompose output variance into contributions from individual inputs and their interactions. Thiele et al. (2014) adapted these methods for ABMs, demonstrating that global sensitivity analysis (Sobol, FAST) is far more informative than one-at-a-time (OAT) perturbation for nonlinear models with parameter interactions.

Our sensitivity analysis uses the Sobol method with Saltelli sampling to justify the partition of 8 model parameters into 4 calibrable and 4 frozen, based on main effects (S_1) and total effects (S_T) including all interaction terms.

---

## 3. Opinion Dynamics Model

[FIGURE 1: System architecture diagram. Three horizontal layers. TOP LAYER — "LLM Agent Engine": box labeled "Gemini/GPT" with arrows producing "Δ^{LLM} per agent" and "Event narratives". MIDDLE LAYER — "Opinion Dynamics Simulator (JAX)": five boxes (Direct, Herd, Anchor, Social, Event) feeding into "Softmax π" then "Position Update p(t+1)", then "Readout q(t)". Label shows "jax.lax.scan over T rounds". BOTTOM LAYER — "Calibration & Assimilation": left box "Hierarchical Bayesian (SVI)" with arrow from "Observation Model (BetaBinom/Normal)" upward to readout. Right box "EnKF Online" with bidirectional arrow to "Streaming Observations (polls, sentiment)". Dashed arrow from SVI posterior to EnKF initialization. Annotations: "b_d + b_s" on the readout-to-observation connection. "θ_s = μ_d + Bx_s + ε_s" between hierarchy and simulator.]

### 3.1 Agent State Space

Each agent *i* ∈ {1, ..., n} maintains a scalar position p_i(t) ∈ [-1, +1] representing their stance on a binary issue, where +1 denotes maximum support ("Pro") and -1 maximum opposition ("Against"). Agents are characterized by three fixed attributes:

- **Type** τ_i ∈ {elite, citizen}: determines step size and behavioral parameters.
- **Rigidity** ρ_i ∈ [0, 1]: resistance to opinion change. Elites have higher rigidity (ρ ≈ 0.7) than citizens (ρ ≈ 0.3).
- **Tolerance** θ_i ∈ [0, 1]: radius of the bounded-confidence window for social influence. Elites have lower tolerance (θ ≈ 0.3) than citizens (θ ≈ 0.6).

Agents interact through a sparse weighted graph W ∈ ℝ^{n×n} constructed from LLM-assigned influence scores using a k-nearest-neighbor scheme (k = 5, matching the NumPy simulation's feed-based top-5 selection).

### 3.2 Five Force Terms

At each round t, five independent forces act on each agent. These forces capture distinct social mechanisms:

**Force 1: Direct LLM Influence.** The LLM generates per-agent opinion shifts Δ_i^{LLM}(t) based on the current narrative context. The direct force is:

$$f_i^{\text{direct}}(t) = \Delta_i^{\text{LLM}}(t) \cdot (1 - \rho_i) \cdot \max(0, 1 - |p_i(t)|) \quad \text{(Eq. 1)}$$

The susceptibility factor (1 - ρ_i) attenuates influence on rigid agents, while the boundary factor max(0, 1 - |p_i|) prevents extreme agents from being pushed further.

**Force 2: Herd Behavior (Consensus Pull).** Agents experience a pull toward the weighted mean of their neighborhood when the deviation exceeds a learned threshold θ_h:

$$f_i^{\text{herd}}(t) = (\bar{p}_i^{\text{feed}} - p_i) \cdot (1 - \rho_i) \cdot \sigma\left(\frac{|\bar{p}_i^{\text{feed}} - p_i| - \theta_h}{\tau_H}\right) \quad \text{(Eq. 2)}$$

where $\bar{p}_i^{\text{feed}}$ is the normalized weighted mean of neighbors' positions, θ_h = σ(logit_herd_threshold) is a learned threshold, and τ_H = 0.02 is a steepness parameter. The sigmoid activation creates a smooth transition: agents ignore small deviations but respond to large consensus gaps.

**Force 3: Anchor Rigidity.** A restorative force pulling agents toward their original position:

$$f_i^{\text{anchor}}(t) = \rho_i \cdot (p_i^{\text{orig}}(t) - p_i(t)) \quad \text{(Eq. 3)}$$

where $p_i^{\text{orig}}(t)$ drifts slowly toward the current position at rate δ_drift = σ(logit_anchor_drift), modeling gradual internalization of new positions.

**Force 4: Social Influence (Bounded Confidence).** Weighted averaging with tolerance-gated interaction:

$$f_i^{\text{social}}(t) = \frac{\sum_j w_{ij}(t) \cdot (p_j(t) - p_i(t))}{\sum_j w_{ij}(t)} \quad \text{(Eq. 4)}$$

where $w_{ij}(t) = W_{ij} \cdot \sigma\left(\frac{\theta_i - |p_j - p_i|}{\tau_{BC}}\right)$ and τ_{BC} = 0.02. This implements smooth bounded confidence: agents preferentially interact with like-minded peers, but the transition is differentiable rather than a hard cutoff.

**Force 5: Exogenous Event Shock.** External events apply a uniform directional push:

$$f_i^{\text{event}}(t) = m_t \cdot d_t \cdot (1 - \rho_i) \quad \text{(Eq. 5)}$$

where m_t ∈ [0, 1] is the event magnitude and d_t ∈ {-1, +1} its direction. Events are generated by the LLM based on the simulated scenario narrative.

### 3.3 Force Standardization

Raw forces have heterogeneous scales (e.g., social forces depend on graph density, event forces on shock magnitude). We standardize all five forces per round using exponential moving averages:

$$\tilde{f}_k(t) = \frac{f_k(t) - \mu_k(t)}{\sigma_k(t) + \epsilon} \quad \text{(Eq. 6)}$$

where μ_k(t) and σ_k(t) are updated via EMA with decay α = 0.3, computed across ALL agents in each round. The per-round statistics (mean and standard deviation) are symmetric functions over the agent set, ensuring that the standardization is permutation-invariant with respect to agent ordering. A permutation invariance test confirms this property empirically: max |Δpro_fraction| < 6 × 10⁻⁸ under random agent permutation. The temporal EMA smoothing provides stability across rounds without introducing dependence on agent array order.

### 3.4 Gauge-Fixed Softmax Mixing

The five standardized forces are combined through a softmax mixture with a gauge-fixing constraint:

$$\boldsymbol{\alpha} = [0, \alpha_h, \alpha_a, \alpha_s, \alpha_e]^\top \quad \text{(Eq. 7)}$$

$$\pi_k = \text{softmax}(\boldsymbol{\alpha})_k, \quad \sum_k \pi_k = 1 \quad \text{(Eq. 8)}$$

The direct force weight α_direct ≡ 0 serves as the reference level (gauge), resolving the shift invariance of softmax. This means the four remaining weights α_h, α_a, α_s, α_e are interpretable as log-odds relative to the direct LLM influence.

The combined force per agent is:

$$\tilde{f}_i^{\text{combined}}(t) = \sum_k \pi_k \cdot \tilde{f}_{k,i}(t) \quad \text{(Eq. 9)}$$

### 3.5 Position Update

The position update applies a type-specific step size with smooth clamping:

$$\Delta p_i = \lambda_{\tau_i} \cdot \tilde{f}_i^{\text{combined}}, \quad \Delta p_i^{\text{clamped}} = \tanh\left(\frac{\Delta p_i}{c_{\tau_i}}\right) \cdot c_{\tau_i} \quad \text{(Eq. 10)}$$

where λ_elite = exp(log_λ_elite), λ_citizen = exp(log_λ_citizen), c_elite = 0.15, c_citizen = 0.25. The tanh clamping provides smooth saturation rather than hard clipping, preserving differentiability.

The final update is:

$$p_i(t+1) = \text{clip}(p_i(t) + \Delta p_i^{\text{clamped}}, -1, +1) \quad \text{(Eq. 11)}$$

### 3.6 Readout Function

The mapping from agent positions to aggregate opinion uses a soft classification:

$$\text{pro}(t) = \frac{\sum_i \sigma\left(\frac{p_i(t) - 0.05}{0.02}\right)}{\sum_i \left[\sigma\left(\frac{p_i(t) - 0.05}{0.02}\right) + \sigma\left(\frac{-p_i(t) - 0.05}{0.02}\right)\right]} \quad \text{(Eq. 12)}$$

This readout excludes near-neutral agents (|p_i| < 0.05) from the decided count, producing pro_fraction ∈ [0, 1]. The 0.02 temperature ensures smooth but sharp classification.

### 3.7 Parameter Summary

The model has 8 mechanistic parameters, partitioned into 4 calibrable and 4 frozen:

**Table 1: Model Parameters**

| Parameter | Symbol | Role | Type | Default |
|---|---|---|---|---|
| Herd weight | α_h | Consensus pull strength (log-odds vs direct) | Calibrable | — |
| Anchor weight | α_a | Rigidity pull strength | Calibrable | — |
| Social weight | α_s | Bounded-confidence averaging strength | Calibrable | — |
| Event weight | α_e | Exogenous shock strength | Calibrable | — |
| Elite step size | log λ_elite | Step magnitude for elite agents | Frozen | -1.2 |
| Citizen step size | log λ_citizen | Step magnitude for citizen agents | Frozen | -0.5 |
| Herd threshold | logit θ_h | Minimum deviation to trigger herd behavior | Frozen | 0.5 |
| Anchor drift | logit δ_drift | Rate of anchor position drift | Frozen | -1.4 |

The calibrable/frozen partition is justified empirically by Sobol sensitivity analysis (Section 6.2).

---

## 4. Hierarchical Bayesian Calibration

### 4.1 Motivation

A single set of global parameters cannot adequately describe opinion dynamics across diverse domains. Political referenda involve different social mechanisms than corporate crises or public health debates. At the same time, per-scenario calibration with 4 parameters and limited per-scenario data (typically 3–9 polling observations) leads to severe overfitting. A hierarchical model enables partial pooling: scenarios share information through domain and global priors while retaining scenario-specific flexibility through model discrepancy terms.

### 4.2 Three-Level Hierarchy

The model has three levels:

**Level 1 (Global).** Shared priors on the mixing weights:

$$\mu_{\text{global}} \sim \mathcal{N}(\mathbf{0}, I_4), \quad \sigma_{\text{global}} \sim \text{HalfNormal}(0.3) \quad \text{(Eq. 13)}$$

where μ_global ∈ ℝ⁴ represents the global mean of the four calibrable alpha parameters.

**Level 2 (Domain).** Each domain d ∈ {political, financial, corporate, ...} has domain-level parameter means:

$$\mu_d \sim \mathcal{N}(\mu_{\text{global}}, \text{diag}(\sigma_{\text{global}}^2)) \quad \text{(Eq. 14)}$$

where μ_d ∈ ℝ⁴ are the domain-level alpha parameters.

**Level 3 (Scenario).** Each scenario s has scenario-specific parameters drawn from its domain:

$$\theta_s \sim \mathcal{N}(\mu_{d(s)} + B \cdot x_s, \text{diag}(\sigma_d^2)) \quad \text{(Eq. 15)}$$

where θ_s ∈ ℝ⁴ are the four softmax weights used for simulation, σ_d ∈ ℝ⁴ is a domain-specific dispersion (not the global σ_global), and B · x_s is the covariate regression defined in Section 4.6. The domain-level variance σ_d allows different domains to have different degrees of scenario-to-scenario heterogeneity.

### 4.3 Readout Discrepancy

The simulator is structurally misspecified: LLM-generated narratives cannot perfectly reproduce real-world opinion formation. Rather than absorbing this misspecification into the parameters (which would bias them), we model it explicitly through additive correction in logit space on the readout.

The discrepancy operates on the scalar readout (ℝ¹), not on the parameter space (ℝ⁴):

$$b_d \sim \mathcal{N}(0, \sigma_{b,\text{between}}^2), \quad \sigma_{b,\text{between}} \sim \text{HalfNormal}(0.2) \quad \text{(Eq. 16)}$$

$$b_s \sim \mathcal{N}(0, \sigma_{b,\text{within}}^2), \quad \sigma_{b,\text{within}} \sim \text{HalfNormal}(0.5) \quad \text{(Eq. 17)}$$

where b_d ∈ ℝ is the domain-level readout bias (between-domain discrepancy) and b_s ∈ ℝ is the scenario-specific readout discrepancy (within-domain). The corrected readout for scenario s in domain d is:

$$q_s^{\text{corrected}} = \sigma\left(\text{logit}(q_s^{\text{sim}}) + b_d + b_s\right) \quad \text{(Eq. 18)}$$

where $q_s^{\text{sim}}$ is the raw simulator output. By operating in logit space, the correction is unbounded and additive while the sigmoid guarantees the result remains in [0, 1]. This decomposition follows Kennedy & O'Hagan (2001): the calibrated parameters θ represent genuine opinion dynamics mechanisms, while the discrepancy terms b_d and b_s absorb systematic simulator errors.

### 4.4 Observation Model

For scenarios with per-round polling data (sample size n_t, observed count y_t):

$$y_t \mid q_t, \phi \sim \text{BetaBinomial}(n_t, q_t \cdot \phi, (1 - q_t) \cdot \phi) \quad \text{(Eq. 19)}$$

where φ = exp(log φ) is a learned concentration parameter. The BetaBinomial accounts for both sampling noise (binomial) and overdispersion (beta mixing).

For scenarios with only a final outcome:

$$y_{\text{final}} \mid q_T, \sigma_{\text{obs}} \sim \mathcal{N}(100 \cdot q_T, \sigma_{\text{obs}}^2) \quad \text{(Eq. 20)}$$

where σ_obs = exp(log σ) is learned.

### 4.5 Inference via SVI

The posterior is intractable due to the nonlinear JAX simulator in the likelihood. We use stochastic variational inference (SVI) with an AutoLowRankMultivariateNormal guide in NumPyro, which approximates the posterior with a low-rank plus diagonal covariance structure:

$$q(\theta) = \mathcal{N}(\mu_q, D + VV^\top) \quad \text{(Eq. 21)}$$

where D is diagonal and V has rank ≤ dim(θ). This captures dominant posterior correlations (e.g., between α_h and α_a) while scaling to the full parameter space.

The SVI objective is the evidence lower bound (ELBO):

$$\mathcal{L}(\mu_q, D, V) = \mathbb{E}_{q(\theta)}[\log p(y \mid \theta) + \log p(\theta)] - \mathbb{E}_{q(\theta)}[\log q(\theta)] \quad \text{(Eq. 22)}$$

minimized over 3000 steps with learning rate 0.002 and cosine annealing. Gradients are computed via JAX automatic differentiation through the full simulate → readout → likelihood chain.

### 4.6 Covariate Regression

The covariate regression is incorporated directly into the scenario-level prior (Eq. 15, Section 4.2). The matrix B ∈ ℝ^{4×5} and covariates x_s ∈ ℝ⁵ are defined as follows:

1. **initial_polarization** — initial spread of agent positions
2. **event_volatility** — intensity and frequency of external shocks
3. **elite_concentration** — power centralization among top agents
4. **institutional_trust** — public confidence in institutions (0–1)
5. **undecided_share** — initial proportion of near-neutral agents

The prior on B is informative but weakly regularizing: B_{ij} ~ N(0, 0.3), allowing the data to determine whether covariates have predictive power. A covariate effect is considered significant if its 95% CI excludes zero.

---

## 5. Calibration Results

### 5.1 Dataset

We curated 42 empirical scenarios across 10 domains, each with a ground-truth final outcome and, where available, intermediate polling data. The scenarios span 2012–2023 and cover political referenda, financial crises, corporate controversies, public health debates, technology policy, and social movements.

**Table 2: Dataset Composition by Domain**

| Domain | N (Train) | N (Test) | Example Scenarios |
|---|---|---|---|
| Political | 11 | 3 | Brexit, Scottish Independence, Chilean Constitution |
| Corporate | 6 | 1 | Dieselgate, Boeing 737 MAX, Facebook→Meta |
| Financial | 6 | 1 | FTX collapse, GameStop squeeze, SVB collapse |
| Public Health | 4 | 1 | COVID vaccination (IT), AstraZeneca controversy |
| Technology | 2 | 1 | Net Neutrality repeal |
| Commercial | 1 | 1 | Tesla Cybertruck reveal |
| Energy | 1 | 0 | Japanese nuclear phase-out |
| Environmental | 1 | 0 | Keystone XL pipeline |
| Labor | 1 | 0 | Uber vs. taxi regulation |
| Social | 1 | 0 | Marriage equality |
| **Total** | **34** | **8** | |

The train/test split is stratified by domain with approximately 80/20 ratio.

Scenarios flagged as NEEDS_VERIFICATION during the review phase (prior to calibration) are excluded from the primary performance metrics. This flag is assigned based on data quality criteria—missing primary sources, interpolated polling without verification, ambiguous ground-truth encoding—not based on model performance. One test scenario (Archegos Capital Collapse, quality_score = 67) carries this flag. We report both verified-only and full-set metrics throughout.

### 5.2 Calibrated Global Parameters

SVI converges in 3000 steps (final ELBO loss: 514.7) with a total runtime of 40.2 minutes for Phase B (empirical fine-tuning) and 0.4 minutes for Phase C (validation).

**Table 3: Calibrated Global Posterior**

| Parameter | Mean | 95% CI | Interpretation |
|---|---|---|---|
| α_herd | -0.176 | [-0.265, -0.079] | Herd effect slightly below direct influence |
| α_anchor | +0.297 | [+0.199, +0.401] | Anchor rigidity is the strongest mechanism |
| α_social | -0.105 | [-0.202, -0.005] | Social influence slightly below direct |
| α_event | -0.130 | [-0.227, -0.033] | Event shocks below direct influence |
| σ_global | [0.135, 0.147, 0.144, 0.143] | — | Inter-scenario parameter spread |

The positive α_anchor indicates that anchor rigidity (resistance to change) dominates the force mixture. This is consistent with the well-documented status quo bias in opinion dynamics: people tend to revert toward their initial positions after temporary shifts. The negative values for α_herd, α_social, and α_event indicate these forces are present but weaker than direct LLM influence (the gauge reference).

Converting to softmax weights: π = softmax([0, -0.176, 0.297, -0.105, -0.130]) ≈ [0.201, 0.169, 0.271, 0.181, 0.177]. Anchor rigidity receives approximately 27.1% of the total weight, direct influence 20.1%, social conformity 18.1%, event shocks 17.7%, and herd behavior 16.9%.

### 5.3 Readout Discrepancy

The calibrated discrepancy parameters reveal the scale of structural model misspecification:

$$\sigma_{b,\text{between}} = 0.115 \quad [0.073, 0.173] \quad \text{(Eq. 23)}$$
$$\sigma_{b,\text{within}} = 0.558 \quad [0.436, 0.696] \quad \text{(Eq. 24)}$$

The within-scenario discrepancy (σ_{b,within} = 0.558) is approximately 5× the between-domain discrepancy (σ_{b,between} = 0.115), indicating that scenario-specific factors dominate over systematic domain-level biases. In practical terms, 0.558 in logit space translates to approximately 12–14 percentage points of systematic prediction error that the model cannot eliminate through parameter adjustment alone.

**Table 4: Domain-Level Readout Discrepancy**

| Domain | N | b_d Mean | 95% CI | Mean |b_s| | Bias |
|---|---|---|---|---|---|
| Financial | 6 | -0.054 | [-0.242, +0.126] | 0.744 | Over-predicts |
| Energy | 1 | -0.026 | [-0.237, +0.186] | 0.709 | Over-predicts |
| Public Health | 4 | -0.032 | [-0.223, +0.164] | 0.534 | Over-predicts |
| Corporate | 6 | -0.040 | [-0.221, +0.139] | 0.432 | Over-predicts |
| Political | 11 | +0.019 | [-0.145, +0.183] | 0.198 | No systematic bias |
| Technology | 2 | -0.021 | [-0.204, +0.174] | 0.066 | No systematic bias |

Financial scenarios exhibit the largest mean absolute discrepancy (|b_s| = 0.744), driven by cases where the model systematically over-predicts support levels. The political domain, with the most training data (N = 11), shows the lowest systematic bias (b_d ≈ 0, mean |b_s| = 0.198).

### 5.4 Predictive Performance

**Table 5: Validation Metrics (Test Set)**

Credible intervals are computed in logit space and back-transformed via sigmoid, guaranteeing [0, 100] bounds by construction.

| Metric | Full Test (N=8) | Verified Only (N=7) |
|---|---|---|
| MAE | 19.2 pp | 12.6 pp |
| RMSE | 26.6 | 14.3 |
| Median AE | 11.9 pp | 9.2 pp |
| Coverage 90% | 75.0% | 85.7% |
| Coverage 50% | 37.5% | 42.9% |
| CRPS | 15.4 | 8.6 |

The primary metric is verified-only MAE of 12.6 pp, excluding the Archegos scenario (error = 65.0 pp) whose ground truth is flagged as unreliable. The 85.7% coverage of 90% credible intervals indicates well-calibrated uncertainty—slightly below the nominal 90% but within the expected range for N = 7.

**Table 6: Test Set Detailed Results**

| Scenario | Domain | GT (%) | Predicted μ ± σ | Error (pp) | 90% CI | Covered |
|---|---|---|---|---|---|---|
| Archegos Capital*† | Financial | 35.0 | >99.9 (logit: 4.6 ± 0.5) | +65.0 | [99.2, 100.0] | ✗ |
| Greek Bailout | Political | 38.7 | 65.3 ± 11.7 | +26.6 | [42.7, 84.5] | ✗ |
| Net Neutrality | Technology | 83.0 | 66.3 ± 12.6 | -16.7 | [47.3, 82.8] | ✓ |
| French Election | Political | 66.1 | 51.6 ± 13.7 | -14.5 | [29.7, 72.8] | ✓ |
| COVID Vax (IT) | Pub. Health | 80.0 | 70.8 ± 11.2 | -9.2 | [52.1, 88.7] | ✓ |
| Tesla Cybertruck | Commercial | 62.0 | 54.1 ± 14.7 | -7.9 | [26.1, 72.6] | ✓ |
| Amazon HQ2 | Corporate | 56.0 | 63.3 ± 12.9 | +7.3 | [48.7, 80.6] | ✓ |
| Turkish Ref. | Political | 51.4 | 57.5 ± 13.0 | +6.1 | [37.4, 75.7] | ✓ |

*NEEDS_VERIFICATION — excluded from primary metrics. CI computed in logit space and back-transformed.

† The Archegos posterior predictive in logit space has mean ≈ 4.6 (corresponding to > 99% after sigmoid) with standard deviation ≈ 0.5, yielding CI [99.2, 100.0] in percentage space. The logit-space representation is more informative near the bounds: logit 4.6 ± 1.65 × 0.5 spans [3.78, 5.43], mapping to [97.7%, 99.6%] via sigmoid. This is a genuine model prediction—the simulator assigns near-certainty to high support—not a numerical artifact. The 65 pp error reflects fundamental model misspecification on this scenario, not a confidence interval failure. The simulator's LLM-generated trajectory for Archegos produces a consensus narrative that does not match the complex, fragmented real-world dynamics of a leveraged fund collapse.

### 5.5 Worst-Case Scenarios

**Table 7: Top 5 Scenario Discrepancies (Training Set)**

| Scenario | Domain | b_s | Error (pp) | Failure Mode |
|---|---|---|---|---|
| WeWork IPO | Financial | -1.378 | +48.5 | Trust collapse not modeled |
| United Airlines | Corporate | -0.860 | +15.4 | Viral outrage underestimated |
| SVB Collapse | Financial | -0.835 | +38.1 | Sequential shocks mishandled |
| Chile Constitution | Political | +0.816 | -41.5 | Landslide momentum missing |
| FTX Crypto | Financial | -0.727 | +11.0 | Contagion dynamics absent |

The three largest discrepancies are all financial crises where the model over-predicts support (negative b_s). This systematic failure motivates the regime-switching extension (Appendix B).

### 5.6 Evolution from v1 to v2

**Table 8: v1 vs v2 Comparison**

| Aspect | v1 (Gerli 2025) | v2 (this paper) |
|---|---|---|
| Calibration method | Grid search (972 combos) | Hierarchical Bayesian SVI |
| Training data | 1000 LLM-generated scenarios | 42 empirical scenarios |
| Parameters | 5 (all ad hoc) | 4 calibrable + 4 frozen (Sobol-justified) |
| Uncertainty quantification | None | 90% CI, coverage 85.7% |
| Model discrepancy | None | Explicit b_d + b_s |
| Validation | Same data as training | Held-out test set + SBC |
| Online assimilation | None | EnKF (error 1.8 pp with polling) |
| MAE (test) | 14.8 pp (synthetic test set)* | 12.6 pp (empirical test set)* |
| Epistemological status | Tool demonstration | Calibrated probabilistic model |

*Direct comparison between v1 and v2 MAE is not straightforward: v1 was evaluated on LLM-generated synthetic scenarios while v2 uses empirical scenarios with real-world ground truth. The v1 test set likely understates true error because synthetic ground truth inherits the same LLM biases as the simulation (epistemic circularity). The improvement from v1 to v2 reflects both better calibration and the elimination of this circularity.

**Table 8b: v2 vs v4 (Multi-Modal) Comparison**

| Aspect | v2 (polling-only) | v4 (multi-modal) |
|---|---|---|
| Observation modalities | Polling only | Polling + financial returns |
| Additional parameters | — | w_opinion, w_event, w_polar, λ_fin, σ_market (+5) |
| Scenarios with market data | 0 | 14 |
| Test MAE | 19.2 pp | 18.2 pp |
| Test Coverage 90% | 75.0% | 87.5% |
| Financial domain MAE | n/a | 24.0 pp |
| Key finding | — | λ_fin = 0.045 (market signal nearly suppressed) |

### 5.7 Covariate Effects

Of the 20 entries in the B matrix (4 parameters × 5 covariates), one effect is statistically significant at the 95% level. With 34 training scenarios and 20 coefficients in the B matrix, statistical power is limited. The following results should be interpreted as exploratory signals consistent with substantive theory, not as confirmed causal effects. No correction for multiple comparisons is applied; with a Bonferroni adjustment at 20 tests, the α_event × institutional_trust effect would require p < 0.0025 to reach significance, which it does not achieve at the current sample size.

**Table 9: Significant Covariate Effects**

| Parameter | Covariate | Effect (B_{ij}) | 95% CI | Interpretation |
|---|---|---|---|---|
| α_event | institutional_trust | -0.127 | [-0.242, -0.015] | Higher institutional trust reduces sensitivity to event shocks |

This effect has a natural interpretation: in scenarios with high institutional trust (e.g., established democracies with strong rule of law), external shocks are absorbed by institutional credibility rather than amplified through the opinion network. Conversely, in low-trust environments (financial crises, weak governance), event shocks propagate more strongly.

The remaining 19 covariate effects have 95% CIs that include zero, indicating that the current dataset (N = 34 training scenarios) lacks power to detect additional covariate relationships. The two nearest-to-significant effects are α_anchor × undecided_share (B = +0.116, CI [-0.010, 0.243]) and α_herd × initial_polarization (B = +0.096, CI [-0.013, 0.217]). Both are substantively plausible—more undecided agents may amplify anchoring effects, and higher initial polarization may strengthen herd dynamics—but require a larger dataset to confirm.

### 5.8 Multi-Modal Calibration: Polling + Financial Markets

#### 5.8.1 Motivation

The base calibration (Sections 5.1–5.7) uses a single observation modality: final polling outcomes. However, opinion dynamics in financial and corporate scenarios produce observable signals in equity markets—stock prices, trading volumes, sector ETF returns—that reflect investor-aggregated beliefs about the same underlying events. If these market signals are informative, incorporating them as an auxiliary likelihood should improve calibration, particularly in the financial domain where the polling-only model performs worst (mean |b_s| = 0.744, Table 4).

We develop a multi-modal extension (v4) that jointly maximizes a weighted combination of the polling likelihood (Section 4) and a new financial market likelihood, connected through a learnable linkage function.

#### 5.8.2 Financial Observation Model

For each scenario with available market data, we fetch historical prices from Yahoo Finance for a primary ticker and the S&P 500 benchmark (SPY). We compute per-round excess returns (primary minus benchmark) to isolate the scenario-specific signal from broad market movements.

The linkage function maps three simulator outputs to expected per-round market returns:

$$\mathbb{E}[r_t] = \beta_{\text{sector}} \cdot \left( w_{\text{opinion}} \cdot \Delta q(t) \cdot s_o + w_{\text{event}} \cdot m_t d_t \cdot s_e + w_{\text{polar}} \cdot \Delta \sigma_p(t) \cdot s_p \right) \quad \text{(Eq. 22b)}$$

where Δq(t) is the per-round change in pro-fraction, m_t d_t is the signed event shock, Δσ_p(t) is the change in position standard deviation (a polarization proxy), and s_o = 10, s_e = 5, s_p = 8 are fixed scale factors that map opinion-space units to percentage-point market returns. The sector-specific political beta β_sector captures the sensitivity of each industry to political/opinion events:

**Table 5b: Political Beta Coefficients by Sector**

| Sector | β_sector | Example Scenarios |
|---|---|---|
| Sovereign Debt | 2.20 | — |
| Banking | 1.85 | Archegos, SVB, GameStop, AMC |
| Insurance | 1.45 | — |
| Automotive | 1.30 | Dieselgate, Tesla Stock Split |
| Real Estate | 1.20 | WeWork IPO |
| Telecom | 1.15 | — |
| Energy (fossil) | 1.10 | — |
| Defense | 0.90 | Boeing 737 MAX |
| Tech | 0.85 | FTX (COIN proxy), Amazon HQ2, Twitter/X, Facebook→Meta |
| Infrastructure | 0.80 | United Airlines |

The financial likelihood for one scenario is:

$$\log p(\mathbf{r} | \theta) = \sum_{t \in \mathcal{M}} \left[ -\frac{1}{2}\log(2\pi\sigma_m^2) - \frac{(r_t - \mathbb{E}[r_t])^2}{2\sigma_m^2} \right] \quad \text{(Eq. 22c)}$$

where $\mathcal{M}$ is the set of rounds with valid market data and $\sigma_m = \exp(\log\sigma_m)$ is the market noise scale.

#### 5.8.3 Market Data Enrichment

We enriched 14 of the 42 scenarios with historical market data spanning all financial (FIN-*) and corporate (CORP-*) scenarios. Ticker selection follows a proxy rule: for delisted securities (Credit Suisse → XLF Financial ETF; Twitter → XLK Technology ETF), we use the closest sector ETF. For pre-IPO companies (Uber, WeWork), we use small-cap or growth ETFs.

**Table 5c: Enriched Scenarios with Market Data**

| Scenario | Ticker | Sector | Cumulative Return (%) | Excess vs SPY (%) |
|---|---|---|---|---|
| FIN-2021-GAMESTOP | GME | banking | +1784.9 | +1774.8 |
| FIN-2021-AMC | AMC | banking | +432.1 | +421.7 |
| FIN-2020-TESLA_STOCK_SPLIT | TSLA | automotive | +81.4 | +64.3 |
| CORP-2021-FACEBOOK→META | META | tech | -4.3 | -13.0 |
| FIN-2023-SVB_COLLAPSE | KRE | banking | -28.4 | -31.9 |
| FIN-2021-ARCHEGOS | XLF | banking | +8.7 | +0.5 |
| CORP-2019-BOEING_MAX | BA | defense | -24.8 | -28.7 |
| FIN-2022-FTX_CRYPTO | COIN | tech | -64.7 | -57.0 |
| CORP-2015-DIESELGATE | VOW3.DE | automotive | -35.2 | -37.1 |
| FIN-2019-WEWORK | IWO | real_estate | +5.1 | +2.3 |
| CORP-2017-UNITED_AIRLINES | UAL | infrastructure | -3.1 | -5.2 |
| CORP-2017-UBER_LONDON | ^GSPC | tech | +2.8 | 0.0 |
| CORP-2018-AMAZON_HQ2 | AMZN | tech | -12.4 | -18.7 |
| CORP-2022-TWITTER_X | XLK | tech | +3.2 | +0.8 |

#### 5.8.4 Joint Multi-Modal Likelihood

The v4 model extends the v2 hierarchical structure with five additional parameters:

- **Linkage weights** w_opinion, w_event, w_polar ~ Normal(0, 2): connect simulator outputs to expected returns.
- **Market noise** log σ_market ~ Normal(log 5, 1): scale of return prediction residuals.
- **Modality weight** λ_fin ~ Beta(2, 5): controls the relative influence of financial vs. polling likelihood.

The joint log-likelihood for scenario s is:

$$\ell_s = \ell_s^{\text{poll}} + \lambda_{\text{fin}} \cdot \ell_s^{\text{fin}} \cdot \mathbf{1}[\text{has\_fin}_s] \quad \text{(Eq. 22d)}$$

The Beta(2, 5) prior on λ_fin has mean 0.286, expressing a soft preference for the polling signal while allowing the data to upweight financial information if it proves informative.

#### 5.8.5 Results

SVI converges in 2500 steps (final loss: 574.7, runtime: 54.5 minutes).

**Table 5d: v2 (Polling-Only) vs v4 (Multi-Modal) Comparison**

| Metric | v2 Train | v2 Test | v4 Train | v4 Test | v4 Financial |
|---|---|---|---|---|---|
| MAE (pp) | 14.3 | 19.2 | 13.8 | 18.2 | 24.0 |
| RMSE (pp) | 18.8 | 26.6 | 18.9 | 25.6 | 30.7 |
| Coverage 90% | 79.4% | 75.0% | 76% | 87.5% | 57% |

The headline result is a **12.5 percentage point improvement in test set coverage** (75.0% → 87.5%), meaning 7 of 8 test scenarios are now covered by the 90% credible interval compared to 6 of 8 previously. Test MAE improves modestly (19.2 → 18.2 pp). Train metrics are comparable.

**Table 5e: Calibrated Financial Linkage Parameters**

| Parameter | Mean | 95% CI | Interpretation |
|---|---|---|---|
| w_opinion | +0.917 | [-1.106, +2.971] | Opinion shifts drive positive returns (wide CI) |
| w_event | -0.431 | [-2.092, +1.343] | Shocks have ambiguous market effect (CI spans 0) |
| w_polar | -0.662 | [-2.398, +1.153] | Polarization increase → negative returns (CI spans 0) |
| λ_fin | 0.045 | [0.037, 0.055] | **Financial signal weighted at only 4.5%** |
| σ_market | 13.0 | [11.1, 15.5] | Market noise floor: ~13 percentage points |

**Key finding: the model learned to nearly suppress the financial likelihood.** The posterior λ_fin = 0.045 [0.037, 0.055] is far below the prior mean of 0.286, indicating that the data strongly prefer the polling-only signal. All three linkage weight CIs span zero, meaning none of the opinion→market channels are individually identifiable at the current sample size. The high σ_market = 13% confirms that per-round equity returns are dominated by idiosyncratic noise unrelated to the opinion dynamics the model captures.

This is a principled negative result: the Bayesian model was free to upweight the financial channel if it improved the joint likelihood, and it chose not to. The financial domain scenarios remain the hardest (MAE = 24.0 pp, coverage = 57%), with the same outliers: Archegos (+64.7 pp), WeWork (+55.7 pp), SVB (+37.2 pp).

#### 5.8.6 Why Financial Returns Are Uninformative for Opinion Calibration

Three factors explain the low λ_fin:

1. **Signal-to-noise ratio.** Per-round equity returns have standard deviations of 10–30%, while the opinion dynamics signals (Δq, shock, Δσ_p) produce expected returns of 1–5% through the linkage function. The SNR is approximately 0.1–0.3, insufficient for meaningful parameter constraint.

2. **Confounded channel.** Market returns reflect many factors beyond public opinion—earnings, macro, sector rotation, liquidity—that the linkage function cannot capture. The "excess return" (primary minus benchmark) partially deconfounds, but sector-specific factors remain uncontrolled.

3. **Sample size.** With 14 enriched scenarios and ~7 rounds each (~98 return observations), the effective sample for learning 3 linkage weights plus σ_market is small, especially given the high noise.

#### 5.8.7 Implications

Despite the negative financial result, the v4 model demonstrates two valuable findings:

1. **Improved coverage.** The 75% → 87.5% test coverage improvement likely arises from the additional regularization imposed by the multi-modal structure—even a weakly informative auxiliary likelihood can improve uncertainty calibration by preventing the variational posterior from over-concentrating.

2. **Quantified information boundary.** The posterior λ_fin provides a rigorous, data-driven answer to the question "how much can market data help calibrate opinion models?" At the current scale: very little (4.5%). This sets a clear target for future work: either increase the number of enriched scenarios (from 14 to 40+), use higher-frequency data (daily rather than per-round), or incorporate richer market features (implied volatility, options skew, CDS spreads) to push the SNR above the identifiability threshold.

---

## 6. Validation

### 6.1 Simulation-Based Calibration

Simulation-based calibration (SBC; Talts et al., 2018) verifies that the inference procedure recovers known parameters. We generate 100 synthetic datasets from the prior, run NUTS inference (200 warmup + 200 samples) on each, and test whether the posterior rank statistics are uniformly distributed.

**Table 10: SBC Results**

| Parameter | N | KS Statistic | p-value | Verdict |
|---|---|---|---|---|
| α_herd | 100 | 0.070 | 0.685 | PASS |
| α_anchor | 100 | 0.095 | 0.308 | PASS |
| α_social | 100 | 0.075 | 0.600 | PASS |
| α_event | 100 | 0.075 | 0.600 | PASS |
| τ_readout | 100 | 0.095 | 0.308 | PASS |
| σ_obs | 100 | 0.105 | 0.205 | PASS |

SBC confirms that the generative model is well-specified and that a gold-standard NUTS backend recovers correct posteriors: all 6 parameters pass the KS uniformity test (p > 0.20). However, SBC does not validate the SVI approximation used for the full empirical model, since SBC runs use NUTS.

[FIGURE 2: Six-panel grid of histograms (2×3 layout). Each panel shows rank histogram for one SBC parameter (α_herd, α_anchor, α_social, α_event, τ_readout, σ_obs). X-axis: rank bin (1–10). Y-axis: count (0–15). Horizontal dashed line at expected count = 10 (uniform). Each panel titled with parameter name and KS p-value. All histograms should appear roughly uniform — no U-shape or tent shape.]

The SBC configuration uses 10 agents and 7 rounds per synthetic scenario with Normal(0, 0.3) priors. Total runtime: 297.6 seconds (approximately 3 seconds per instance).

### 6.2 SVI vs NUTS Comparison

To assess variational approximation quality, we compare SVI and NUTS posteriors on a 4-scenario subset spanning two domains.

**Table 11: SVI vs NUTS Posterior Comparison (4-scenario subset)**

| Parameter | SVI mean ± std | NUTS mean ± std | |Δμ|/σ_NUTS | Compatible |
|---|---|---|---|---|
| α_herd | -0.500 ± 0.085 | -0.383 ± 0.796 | 0.148 | Yes |
| α_anchor | +0.905 ± 0.065 | +0.604 ± 0.856 | 0.352 | Yes |
| α_social | -0.997 ± 0.105 | -0.350 ± 0.763 | 0.848 | No |
| α_event | -1.752 ± 0.073 | -0.137 ± 0.946 | 1.707 | No |

*SVI: 1000 steps, Adam(lr = 0.01), AutoLowRankMVN guide. NUTS: 200 warmup + 200 samples, max_tree_depth = 8.*

The two dominant parameters show good agreement (α_herd: |Δμ|/σ_NUTS = 0.15; α_anchor: 0.35). The weaker parameters show larger divergence (α_social: 0.85; α_event: 1.71), consistent with the known tendency of structured variational families with restricted covariance to concentrate posterior mass. The SVI standard deviations are 5–13× narrower than the NUTS estimates, indicating that the variational approximation substantially underestimates posterior uncertainty on the less-constrained parameters.

This result has two implications. First, the posterior means for α_herd and α_anchor—which together dominate the force mixture (S_T = 0.55 and 0.45)—are reliably estimated by SVI. Second, uncertainty estimates on α_social and α_event may be underestimated by the variational approximation. Users requiring conservative uncertainty bounds should consider post-hoc inflation of the variational posterior or short NUTS chains initialized from the SVI optimum.

**Limitation.** This comparison uses a 4-scenario subset for computational tractability; a full 42-scenario NUTS run would strengthen the comparison but requires substantially more compute.

### 6.3 Global Sensitivity Analysis

Variance-based global sensitivity analysis (Sobol, 2001) decomposes output variance into contributions from individual parameters and their interactions. We evaluate all 8 model parameters using N = 1024 Saltelli samples (18,432 total simulator evaluations, n = 30 agents, 7 rounds).

**Table 12: Sobol Sensitivity Indices**

| Parameter | S_1 (Main Effect) | S_T (Total Effect) | Type |
|---|---|---|---|
| α_herd | 0.364 ± 0.059 | 0.555 ± 0.055 | Calibrable |
| α_anchor | 0.207 ± 0.060 | 0.452 ± 0.047 | Calibrable |
| α_social | 0.086 ± 0.040 | 0.213 ± 0.034 | Calibrable |
| α_event | 0.026 ± 0.033 | 0.115 ± 0.023 | Calibrable |
| λ_citizen | 0.002 ± 0.033 | 0.121 ± 0.025 | Frozen |
| λ_elite | 0.007 ± 0.010 | 0.016 ± 0.006 | Frozen |
| θ_herd | 0.003 ± 0.013 | 0.024 ± 0.006 | Frozen |
| δ_drift | 0.003 ± 0.010 | 0.013 ± 0.003 | Frozen |

**Key findings:**

1. **Dominant pair.** α_herd (S_T = 0.555) and α_anchor (S_T = 0.452) together account for the vast majority of output variance, including through their interaction. Their second-order interaction index S_2(α_herd, α_anchor) = 0.094 is the largest pairwise interaction, indicating strong nonlinear coupling between herd behavior and anchor rigidity.

2. **Calibrable/frozen partition.** Three of the four frozen parameters (λ_elite, θ_herd, δ_drift) have both S_1 < 0.01 and S_T < 0.025, confirming negligible influence on output variance. The fourth, λ_citizen, has S_1 = 0.002 but S_T = 0.121 due to interaction effects — a non-negligible contribution. The decision to freeze λ_citizen is a pragmatic choice for computational stability (keeping the calibrable space at 4 dimensions), not a conclusion definitively justified by the data. Section 6.4 quantifies the resulting sensitivity.

3. **Sum diagnostics.** ΣS_1 = 0.70 indicates that approximately 30% of output variance arises from parameter interactions. ΣS_T = 1.51 > 1 confirms significant interaction effects, consistent with the nonlinear softmax mixing mechanism.

[FIGURE 3: Two-panel horizontal bar chart. Left panel: S_1 (main effects). Right panel: S_T (total effects). Y-axis: 8 parameter names (α_herd at top, δ_drift at bottom). X-axis: 0 to 0.6. Error bars show ±CI. Visual grouping: top 4 bars (calibrable) in blue, bottom 4 (frozen) in gray. Key visual: α_herd S_T bar extends to ~0.55, dominant. Note: λ_citizen S_T bar extends to ~0.12, visually non-trivial despite being in the "frozen" group — this is the basis for the discussion in Section 6.4.]

### 6.4 Step Size Sensitivity

Sobol analysis identifies λ_citizen as having non-negligible total effect (S_T = 0.121) despite near-zero main effect (S_1 = 0.002), warranting direct investigation. A perturbation analysis varies λ_citizen by ±40% around the default (0.25), measuring MAE impact on 5 representative scenarios.

**Table 13: λ_citizen Sensitivity Analysis**

| Scenario | MAE (0.6×) | MAE (0.8×) | MAE (1.0×) | MAE (1.2×) | MAE (1.4×) | max Δ (pp) |
|---|---|---|---|---|---|---|
| iPhone X | 5.8 | 5.9 | 5.9 | 7.0 | 11.5 | 5.6 |
| Tesla Cybertruck | 8.5 | 10.5 | 11.3 | 11.6 | 11.8 | 3.2 |
| Dieselgate | 29.8 | 29.6 | 29.6 | 29.6 | 29.7 | 0.2 |
| Uber London | 15.0 | 15.0 | 15.0 | 15.0 | 15.0 | 0.0 |
| United Airlines | 17.7 | 14.2 | 12.0 | 10.8 | 9.9 | 7.9 |

The maximum MAE variation across all scenarios is 7.9 pp (United Airlines), concentrated in high-volatility cases. The median sensitivity across scenarios is 3.2 pp. Scenarios with strong anchoring dynamics (Dieselgate, Uber) are essentially insensitive (max Δ < 0.2 pp), while scenarios dominated by citizen-driven opinion shifts show meaningful dependence.

This result indicates that λ_citizen should be included as a calibrable parameter in future work. The current frozen value was selected from prior predictive checks but is not empirically optimized. We note this as a known limitation that may contribute to the systematic errors observed in high-volatility corporate scenarios.

### 6.5 Cross-Validation with Expanded Dataset

To assess sensitivity to dataset size, we compare calibration on the original 22-scenario dataset versus the expanded 42-scenario dataset:

**Table 14: 22-Scenario vs. 42-Scenario Calibration**

| Metric | 22-Scenario | 42-Scenario | Δ |
|---|---|---|---|
| N (train / test) | 16 / 6 | 34 / 8 | +18 / +2 |
| MAE (test) | 11.7 pp | 19.2 pp | +7.4 |
| RMSE (test) | 14.7 | 26.6 | +11.9 |
| Coverage 90% (test) | 83.3% | 75.0% | -8.3% |
| MAE (train) | 16.3 pp | 14.3 pp | -2.0 |
| Coverage 90% (train) | 68.8% | 79.4% | +10.6% |

The expanded dataset improves training fit (MAE 16.3 → 14.3 pp, coverage 68.8% → 79.4%) but shows degraded test performance, largely driven by the inclusion of more challenging financial and corporate scenarios in the test set. When the NEEDS_VERIFICATION scenario is excluded, test MAE drops to 12.6 pp—comparable to the 22-scenario result (11.7 pp). This suggests that the model's performance ceiling is scenario-dependent rather than data-limited.

### 6.6 Null-Baseline Predictive Skill (Diebold–Mariano)

Sections 6.1–6.5 validate the model's calibration and parameter recovery. They do not, however, answer a distinct and arguably more fundamental question: **does the calibrated LLM-agent simulator actually forecast opinion trajectories better than a trivial statistical baseline?** The 42-scenario dataset averages only 6.3 rounds of ground-truth polling per scenario, so absolute error in the single digits can, in principle, be achieved by a naive model. This subsection quantifies that skill floor.

**Method.** We evaluate four standard forecasters as null baselines against the per-round normalized support $s_t = \text{pro\_pct}_t / 100 \in [0, 1]$ and signed position $p_t = (\text{pro\_pct}_t - \text{against\_pct}_t)/100 \in [-1, 1]$ of every empirical scenario:

1. **Naive persistence** — $\hat{y}_{t+1} = y_t$.
2. **Running mean** — $\hat{y}_{t+1} = \frac{1}{t} \sum_{i=1}^{t} y_i$, a zero-drift random walk collapsed to the sample average.
3. **OLS linear trend** — closed-form slope/intercept on rolling history, extrapolated one step ahead.
4. **AR(1)** — $\hat{y}_{t+1} = \mu + \hat{\phi}(y_t - \mu)$ with $\hat\phi$ from the sample autocorrelation, truncated to $[-0.99, 0.99]$ for stability.

Each forecaster produces a one-step-ahead trajectory using the first 20% of each scenario as training history, emitting one forecast per subsequent round and rolling the realized value forward after each step (one-step-ahead, not recursive multistep). Pairwise predictive skill is tested with the Diebold–Mariano (DM) statistic (Diebold & Mariano, 1995) applied to squared errors, with the Harvey–Leybourne–Newbold (HLN) small-sample correction (Harvey, Leybourne & Newbold, 1997): at horizon $h$, the HLN scale is

$$
\sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}},
$$

and the test statistic is referred to a Student-$t$ distribution with $n - 1$ degrees of freedom rather than the asymptotic $\mathcal{N}(0, 1)$. The HLN correction is non-optional at our sample sizes ($n = 4$ to $n = 9$ forecast pairs per scenario), where uncorrected DM over-rejects.

**Results — pooled across 43 empirical scenarios.**

**Table 15a: Null-baseline skill on normalized support (pro_pct/100)**

| Baseline | Mean RMSE | Median RMSE | Mean terminal error | Significant beats of persistence |
|----------|-----------|-------------|---------------------|-----------------------------------|
| Naive persistence | 0.038 | 0.021 | 0.026 | — |
| Running mean | 0.063 | 0.034 | 0.078 | 0 / 43 |
| OLS linear trend | 0.042 | 0.015 | 0.043 | 4 / 43 |
| AR(1) | 0.055 | 0.034 | 0.057 | 0 / 43 |

**Table 15b: Null-baseline skill on signed position ((pro − against) / 100)**

| Baseline | Mean RMSE | Median RMSE | Mean terminal error | Significant beats of persistence |
|----------|-----------|-------------|---------------------|-----------------------------------|
| Naive persistence | 0.072 | 0.040 | 0.097 | — |
| Running mean | 0.117 | 0.077 | 0.148 | 0 / 43 |
| OLS linear trend | 0.082 | 0.035 | 0.121 | 6 / 43 |
| AR(1) | 0.102 | 0.058 | 0.118 | 0 / 43 |

*"Significant beats of persistence" counts scenarios where $\text{DM}(\text{baseline}, \text{persistence})$ yields $p < 0.05$ with the baseline having lower mean squared loss. "Terminal error" is the absolute deviation of the last emitted forecast from the verified ground-truth outcome.*

**Three findings.**

1. **Naive persistence is a surprisingly strong baseline.** The mean RMSE of 0.038 on normalized support corresponds to a 3.8 percentage-point one-step-ahead error on pro-support — below the 12.6 pp MAE the hierarchical Bayesian model achieves on full-scenario final outcomes. The two metrics are not directly comparable (one-step-ahead vs full-scenario terminal forecast; normalized support vs raw pp) but the number establishes a tight budget: if a complex LLM simulator cannot improve on a one-line forecaster for intermediate dynamics, the added complexity is difficult to justify.

2. **No baseline reliably dominates persistence.** OLS linear trend attains the lowest median RMSE on both metrics (0.015 on support, 0.035 on signed position), but statistical significance relative to persistence is achieved on only 4/43 and 6/43 scenarios respectively. The running mean and AR(1) baselines never beat persistence at $p < 0.05$, and on the mean they lose substantially (RMSE 0.063 vs 0.038 on support). Empirical polling trajectories are heavily persistent, with round-to-round variance dominated by the previous round's level rather than by trend or mean-reversion components detectable at our sample sizes.

3. **Skill is domain-heterogeneous by a factor of five.** Table 16 breaks persistence RMSE down by domain: political scenarios are the easiest to forecast (0.012 on support), financial the hardest (0.063), with corporate and commercial clustered in between. This spread is the most actionable finding of the benchmark: a calibrated simulator should be expected to add little lift on low-volatility political polling (where persistence is near-optimal) but has substantial room to contribute on financial and corporate scenarios, which is precisely where the hierarchical model's worst performance is observed in Section 5.5.

**Table 16: Persistence RMSE by domain (support metric, n scenarios in parentheses)**

| Domain | n | Persistence RMSE | OLS linear trend RMSE | AR(1) RMSE |
|--------|---|------------------|-----------------------|------------|
| Political | 15 | 0.012 | 0.009 | 0.020 |
| Labor | 1 | 0.009 | 0.008 | 0.016 |
| Environmental | 2 | 0.021 | 0.013 | 0.033 |
| Corporate | 8 | 0.048 | 0.062 | 0.065 |
| Public health | 5 | 0.052 | 0.054 | 0.081 |
| Commercial | 5 | 0.059 | 0.075 | 0.079 |
| Financial | 7 | 0.063 | 0.073 | 0.094 |

*Values are mean RMSE per baseline on the `support` metric (pro_pct/100). Lower is better; persistence is the column to beat.*

The 5× persistence RMSE gap between political (0.012) and financial (0.063) mirrors the $|b_s|_\text{financial} = 0.74$ logit-space bias of Section 5.3. Both point to the same conclusion: financial scenarios exhibit regime-switching dynamics (e.g. SVB collapse, Dieselgate) that neither simple persistence nor a stationary hierarchical Gaussian discrepancy captures well.

**Interpretation as a skill floor.** We do not claim that our calibrated model beats all four baselines on all scenarios — Sections 5.4 and 5.5 make the opposite point explicit. Rather, we release Tables 15a, 15b, and 16 as a *reproducible reference distribution* that future iterations of DigitalTwinSim (including the EnKF of Section 7 once extended to multi-step ahead) can be compared against without ambiguity. Appendix D describes how any modification to the sim can be automatically re-evaluated against this same benchmark via a one-line command.

### 6.7 Empirical Coverage via Residual Bootstrap

The credible intervals reported in Section 5.4 are computed in logit space and back-transformed through the sigmoid, guaranteeing bounds within $[0, 100]$ by construction. This is a structural property of the parametric likelihood; it does not, by itself, establish that the intervals are empirically well-calibrated. Section 6.5's 85.7% coverage of 90% credible intervals on verified test scenarios does — but that calculation assumes a specific parametric form for the posterior. We provide a model-free complement below.

**Method.** For each empirical scenario and each baseline forecaster, we pool the in-sample residuals $r_t = y_t - \hat{y}_t$ and construct a residual-bootstrap predictive interval at each round: draw $B = 500$ residuals with replacement, add them to $\hat{y}_t$, and take empirical $[\alpha/2, 1 - \alpha/2]$ quantiles. Empirical coverage is then the fraction of realized values $y_t$ that fall inside $[\hat{y}_t + r_{(B\alpha/2)}, \hat{y}_t + r_{(B(1-\alpha/2))}]$, aggregated across all $\sum_i n_i = 271$ round-level observations in the corpus. A nonparametric bootstrap on the coverage statistic itself yields a 95% confidence band, and a coverage report flags the interval as "calibrated" iff the nominal coverage lies within the band.

**Results.** At a nominal 90% level, the persistence baseline's residual-bootstrap intervals cover 88.4% of round-level observations pooled across the 43-scenario corpus (bootstrap 95% CI: [85.6%, 91.1%]), agreeing with nominal within the bootstrap band. The OLS linear trend baseline's intervals cover 89.2% (CI [86.4%, 91.9%]). On individual scenarios, coverage fluctuates substantially — 26 out of 43 land above 90%, 17 below, with scenarios in the `critical` tension bucket (Dieselgate, United Airlines, iPhone X, Eurozone Monetary Shock) systematically under-covered by the persistence model as expected.

**Interpretation.** The model-free residual-bootstrap agrees with the logit-space parametric coverage of Section 5.4 at the corpus aggregate level. The agreement is worth noting because the two calculations use entirely independent mathematical machinery (sigmoid-transformed Gaussian posteriors vs pooled empirical residual quantiles) and yet converge on similar calibration conclusions. Disagreement would have flagged a potential systematic mismatch between the parametric posterior and empirical error distributions; agreement increases our confidence that the reported 85.7% coverage is not an artifact of the logit parameterization.

### 6.8 Scenario-Diversity Matrix

A pooled performance number can hide systematic under-coverage of a single axis value — for example, a model might average well across domains while failing uniformly on APAC scenarios or high-tension ones. To make axis-level gaps visible, we define a three-axis scenario matrix:

- **Domain** (7 values): financial, commercial, corporate, political, public_health, environmental, labor.
- **Region** (5 values): EU, US, APAC, LATAM, GLOBAL (region inferred from ISO country code).
- **Tension** (4 values): low, moderate, high, critical (inferred from per-scenario volatility: the population standard deviation of round-to-round signed-position deltas, bucketed at 0.03 / 0.07 / 0.15).

The Cartesian product contains $7 \times 5 \times 4 = 140$ cells, each representing a qualitatively distinct operating regime.

**Table 17: Scenario-matrix coverage of the 43-scenario empirical corpus**

| Axis | Value | Scenarios |
|------|-------|-----------|
| Domain | political | 15 |
| Domain | corporate | 8 |
| Domain | financial | 7 |
| Domain | commercial | 5 |
| Domain | public_health | 5 |
| Domain | environmental | 2 |
| Domain | labor | 1 |
| Region | US | 22 |
| Region | EU | 16 |
| Region | APAC | 2 |
| Region | LATAM | 2 |
| Region | GLOBAL | 1 |
| Tension | low | 26 |
| Tension | moderate | 7 |
| Tension | critical | 6 |
| Tension | high | 4 |

The corpus occupies 25 of the 140 possible cells (18%). No axis value is empty — every domain, region, and tension bucket contains at least one scenario — but the distribution is visibly uneven: US + EU scenarios account for 38/43 (88%), and the labor / environmental / GLOBAL buckets each contain at most two scenarios. We report these imbalances explicitly so that downstream claims ("the model generalizes across tensions") can be weighed against the small-sample cells that support them.

**Forward-looking use.** The matrix is intended as a gating criterion for future scenario additions. A prospective new scenario is prioritized if it fills an under-sampled cell — for example, an APAC labor scenario would lift two weak axes at once. Tables 17 and 16 together suggest that the highest-marginal-value additions would be (i) additional APAC and LATAM scenarios across any domain, (ii) additional financial scenarios with `critical` tension, which currently has only 6 supporting scenarios despite representing the most diagnostic regime for model discrimination.

---

## 7. Online Data Assimilation via Ensemble Kalman Filter

### 7.1 Motivation

The calibrated posterior provides a static prediction: given a scenario, produce a probabilistic forecast of the final outcome. But real-world opinion dynamics unfold over time, and intermediate observations (polls, social media sentiment, official statements) become available as the process evolves. An online assimilation system should:

1. Start from the calibrated prior (posterior from Section 4).
2. Incorporate streaming observations as they arrive.
3. Jointly update both model parameters and agent states.
4. Produce calibrated probabilistic forecasts at each round.

This bridges the gap between offline calibration and real-time digital twin operation.

### 7.2 Ensemble Kalman Filter Formulation

We adopt a stochastic Ensemble Kalman Filter (Evensen, 2003) operating on an augmented state vector that combines model parameters with agent positions.

**State vector.** Each ensemble member j ∈ {1, ..., E} maintains:

$$\mathbf{x}_j = [\boldsymbol{\theta}_j, \mathbf{z}_j]^\top \quad \text{(Eq. 25)}$$

where $\boldsymbol{\theta}_j = [\alpha_h, \alpha_a, \alpha_s, \alpha_e]$ are the four calibrable parameters and $\mathbf{z}_j = [p_1, ..., p_n]$ are the n agent positions. The state dimension is 4 + n.

**Forecast step.** At each round t, the forecast propagates parameters as a random walk and states through the JAX simulator:

$$\boldsymbol{\theta}_j^f(t+1) = \boldsymbol{\theta}_j^a(t) + \boldsymbol{\eta}_j^\theta, \quad \boldsymbol{\eta}_j^\theta \sim \mathcal{N}(0, Q_\theta I) \quad \text{(Eq. 26)}$$

$$\mathbf{z}_j^f(t+1) = \text{step\_round}(\mathbf{z}_j^a(t), \boldsymbol{\theta}_j^f(t+1), \text{event}_t) + \boldsymbol{\eta}_j^z, \quad \boldsymbol{\eta}_j^z \sim \mathcal{N}(0, Q_z I) \quad \text{(Eq. 27)}$$

where Q_θ = 0.01 controls parameter exploration speed and Q_z = 0.005 adds stochastic perturbation to agent positions. The forecast is parallelized over the ensemble using `jax.vmap`.

**Update step.** When an observation y_obs with variance R is available:

1. Compute ensemble predictions: $\hat{y}_j = h(\mathbf{z}_j^f)$ where h(·) is the readout function (Section 3.6).

2. Compute anomalies: $\mathbf{X}^{\text{anom}} = \mathbf{X}^f - \bar{\mathbf{X}}^f$, $\hat{y}^{\text{anom}} = \hat{y} - \bar{\hat{y}}$.

3. Cross-covariance and innovation variance:

$$\mathbf{P}_{xh} = \frac{1}{E-1} \mathbf{X}^{\text{anom}} (\hat{y}^{\text{anom}})^\top, \quad P_{hh} = \frac{1}{E-1} \|\hat{y}^{\text{anom}}\|^2 \quad \text{(Eq. 28)}$$

4. Kalman gain: $\mathbf{K} = \mathbf{P}_{xh} / (P_{hh} + R)$ (Eq. 29)

5. Perturbed update:

$$\mathbf{x}_j^a = \mathbf{x}_j^f + \mathbf{K} \cdot (y_{\text{obs}} + \epsilon_j - \hat{y}_j), \quad \epsilon_j \sim \mathcal{N}(0, R) \quad \text{(Eq. 30)}$$

6. Multiplicative inflation to prevent ensemble collapse:

$$\mathbf{x}_j^a \leftarrow \bar{\mathbf{x}}^a + \gamma (\mathbf{x}_j^a - \bar{\mathbf{x}}^a), \quad \gamma = 1.02 \quad \text{(Eq. 31)}$$

### 7.3 Observation Adapters

The EnKF supports three observation types through adapter classes:

- **PollingSurvey**: pro_pct ∈ [0, 100] with variance inversely proportional to sample size: R = pro_pct · (100 - pro_pct) / sample_size.
- **SentimentSignal**: Maps sentiment scores to approximate pro_pct with configurable noise floor.
- **OfficialResult**: Final certified outcome with minimal observation noise (R = 1.0).

### 7.4 Brexit Case Study

We note that the Brexit scenario is part of the calibration training set. The offline posterior therefore already encodes information from this scenario's ground truth, and the prior prediction (round 0: 50.3%) starts close to the final outcome. This case study demonstrates the operational mechanics of EnKF assimilation — observation ingestion, CI dynamics, baseline comparison — rather than constituting an independent out-of-sample validation. A rigorous EnKF validation would require assimilation on held-out scenarios with real-time polling data, which we identify as a priority for future work.

We demonstrate the EnKF on the Brexit referendum scenario (ground truth: 51.89% Leave). Six polling observations are available across the 6-round simulation, one per round, with polls ranging from 41.0% to 44.0% (consistent with the well-documented polling bias that underestimated Leave support).

**Table 15a: EnKF Round-by-Round on Brexit (GT = 51.89%)**

| Round | Observation | EnKF Mean (%) | 90% CI | CI Width (pp) | Δ Width |
|---|---|---|---|---|---|
| 0 (prior) | — | 50.3 | [50.2, 50.4] | 0.2 | — |
| 1 | 41.0% (N=1000) | 50.7 | [50.5, 51.3] | 0.8 | +0.6 |
| 2 | 42.0% (N=1000) | 51.4 | [50.9, 52.0] | 1.0 | +0.2 |
| 3 | 43.0% (N=1000) | 50.3 | [50.1, 50.7] | 0.6 | -0.5 |
| 4 | 43.0% (N=1000) | 50.0 | [50.0, 50.1] | 0.1 | -0.5 |
| 5 | 44.0% (N=1000) | 50.0 | [50.0, 50.1] | 0.1 | +0.0 |
| 6 (final) | 44.0% (N=1000) | 50.1 | [50.0, 50.4] | 0.4 | +0.3 |

We note that the final 90% CI [50.0, 50.4] does not cover the ground truth (51.89%). This indicates that the EnKF posterior is under-dispersed at the final round — the ensemble has collapsed to a narrow band around 50.1% that excludes the true value. This under-dispersion likely results from two factors: (i) the prior ensemble is already highly concentrated (initialized from a well-calibrated posterior on a training-set scenario), leaving little room for the filter to explore; and (ii) the multiplicative inflation factor (γ = 1.02) may be insufficient to maintain ensemble diversity over 6 rounds of updates. Higher inflation (γ = 1.05–1.10) or adaptive inflation schemes could address this. The point prediction (50.1%, error 1.8 pp) is accurate, but the uncertainty estimate should be treated as overconfident.

The prior CI width (0.2 pp) reflects the highly concentrated offline posterior—the ensemble is initialized from a well-calibrated distribution. The CI initially expands as the dynamics model evolves (rounds 1–2), reflecting genuine uncertainty about the trajectory, then contracts sharply by round 4 (0.1 pp) as repeated observations constrain the ensemble. The slight expansion at round 6 reflects the final forecast step incorporating both the observation and the model's forward projection.

Despite the polls systematically underestimating the final outcome by ~8 pp, the dynamics model—which captures mechanisms such as late-breaking opinion shifts and shy voter effects through the herd and anchor forces—produces a final prediction of 50.1%, within 1.8 pp of the ground truth.

**Table 15b: EnKF vs Baselines**

| Method | Final Prediction (%) | Error (pp) | Uses Dynamics | Updates Params |
|---|---|---|---|---|
| Last available poll | 44.0 | 7.9 | No | No |
| Running poll average | 42.8 | 9.1 | No | No |
| EnKF (state only, θ fixed) | 50.1 | 1.8 | Yes | No |
| EnKF (state + params) | 50.1 | 1.8 | Yes | Yes |

The EnKF with dynamics model reduces prediction error by 77% compared to the last-available-poll baseline (1.8 pp vs 7.9 pp) and 80% compared to the running average (1.8 pp vs 9.1 pp). The dynamics model—even with frozen parameters—adds substantial value by propagating opinion evolution between observation points and capturing mechanisms that simple extrapolation misses.

State-only and state+params variants converge to the same prediction with 6 observations, which is expected: with abundant data, the state update dominates the parameter update. The value of joint parameter-state estimation would be more evident with sparse observations (1–2 polls), where the calibrated prior on θ provides more leverage.

It is important to note that the 1.8 pp EnKF error is achieved *with streaming polling data*—it is not directly comparable to the offline-only MAE of 12.6 pp, which uses no scenario-specific observations. The EnKF demonstrates what becomes possible when the digital twin receives live data, not a claim about the base model's accuracy.

[FIGURE 4: Two-panel vertically stacked plot. TOP PANEL: X-axis rounds 0–6, Y-axis pro% (35–55%). Solid blue line: EnKF mean prediction (starts at 50.3%, fluctuates between 50.0–51.4%, ends at 50.1%). Light blue shaded area: 90% CI. Red dashed horizontal line at 51.89% (GT). Green dots at each round showing poll values (41–44%), visually below the model prediction. Key visual: model prediction stays near 50% despite polls at 41–44%, demonstrating that the dynamics model "sees through" the polling bias. CI narrows from 0.8 pp (round 1) to 0.1 pp (round 4–5), then slightly expands to 0.4 pp at final round. BOTTOM PANEL: X-axis rounds 0–6, Y-axis CI width in pp (0–1.2). Step function showing width pattern: 0.2 (prior) → 0.8 → 1.0 → 0.6 → 0.1 → 0.1 → 0.4. Annotations: "Prior width = 0.2 pp" at round 0, "Final width = 0.4 pp" at round 6.]

### 7.5 Convergence Verification

A synthetic convergence test with known ground-truth parameters confirms that the EnKF recovers true parameters when given sufficient observations. Starting from a vague prior centered at zero (true values: α_h = -0.2, α_a = 0.3, α_s = -0.1, α_e = -0.15), all four parameters converge to within 0.5σ of their true values after 9 rounds of observations with 3 pp noise. A no-observation baseline confirms that without data, the EnKF produces prior-consistent forecasts with no information gain, as expected.

---

## 8. Discussion

### 8.1 Strengths

**Principled calibration.** By combining LLM-generated agent dynamics with Bayesian inference, we avoid the common pitfall of treating LLM outputs as ground truth. The hierarchical structure enables partial pooling across diverse domains while the discrepancy model explicitly accounts for structural misspecification.

**Validated inference.** SBC confirms that the generative model is well-specified and that NUTS recovers correct posteriors (6/6 parameters pass). Sobol analysis provides a principled basis for the calibrable/frozen partition.

**Online assimilation.** The EnKF bridges offline calibration to real-time operation, reducing Brexit prediction error by 77% over polling baselines with six observations.

**Differentiable architecture.** The entire pipeline—simulation, readout, likelihood—is implemented in JAX and compatible with automatic differentiation, enabling gradient-based inference at scale.

### 8.2 Limitations

**Financial domain performance.** The model systematically over-predicts support in financial crisis scenarios (mean |b_s| = 0.744 in logit space). This reflects a fundamental limitation: financial crises involve trust cascades, contagion dynamics, and informational asymmetries that the current force model—designed for gradual opinion evolution—cannot capture. The multi-modal calibration (Section 5.8) demonstrates that incorporating historical equity returns does not substantially alleviate this limitation: the posterior modality weight λ_fin = 0.045 indicates that market returns are too noisy to constrain the opinion dynamics parameters, and the financial domain MAE remains at 24.0 pp with only 57% coverage even when market data is included in the likelihood.

**Variational approximation quality.** The SVI posterior agrees with NUTS on the dominant parameters (α_herd, α_anchor: |Δμ|/σ_NUTS < 0.4) but shows concentration bias on weaker parameters (α_social: 0.85; α_event: 1.71). Uncertainty estimates on α_social and α_event are likely underestimated. This is a known limitation of structured variational families with restricted covariance (Blei et al., 2017)—even the low-rank plus diagonal structure of the AutoLowRankMultivariateNormal guide cannot fully capture the posterior geometry on weakly identified parameters, trading exactness for computational tractability.

**Step size sensitivity.** The frozen citizen step size λ_citizen shows non-negligible sensitivity: ±40% perturbation produces up to 7.9 pp MAE variation on individual scenarios, with a median of 3.2 pp across test cases. This confirms the Sobol total effect (S_T = 0.121) and indicates λ_citizen should be promoted to a calibrable parameter in future work. The current frozen value, while selected from prior predictive checks, contributes to systematic errors in high-volatility scenarios.

**LLM stochasticity.** The current framework treats LLM-generated agent behaviors (Δ^{LLM}_i and events) as fixed inputs to the force system. In practice, LLM outputs are stochastic: different seeds produce different narratives, coalition dynamics, and opinion shifts. This stochasticity is absorbed into the discrepancy terms b_s but is not explicitly separated from structural model error.

A more principled approach would run multiple LLM rollouts per scenario (e.g., 5–10 with different seeds), producing an ensemble of behavioral trajectories, and calibrate on the predictive mean or on a marginal likelihood that integrates over narrative paths. This would disentangle LLM variance from opinion dynamics misspecification, producing more interpretable discrepancy terms. We defer this to future work as it requires substantial additional compute (5–10× per scenario).

**Sample size.** With 42 scenarios (34 training), the dataset is small by machine learning standards. Several domains have only 1–2 scenarios, making domain-level inference unreliable for those domains. The 4 domains with zero test scenarios (energy, environmental, labor, social) cannot be validated out-of-sample.

**Observation model simplifications.** The BetaBinomial likelihood assumes independent polling rounds and does not model autocorrelation in opinion trajectories. The EnKF's Gaussian assumption may not hold for highly polarized scenarios where the opinion distribution is bimodal.

**Regime switching.** We have developed a preliminary regime-switching extension for discontinuous crisis dynamics, described in Appendix B. The extension uses soft sigmoid-based switching between normal and crisis regimes, preserving JAX differentiability. Results are mixed: regime switching substantially improves scenarios where the crisis push aligns with shock direction (Dieselgate: 29.6 → 0.4 pp error) but degrades scenarios with ambiguous or multi-directional events (SVB: 38.8 → 61.9 pp worse). The crisis parameters are hand-tuned defaults and require calibration via SVI on the full empirical dataset. We consider this a promising direction that is not yet ready for production use.

### 8.3 Future Work

1. **λ_citizen calibration.** Promote λ_citizen from frozen to calibrable based on the sensitivity analysis results. This increases the parameter space from 4 to 5 calibrable parameters but may reduce systematic errors in high-volatility scenarios.

2. **v3 regime switching calibration.** Run SVI on the v3 hierarchical model with learnable crisis parameters to determine optimal activation thresholds and crisis dynamics.

3. **LLM ensemble calibration.** Run multiple LLM rollouts per scenario to separate LLM stochasticity from structural model error in the discrepancy decomposition.

4. **NUTS initialization from SVI.** Use the SVI posterior as a warm start for short NUTS chains to obtain better-calibrated uncertainty estimates, particularly for α_social and α_event.

5. **Temporal observation model.** Replace the independent-rounds assumption with a state-space observation model that captures autocorrelation.

6. **Expanded dataset.** Curate additional scenarios in underrepresented domains (energy, environmental, labor) to improve domain-level estimates.

7. **EnKF + regime switching integration.** Combine online assimilation with regime detection to dynamically activate crisis dynamics based on observed data.

8. **Out-of-sample EnKF validation.** Apply the EnKF to held-out scenarios with sequential polling data to validate assimilation performance independently of the calibration training set.

9. **Higher-resolution financial linkage.** The current multi-modal calibration (Section 5.8) operates at per-round granularity (~14-day windows), which averages out the high-frequency market response to opinion events. Daily or intraday returns around specific events (earnings calls, policy announcements, viral episodes) would provide a higher signal-to-noise ratio. Additionally, options-implied volatility and credit default swap spreads carry information about tail-risk perception that is more directly linked to opinion dynamics than equity returns.

10. **Expanded market data coverage.** Extending market enrichment from 14 to all 42 scenarios—using sector ETFs as proxies for political, public health, and environmental domains—would increase the effective sample for financial linkage parameter estimation, potentially pushing λ_fin above the identifiability threshold.

---

## 9. Conclusion

We have presented DigitalTwinSim, a framework for transforming LLM-agent opinion simulations from uncalibrated narrative generators into quantitative digital twins with calibrated uncertainty. The key insight is that LLM-agent simulations, despite their structural misspecification, contain learnable signal about opinion dynamics mechanisms—but only when combined with principled Bayesian calibration and explicit model discrepancy.

The calibrated model achieves 12.6 pp MAE on verified held-out scenarios (19.2 pp on the full test set including one data-quality-flagged scenario) with 85.7% coverage of 90% credible intervals. Credible intervals are computed in logit space and back-transformed, guaranteeing [0, 100] bounds by construction. Sobol sensitivity analysis identifies herd behavior and anchor rigidity as the dominant mechanisms (S_T = 0.55 and 0.45 respectively), with their interaction accounting for most nonlinear output variance. Simulation-based calibration confirms posterior validity across all 6 parameters, though comparison with NUTS reveals that the SVI approximation underestimates uncertainty on weaker parameters.

The Ensemble Kalman Filter extends the framework to online operation, achieving 1.8 pp error on an in-sample Brexit case study with six streaming polling observations—a 77% improvement over the last-available-poll baseline. This demonstrates the value of combining mechanistic opinion dynamics with data assimilation, even when polling data alone would suggest a different outcome.

A multi-modal extension that jointly calibrates on polling outcomes and financial market returns demonstrates improved test coverage (75.0% → 87.5%) but reveals a principled negative result: the learned modality weight λ_fin = 0.045 indicates that equity returns carry insufficient signal-to-noise for meaningfully constraining opinion dynamics parameters. All three linkage weight posteriors (opinion, event, polarization → market returns) have credible intervals spanning zero. This quantifies the information boundary between opinion dynamics and financial markets at the current data granularity and sample size, and sets concrete targets for future work: higher-frequency market data, richer financial features, and expanded scenario coverage.

The framework has known limitations: sensitivity of the frozen λ_citizen parameter (up to 7.9 pp MAE variation), systematic over-prediction in financial domains (mean |b_s| = 0.74 logit), and variational posterior concentration on weaker parameters. These are reported transparently and motivate concrete next steps: promoting λ_citizen to calibrable, running NUTS refinement of the SVI posterior, and calibrating the regime-switching extension for crisis dynamics.

The framework's modular design—force-based dynamics, hierarchical Bayesian calibration, online assimilation, regime switching, multi-modal likelihood—allows each component to be extended independently. The immediate priorities are λ_citizen calibration, NUTS-initialized uncertainty refinement, and higher-resolution financial linkage, all of which can be pursued without modifying the core architecture.

---

## Appendix A: Implementation Details

### A.1 Software Stack

The simulator and inference pipeline are implemented in Python using:
- **JAX** (0.9+) for differentiable simulation and automatic differentiation.
- **NumPyro** for probabilistic programming and variational inference.
- **SALib** for Sobol sensitivity analysis.
- All simulation code is `jax.jit`-compatible and uses `jax.lax.scan` for sequential round stepping, avoiding Python loops.

### A.2 Computational Requirements

| Task | Runtime | Hardware |
|---|---|---|
| SVI calibration (3000 steps, 42 scenarios) | 40.2 min | CPU (Apple Silicon) |
| SBC (100 instances, NUTS) | 5.0 min | CPU |
| Sobol analysis (18,432 evaluations) | ~10 min | CPU |
| SVI vs NUTS comparison (4 scenarios) | ~15 min | CPU |
| EnKF online assimilation (50 members, 9 rounds) | ~5 sec | CPU |

### A.3 Gauge Fixing

The softmax function is shift-invariant: softmax(α + c) = softmax(α) for any scalar c. With 5 forces and 4 free parameters, one weight must be fixed to resolve the gauge. We fix α_direct = 0, making the remaining weights interpretable as log-odds ratios relative to direct LLM influence. This is analogous to the reference category in multinomial logistic regression.

### A.4 EMA Standardization

The EMA standardization computes per-force, per-round statistics (mean and standard deviation) across ALL agents in each round, accumulated via exponential moving average over rounds with decay α = 0.3. The current-round statistics are:

$$\mu_k^{\text{cur}}(t) = \frac{1}{n}\sum_i f_{k,i}(t), \quad \sigma_k^{\text{cur}}(t) = \sqrt{\frac{1}{n}\sum_i (f_{k,i}(t) - \mu_k^{\text{cur}})^2 + \epsilon}$$

The gradient-safe square root (adding ε = 10⁻⁸ before taking the root) avoids NaN gradients when all agent forces are identical. These are then smoothed temporally:

$$\mu_k(t) = \alpha \cdot \mu_k(t-1) + (1-\alpha) \cdot \mu_k^{\text{cur}}(t)$$

A permutation invariance test confirms that agent ordering does not affect the standardized forces (max absolute difference < 6 × 10⁻⁸ under random permutation of the agent array). This property holds because mean and variance are symmetric statistics over the agent set.

---

## Appendix B: Regime Switching for Crisis Dynamics (Preliminary)

### B.1 Motivation

The five worst-performing scenarios (Section 5.5) share a common pattern: rapid, discontinuous opinion shifts that the base model's linear force combination cannot reproduce. Financial crises (WeWork b_s = -1.378, SVB b_s = -0.835, FTX b_s = -0.727) exhibit trust collapses where public opinion drops precipitously within 1–2 rounds. The base model, with its smooth force mixing and capped step sizes, can only produce gradual movements.

We address this with a regime-switching extension that introduces a latent "crisis regime" with amplified dynamics, activated by observable signals (shock magnitude, position velocity, institutional trust).

### B.2 Two-Regime Architecture

The model switches between two regimes:

- **Regime 0 (Normal)**: Standard force-based opinion evolution as described in Section 3.
- **Regime 1 (Crisis)**: Amplified step sizes, suppressed anchor rigidity, amplified herd and event forces, and a direct crisis push that bypasses the softmax mixing entirely.

The switching is *soft*: regime probability P(crisis) ∈ (0, 1) is computed via sigmoid, and all parameters are interpolated:

$$\boldsymbol{\theta}_{\text{eff}} = (1 - p) \cdot \boldsymbol{\theta}_{\text{normal}} + p \cdot \boldsymbol{\theta}_{\text{crisis}} \quad \text{(Eq. B.1)}$$

This preserves JAX differentiability and `jax.lax.scan` compatibility—no discrete branching is needed.

### B.3 Regime Probability

The crisis probability is computed via logistic regression on four observable signals:

$$\text{logit}(c_t) = \beta_0 + \beta_{\text{shock}} (m_t - \tau_{\text{shock}}) + \beta_{\text{vel}} (v_t - \tau_{\text{vel}}) + \beta_{\text{trust}} (1 - \text{trust}) + \beta_{\text{recovery}} \cdot r_t + \beta_{\text{momentum}} \cdot c_{t-1} \quad \text{(Eq. B.2)}$$

where m_t is the shock magnitude, v_t the mean absolute position velocity, trust the scenario-level institutional trust, r_t a soft count of crisis rounds, and p_{t-1} the previous crisis probability.

### B.4 Crisis Dynamics

In the crisis regime, three modifications occur:

1. **Step size amplification.** λ_crisis = λ_normal × 3.0, with relaxed caps.
2. **Force reweighting.** Herd amplified by log(2.0), event amplified by log(2.5), anchor suppressed by -2.0 in logit space.
3. **Direct crisis push.** A force that bypasses the softmax mixing: Δp_i^{crisis} = p · κ · m_t · d_t · (1 - ρ_i), where κ = 0.4.

**Table B.1: Crisis Parameter Defaults**

| Parameter | Default | Role |
|---|---|---|
| λ_multiplier | 3.0 | Step size amplification in crisis |
| anchor_suppression | 0.1 | Anchor force multiplied by this in crisis |
| event_amplification | 2.5 | Event force amplified in softmax |
| herd_amplification | 2.0 | Herd force amplified in softmax |
| contagion_speed (κ) | 0.4 | Direct crisis push strength |
| shock_trigger (τ_shock) | 0.5 | Shock magnitude threshold |
| trust_sensitivity | 1.5 | Low trust → higher crisis probability |
| crisis_duration_mean | 2.5 | Expected rounds before recovery |

### B.5 Results

We compare the base model (v2) against the regime-switching model (v3) on selected financial and corporate scenarios:

**Table B.2: v2 vs v3 on Financial/Corporate Scenarios**

| Scenario | GT (%) | v2 Error | v3 Error | Δ (pp) | Max P(crisis) | Crisis Rounds |
|---|---|---|---|---|---|---|
| Dieselgate | 32.0 | 29.6 | 0.4 | +29.1 | 0.930 | 3 |
| FTX Crypto | 22.0 | 11.4 | 5.3 | +6.0 | 1.000 | 2 |
| Facebook→Meta | 26.0 | 14.6 | 6.4 | +8.2 | 0.911 | 2 |
| Archegos | 35.0 | 65.0 | 54.5 | +10.4 | 0.930 | 2 |
| Boeing 737 MAX | 40.0 | 9.5 | 23.7 | -14.2 | 0.833 | 1 |
| SVB Collapse | 38.0 | 38.8 | 61.9 | -23.1 | 0.989 | 2 |
| Twitter/X | 41.0 | 7.8 | 25.5 | -17.7 | 0.826 | 2 |

Results are mixed. Regime switching produces substantial improvements when crisis dynamics align with shock direction but degrades performance when the crisis push amplifies the wrong direction. The high max P(crisis) across all scenarios (0.67–1.0) with default thresholds suggests the trigger is too sensitive and requires calibration via SVI.

### B.6 Design for Calibration

The crisis parameters are designed to be learnable within the hierarchical framework. The v3 model extends the v2 prior with 5 additional global parameters (crisis_lambda_mult, crisis_anchor_supp, crisis_event_amp, crisis_herd_amp, shock_trigger), shared globally as crisis dynamics are hypothesized to be a universal mechanism modulated by scenario-specific triggers.

---

## Appendix C: Full Scenario List

**Table C.1: Full Empirical Dataset (42 scenarios)**

| # | Scenario ID | Domain | GT (%) | Rounds | Polls | Split | Notes |
|---|---|---|---|---|---|---|---|
| 1 | COM-2017-IPHONE_X | commercial | 65.0 | 5 | 2/5 | Train | |
| 2 | COM-2019-TESLA_CYBERTRUCK_REVEAL | commercial | 62.0 | 7 | 1/7 | Test | |
| 3 | CORP-2015-DIESELGATE_VW | corporate | 32.0 | 5 | 1/5 | Train | |
| 4 | CORP-2017-UBER_LONDON_LICENSE | corporate | 65.0 | 9 | 1/9 | Train | |
| 5 | CORP-2017-UNITED_AIRLINES_DRAGGING | corporate | 28.0 | 6 | 1/6 | Train | |
| 6 | CORP-2018-AMAZON_HQ2_NYC | corporate | 56.0 | 7 | 1/7 | Test | |
| 7 | CORP-2019-BOEING_737_MAX | corporate | 40.0 | 7 | 1/7 | Train | |
| 8 | CORP-2021-FACEBOOK_META_REBRAND | corporate | 26.0 | 7 | 1/7 | Train | |
| 9 | CORP-2022-TWITTER_X_ACQUISITION | corporate | 41.0 | 5 | 1/5 | Train | |
| 10 | ENE-2012-JAPANESE_NUCLEAR_RESTART | energy | 35.0 | 6 | 4/6 | Train | |
| 11 | ENV-2018-GRETA_CLIMATE_STRIKES | environmental | 71.0 | 6 | 0/6 | Train | |
| 12 | FIN-2019-WEWORK_IPO_COLLAPSE | financial | 18.0 | 7 | 1/7 | Train | |
| 13 | FIN-2020-TESLA_STOCK_SPLIT | financial | 72.0 | 7 | 0/7 | Train | |
| 14 | FIN-2021-AMC_SHORT_SQUEEZE | financial | 62.0 | 7 | 0/7 | Train | |
| 15 | FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE | financial | 35.0 | 7 | 0/7 | Test | NEEDS_VERIF |
| 16 | FIN-2021-GAMESTOP | financial | 72.0 | 7 | 1/7 | Train | |
| 17 | FIN-2022-FTX_CRYPTO_CRISIS | financial | 22.0 | 7 | 2/7 | Train | |
| 18 | FIN-2023-SVB_COLLAPSE | financial | 38.0 | 6 | 2/6 | Train | |
| 19 | LAB-2015-UBER_VS_TAXI_FRANCE | labor | 45.0 | 7 | 2/7 | Train | |
| 20 | PH-2021-ASTRAZENECA_HESITANCY | public_health | 62.0 | 7 | 0/7 | Train | |
| 21 | PH-2021-COVID_VAX_IT | public_health | 80.0 | 7 | 7/7 | Test | |
| 22 | PH-2021-MASKING_MANDATE_USA | public_health | 63.0 | 5 | 2/5 | Train | |
| 23 | PH-2021-VACCINE_HESITANCY_USA | public_health | 67.0 | 5 | 5/5 | Train | |
| 24 | PH-2022-MONKEYPOX_CONCERN_USA | public_health | 47.0 | 7 | 1/7 | Train | |
| 25 | POL-2011-REFERENDUM_DIVORZIO_MALTA | political | 53.2 | 6 | 0/6 | Train | |
| 26 | POL-2014-SCOTTISH_INDEPENDENCE | political | 44.7 | 7 | 7/7 | Train | |
| 27 | POL-2015-GREEK_BAILOUT_GREXIT | political | 38.7 | 5 | 0/5 | Test | |
| 28 | POL-2016-BREXIT | political | 51.9 | 6 | 6/6 | Train | |
| 29 | POL-2016-REFERENDUM_COSTITUZIONALE_IT | political | 40.9 | 7 | 0/7 | Train | |
| 30 | POL-2017-PRESIDENZIALI_FRANCIA | political | 66.1 | 6 | 6/6 | Test | |
| 31 | POL-2017-INDIPENDENZA_CATALOGNA | political | 48.0 | 5 | 1/5 | Train | |
| 32 | POL-2017-TURKISH_CONSTITUTIONAL_REF | political | 51.4 | 6 | 0/6 | Test | |
| 33 | POL-2018-USA_MIDTERM_HOUSE | political | 53.4 | 6 | 6/6 | Train | |
| 34 | POL-2018-REFERENDUM_ABORTO_IRLANDA | political | 66.4 | 6 | 3/6 | Train | |
| 35 | POL-2019-EUROPEE_ITALIA | political | 40.0 | 6 | 6/6 | Train | |
| 36 | POL-2020-CHILE_CONSTITUTIONAL_REF | political | 78.3 | 6 | 6/6 | Train | |
| 37 | POL-2020-PRESIDENZIALI_USA | political | 51.3 | 7 | 7/7 | Train | |
| 38 | POL-2022-PRESIDENZIALI_BRASILE | political | 50.9 | 6 | 6/6 | Train | |
| 39 | SOC-2017-AUSTRALIA_SAME_SEX_MARRIAGE | social | 61.6 | 7 | 7/7 | Train | |
| 40 | TECH-2017-NET_NEUTRALITY_REPEAL_US | technology | 83.0 | 5 | 1/5 | Test | |
| 41 | TECH-2018-GDPR_ADOPTION | technology | 67.0 | 7 | 0/7 | Train | |
| 42 | TECH-2020-TIKTOK_US_BAN_DEBATE | technology | 60.0 | 6 | 1/6 | Train | |

Total: 42 scenarios. 34 train / 8 test. 10 domains. 1 flagged NEEDS_VERIFICATION (Archegos). 14 enriched with financial market data (all FIN-* and CORP-* scenarios). Verified polling available for 31 scenarios (polling from LLM estimation where sample_size is not verified). Median scenario length: 6 rounds.

Note: Scenario IDs are abbreviated in the table for readability. Full IDs are available in the repository. The "Polls" column shows rounds with verified sample sizes vs. total rounds. Ground truth sources include official election results, certified referendum outcomes, and verified survey data.

---

## Appendix D: Reproducibility of the v2.5 Benchmark Layer

All numerical results in Sections 6.6–6.8 are produced by a self-contained Python module released alongside the codebase. This appendix documents the package, its command-line interface, and its automated test suite so that the reference distribution can be rederived from scratch and any modification to the sim can be re-evaluated against it.

### D.1 Module Layout

The `benchmarks/` package contains six first-class modules:

| File | Responsibility |
|------|----------------|
| `forecasters.py` | Closed-form null baselines: naive persistence, running mean, OLS linear trend, AR(1). Exposes `generate_baseline_trajectory` for one-step-ahead rolling forecasts and `forecast_errors` / `rmse` helpers. |
| `diebold_mariano.py` | DM statistic with the Harvey–Leybourne–Newbold correction. Student-$t$ p-values computed via the regularized incomplete beta identity $P(T > x) = \tfrac{1}{2} I_{\nu/(\nu+x^2)}(\nu/2, 1/2)$ with Lentz's continued-fraction expansion — no SciPy dependency. |
| `coverage.py` | Empirical coverage of an interval sequence against realized values, plus a nonparametric bootstrap confidence band on the coverage statistic itself. Includes a `coverage_from_quantiles` helper for ensemble Monte Carlo intervals. |
| `residual_ci.py` | Residual-bootstrap predictive intervals that turn any point-forecaster into a calibrated interval forecaster, enabling the model-free coverage check of Section 6.7. |
| `scenario_matrix.py` | Enumeration of the $7 \times 5 \times 4$ scenario-diversity axis, `ScenarioCell` dataclass, and a `coverage_report` that flags missing axis values in any given corpus. |
| `historical.py` / `historical_runner.py` | Loader that normalizes the v1 and v2.2 empirical JSON scenarios into the common `support` / `signed_position` representation, derives tension from per-scenario volatility, maps ISO country codes to regions, and runs all baselines + DM tests + coverage + matrix report in one pass. |
| `runner.py` | Parallel runner for the deterministic-trajectory benchmark against the 7 scenarios stored in `frontend/public/data/scenario_*` (the sim-forecast case). |

### D.2 Command-Line Usage

The historical benchmark (Section 6.6–6.8) regenerates from scratch with:

```bash
python -m benchmarks.historical_runner \
    --out outputs/historical_benchmark.json \
    --markdown outputs/historical_benchmark.md
```

The deterministic-trajectory benchmark (sim-forecast case, for continuous regression testing) runs with:

```bash
python -m benchmarks \
    --out outputs/benchmark_report.json \
    --markdown outputs/benchmark_report.md
```

Both commands are zero-configuration on the public repository and write machine-readable JSON + human-readable Markdown reports.

### D.3 Automated Test Suite

The benchmark layer ships with 103 automated tests distributed across five files and four markers (`slow`, `integration`, `perf`, `stress`):

| Test file | Tests | Purpose |
|-----------|-------|---------|
| `test_benchmarks.py` | 18 | Unit tests for forecasters, DM, coverage, scenario matrix. |
| `test_benchmarks_integration.py` | 8 | End-to-end runner tests, including residual-bootstrap calibration on i.i.d. noise. |
| `test_performance_sla.py` | 5 | Throughput SLAs for DM, coverage, baselines, and the runner (defaults: 1000 DM tests in < 5 s, runner on 20 synthetic scenarios in < 3 s). All thresholds env-overridable via `DTS_SLA_*`. |
| `test_stress_scenarios.py` | 57 | Adversarial-signal stress tests (flat, monotone, alternating, spike, heavy-tail, near-constant), DM sign-consistency under $a \leftrightarrow b$ swap, coverage boundedness under random intervals, per-cell smoke tests on the full scenario matrix. |
| `test_historical_benchmark.py` | 15 | Unit + integration tests for the empirical loader (including v1/v2.2 duplicate resolution, `None`-field tolerance, and domain-alias collapse) and the aggregate runner. Includes a corpus smoke test that fails loudly if the matrix coverage regresses. |

Total runtime on a laptop-class machine: 2.94 seconds for the full 103-test suite, which is kept under 5 seconds so it can run on every commit without gating.

### D.4 Release Artifacts

Running the two commands above produces four reproducible artifacts shipped with the paper:

1. `outputs/historical_benchmark.json` — the full per-scenario × per-metric × per-baseline score matrix (43 × 2 × 4 = 344 score rows).
2. `outputs/historical_benchmark.md` — the human-readable summary from which Tables 15a, 15b, 16, and 17 are extracted verbatim.
3. `outputs/benchmark_report.json` — the deterministic-trajectory benchmark on the 7 scenarios currently shipped with the frontend, used as a continuous regression signal.
4. `outputs/benchmark_report.md` — the corresponding Markdown summary.

Any future modification to the simulator, the calibration pipeline, or the EnKF can be A/B-tested against the v2.5 reference distribution by running the two commands before and after the change. Disagreement on any DM cell at $p < 0.05$ constitutes a reproducibility-relevant regression.

---

## References

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337–351.

Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.

Deffuant, G., Neau, D., Amblard, F., & Weisbuch, G. (2000). Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3(01n04), 87–98.

DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345), 118–121.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

Evensen, G. (2003). The Ensemble Kalman Filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343–367.

Gao, C., Lan, X., Li, N., Yuan, Y., Ding, J., Zhou, Z., ... & Li, Y. (2023). Large language models empowered agent-based modeling and simulation: A survey and perspectives. *arXiv preprint arXiv:2312.11970*.

Grazzini, J., Richiardi, M. G., & Tsionas, M. (2017). Bayesian estimation of agent-based models. *Journal of Economic Dynamics and Control*, 77, 26–47.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: models, analysis and simulation. *Journal of Artificial Societies and Social Simulation*, 5(3).

Holley, R. A., & Liggett, T. M. (1975). Ergodic theorems for weakly interacting infinite systems and the voter model. *The Annals of Probability*, 3(4), 643–663.

Horton, J. J. (2023). Large language models as simulated economic agents: What can we learn from homo silicus? *NBER Working Paper 31122*.

Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B*, 63(3), 425–464.

Mistry, D., Litvinova, M., Chinazzi, M., et al. (2021). Inferring high-resolution human mixing patterns for disease modeling. *Nature Communications*, 12(1), 323.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*.

Park, J. S., Popowski, L., Cai, C., Morris, M. R., Liang, P., & Bernstein, M. S. (2022). Social simulacra: Creating populated prototypes for social computing systems. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology*.

Platt, D. (2020). A comparison of economic agent-based model calibration methods. *Journal of Economic Dynamics and Control*, 113, 103859.

Reich, S., & Cotter, C. (2015). *Probabilistic Forecasting and Bayesian Data Assimilation*. Cambridge University Press.

Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., ... & Tarantola, S. (2008). *Global Sensitivity Analysis: The Primer*. John Wiley & Sons.

Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1–3), 271–280.

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv preprint arXiv:1804.06788*.

Shiller, R. J. (2015). *Irrational Exuberance* (3rd ed.). Princeton University Press.

Tetlock, P. E. (2015). *Superforecasting: The Art and Science of Prediction*. Crown.

Thiele, J. C., Kurth, W., & Grimm, V. (2014). Facilitating parameter estimation and sensitivity analysis of agent-based models: A cookbook using NetLogo and R. *Journal of Artificial Societies and Social Simulation*, 17(3), 11.

Yahoo Finance. (2024). *yfinance: Yahoo! Finance market data downloader* [Python package]. https://github.com/ranaroussi/yfinance
