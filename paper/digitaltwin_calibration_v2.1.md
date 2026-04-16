# DigitalTwinSim: Bayesian Calibration and Online Data Assimilation for LLM-Agent Opinion Dynamics Simulation

**Alberto Giovanni Gerli**

---

## Abstract

We present DigitalTwinSim, a computational framework that combines large language model (LLM)-driven agent-based simulation with Bayesian calibration and online data assimilation for modeling public opinion dynamics. The system addresses a fundamental limitation of LLM-agent simulations: their outputs are stochastic, structurally misspecified, and uncalibrated against empirical data. We formulate opinion dynamics as a force-based system with five competing mechanisms—direct LLM influence, social conformity, herd behavior, anchor rigidity, and exogenous shocks—combined through a gauge-fixed softmax mixture. A three-level hierarchical Bayesian model (global, domain, scenario) with explicit readout discrepancy is calibrated via stochastic variational inference (SVI) on 42 empirical scenarios spanning 10 domains. The calibrated model achieves 12.6 pp mean absolute error on verified held-out scenarios (19.2 pp on the full test set including one ambiguous scenario) with 85.7% coverage of 90% credible intervals. Credible intervals are computed in logit space and back-transformed via sigmoid, guaranteeing bounds within [0, 100] by construction. Simulation-based calibration confirms posterior validity (6/6 parameters pass KS uniformity, all p > 0.20). Variance-based global sensitivity analysis identifies herd behavior (S_T = 0.55) and anchor rigidity (S_T = 0.45) as the dominant mechanisms, with their interaction (S_2 = 0.094) accounting for most nonlinear output variance—justifying the decision to freeze four mechanistic parameters. We extend the framework with an Ensemble Kalman Filter (EnKF) for online data assimilation, enabling live updating of both model parameters and agent states as streaming observations arrive. On the Brexit referendum scenario, the EnKF reduces prediction error to 1.8 pp with six polling observations—a 77% improvement over the last-available-poll baseline—while maintaining calibrated uncertainty. We discuss limitations including systematic over-prediction in financial domains (mean |b_s| = 0.74 in logit space), sensitivity of the frozen citizen step size parameter (up to 7.9 pp MAE variation under ±40% perturbation), and the irreducible gap between LLM-generated narratives and real-world opinion formation.

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

The framework transforms LLM-agent simulations from narrative-generation tools into quantitative digital twins: calibrated probabilistic models that can be validated, updated with data, and used for counterfactual analysis.

### 1.1 Paper Organization

Section 2 reviews related work. Section 3 presents the opinion dynamics model and its five force terms. Section 4 describes the hierarchical Bayesian calibration framework. Section 5 presents calibration results on 42 empirical scenarios. Section 6 covers validation through SBC, sensitivity analysis, and robustness checks. Section 7 introduces the EnKF online assimilation module. Section 8 discusses limitations and future work. Section 9 concludes. Appendix A provides implementation details. Appendix B describes the preliminary regime-switching extension. Appendix C lists the full scenario dataset.

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

[FIGURE 1: System architecture. Left: LLM generates per-agent behavioral shifts Δ^{LLM} and event narratives. Center: Five-force opinion dynamics model with gauge-fixed softmax mixing. Right: Observation model (BetaBinomial / Normal) connects simulated trajectories to empirical data. Bottom: Hierarchical Bayesian calibration with three levels (global → domain → scenario) and explicit readout discrepancy (b_d, b_s). Arrow from calibrated posterior to EnKF initialization for online mode.]

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

$$\theta_s \sim \mathcal{N}(\mu_{d(s)}, \text{diag}(\sigma_{\text{global}}^2)) \quad \text{(Eq. 15)}$$

where θ_s ∈ ℝ⁴ are the four softmax weights used for simulation.

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

The scenario-level parameters incorporate observable scenario characteristics through a linear regression:

$$\theta_s = \mu_{d(s)} + B \cdot x_s + \varepsilon_s, \quad \varepsilon_s \sim \mathcal{N}(0, \text{diag}(\sigma_d^2)) \quad \text{(Eq. 23)}$$

where B ∈ ℝ^{4×5} is a regression matrix and x_s ∈ ℝ⁵ are normalized scenario covariates:

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

$$\sigma_{b,\text{between}} = 0.115 \quad [0.073, 0.173] \quad \text{(Eq. 24)}$$
$$\sigma_{b,\text{within}} = 0.558 \quad [0.436, 0.696] \quad \text{(Eq. 25)}$$

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
| Archegos Capital* | Financial | 35.0 | 100.0 ± 3.1 | +65.0 | [100.0, 100.0] | ✗ |
| Greek Bailout | Political | 38.7 | 65.3 ± 11.7 | +26.6 | [42.7, 84.5] | ✗ |
| Net Neutrality | Technology | 83.0 | 66.3 ± 12.6 | -16.7 | [47.3, 82.8] | ✓ |
| French Election | Political | 66.1 | 51.6 ± 13.7 | -14.5 | [29.7, 72.8] | ✓ |
| COVID Vax (IT) | Pub. Health | 80.0 | 70.8 ± 11.2 | -9.2 | [52.1, 88.7] | ✓ |
| Tesla Cybertruck | Commercial | 62.0 | 54.1 ± 14.7 | -7.9 | [26.1, 72.6] | ✓ |
| Amazon HQ2 | Corporate | 56.0 | 63.3 ± 12.9 | +7.3 | [48.7, 80.6] | ✓ |
| Turkish Ref. | Political | 51.4 | 57.5 ± 13.0 | +6.1 | [37.4, 75.7] | ✓ |

*NEEDS_VERIFICATION — excluded from primary metrics. CI computed in logit space and back-transformed; note Archegos CI collapses to [100, 100] because the predicted mean saturates at the sigmoid boundary.

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
| MAE (test) | 14.8 pp (on synthetic) | 12.6 pp (on empirical) |
| Epistemological status | Tool demonstration | Calibrated probabilistic model |

---

## 6. Validation

### 6.1 Simulation-Based Calibration

Simulation-based calibration (SBC; Talts et al., 2018) verifies that the inference procedure recovers known parameters. We generate 100 synthetic datasets from the prior, run NUTS inference (200 warmup + 200 samples) on each, and test whether the posterior rank statistics are uniformly distributed.

**Table 9: SBC Results**

| Parameter | N | KS Statistic | p-value | Verdict |
|---|---|---|---|---|
| α_herd | 100 | 0.070 | 0.685 | PASS |
| α_anchor | 100 | 0.095 | 0.308 | PASS |
| α_social | 100 | 0.075 | 0.600 | PASS |
| α_event | 100 | 0.075 | 0.600 | PASS |
| τ_readout | 100 | 0.095 | 0.308 | PASS |
| σ_obs | 100 | 0.105 | 0.205 | PASS |

SBC confirms that the generative model is well-specified and that a gold-standard NUTS backend recovers correct posteriors: all 6 parameters pass the KS uniformity test (p > 0.20). However, SBC does not validate the SVI approximation used for the full empirical model, since SBC runs use NUTS.

[FIGURE 2: SBC rank histograms for all 6 parameters. Each histogram should show approximately uniform distribution across 10 bins. All parameters pass the KS test for uniformity.]

The SBC configuration uses 10 agents and 7 rounds per synthetic scenario with Normal(0, 0.3) priors. Total runtime: 297.6 seconds (approximately 3 seconds per instance).

### 6.2 SVI vs NUTS Comparison

To assess variational approximation quality, we compare SVI and NUTS posteriors on a 4-scenario subset spanning two domains.

**Table 10: SVI vs NUTS Posterior Comparison (4-scenario subset)**

| Parameter | SVI mean ± std | NUTS mean ± std | |Δμ|/σ_NUTS | Compatible |
|---|---|---|---|---|
| α_herd | -0.500 ± 0.085 | -0.383 ± 0.796 | 0.148 | Yes |
| α_anchor | +0.905 ± 0.065 | +0.604 ± 0.856 | 0.352 | Yes |
| α_social | -0.997 ± 0.105 | -0.350 ± 0.763 | 0.848 | No |
| α_event | -1.752 ± 0.073 | -0.137 ± 0.946 | 1.707 | No |

*SVI: 1000 steps, Adam(lr = 0.01), AutoLowRankMVN guide. NUTS: 200 warmup + 200 samples, max_tree_depth = 8.*

The two dominant parameters show good agreement (α_herd: |Δμ|/σ_NUTS = 0.15; α_anchor: 0.35). The weaker parameters show larger divergence (α_social: 0.85; α_event: 1.71), consistent with the known tendency of mean-field variational families to concentrate posterior mass. The SVI standard deviations are 5–13× narrower than the NUTS estimates, indicating that the variational approximation substantially underestimates posterior uncertainty on the less-constrained parameters.

This result has two implications. First, the posterior means for α_herd and α_anchor—which together dominate the force mixture (S_T = 0.55 and 0.45)—are reliably estimated by SVI. Second, uncertainty estimates on α_social and α_event may be underestimated by the variational approximation. Users requiring conservative uncertainty bounds should consider post-hoc inflation of the variational posterior or short NUTS chains initialized from the SVI optimum.

### 6.3 Global Sensitivity Analysis

Variance-based global sensitivity analysis (Sobol, 2001) decomposes output variance into contributions from individual parameters and their interactions. We evaluate all 8 model parameters using N = 1024 Saltelli samples (18,432 total simulator evaluations, n = 30 agents, 7 rounds).

**Table 11: Sobol Sensitivity Indices**

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

2. **Calibrable/frozen partition justified.** The four frozen parameters (λ_elite, λ_citizen, θ_herd, δ_drift) all have S_1 < 0.01, confirming that they can be fixed at default values without significant loss of model expressiveness.

3. **Sum diagnostics.** ΣS_1 = 0.70 indicates that approximately 30% of output variance arises from parameter interactions. ΣS_T = 1.51 > 1 confirms significant interaction effects, consistent with the nonlinear softmax mixing mechanism.

[FIGURE 3: Sobol indices bar chart. Two panels: S_1 (main effects) and S_T (total effects) for all 8 parameters.]

### 6.4 Step Size Sensitivity

Sobol analysis identifies λ_citizen as having non-negligible total effect (S_T = 0.121) despite near-zero main effect (S_1 = 0.002), warranting direct investigation. A perturbation analysis varies λ_citizen by ±40% around the default (0.25), measuring MAE impact on 5 representative scenarios.

**Table 12: λ_citizen Sensitivity Analysis**

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

**Table 13: 22-Scenario vs. 42-Scenario Calibration**

| Metric | 22-Scenario | 42-Scenario | Δ |
|---|---|---|---|
| N (train / test) | 16 / 6 | 34 / 8 | +18 / +2 |
| MAE (test) | 11.7 pp | 19.2 pp | +7.4 |
| RMSE (test) | 14.7 | 26.6 | +11.9 |
| Coverage 90% (test) | 83.3% | 75.0% | -8.3% |
| MAE (train) | 16.3 pp | 14.3 pp | -2.0 |
| Coverage 90% (train) | 68.8% | 79.4% | +10.6% |

The expanded dataset improves training fit (MAE 16.3 → 14.3 pp, coverage 68.8% → 79.4%) but shows degraded test performance, largely driven by the inclusion of more challenging financial and corporate scenarios in the test set. When the NEEDS_VERIFICATION scenario is excluded, test MAE drops to 12.6 pp—comparable to the 22-scenario result (11.7 pp). This suggests that the model's performance ceiling is scenario-dependent rather than data-limited.

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

$$\mathbf{x}_j = [\boldsymbol{\theta}_j, \mathbf{z}_j]^\top \quad \text{(Eq. 26)}$$

where $\boldsymbol{\theta}_j = [\alpha_h, \alpha_a, \alpha_s, \alpha_e]$ are the four calibrable parameters and $\mathbf{z}_j = [p_1, ..., p_n]$ are the n agent positions. The state dimension is 4 + n.

**Forecast step.** At each round t, the forecast propagates parameters as a random walk and states through the JAX simulator:

$$\boldsymbol{\theta}_j^f(t+1) = \boldsymbol{\theta}_j^a(t) + \boldsymbol{\eta}_j^\theta, \quad \boldsymbol{\eta}_j^\theta \sim \mathcal{N}(0, Q_\theta I) \quad \text{(Eq. 27)}$$

$$\mathbf{z}_j^f(t+1) = \text{step\_round}(\mathbf{z}_j^a(t), \boldsymbol{\theta}_j^f(t+1), \text{event}_t) + \boldsymbol{\eta}_j^z, \quad \boldsymbol{\eta}_j^z \sim \mathcal{N}(0, Q_z I) \quad \text{(Eq. 28)}$$

where Q_θ = 0.01 controls parameter exploration speed and Q_z = 0.005 adds stochastic perturbation to agent positions. The forecast is parallelized over the ensemble using `jax.vmap`.

**Update step.** When an observation y_obs with variance R is available:

1. Compute ensemble predictions: $\hat{y}_j = h(\mathbf{z}_j^f)$ where h(·) is the readout function (Section 3.6).

2. Compute anomalies: $\mathbf{X}^{\text{anom}} = \mathbf{X}^f - \bar{\mathbf{X}}^f$, $\hat{y}^{\text{anom}} = \hat{y} - \bar{\hat{y}}$.

3. Cross-covariance and innovation variance:

$$\mathbf{P}_{xh} = \frac{1}{E-1} \mathbf{X}^{\text{anom}} (\hat{y}^{\text{anom}})^\top, \quad P_{hh} = \frac{1}{E-1} \|\hat{y}^{\text{anom}}\|^2 \quad \text{(Eq. 29)}$$

4. Kalman gain: $\mathbf{K} = \mathbf{P}_{xh} / (P_{hh} + R)$ (Eq. 30)

5. Perturbed update:

$$\mathbf{x}_j^a = \mathbf{x}_j^f + \mathbf{K} \cdot (y_{\text{obs}} + \epsilon_j - \hat{y}_j), \quad \epsilon_j \sim \mathcal{N}(0, R) \quad \text{(Eq. 31)}$$

6. Multiplicative inflation to prevent ensemble collapse:

$$\mathbf{x}_j^a \leftarrow \bar{\mathbf{x}}^a + \gamma (\mathbf{x}_j^a - \bar{\mathbf{x}}^a), \quad \gamma = 1.02 \quad \text{(Eq. 32)}$$

### 7.3 Observation Adapters

The EnKF supports three observation types through adapter classes:

- **PollingSurvey**: pro_pct ∈ [0, 100] with variance inversely proportional to sample size: R = pro_pct · (100 - pro_pct) / sample_size.
- **SentimentSignal**: Maps sentiment scores to approximate pro_pct with configurable noise floor.
- **OfficialResult**: Final certified outcome with minimal observation noise (R = 1.0).

### 7.4 Brexit Case Study

We demonstrate the EnKF on the Brexit referendum scenario (ground truth: 51.89% Leave). Six polling observations are available across the 9-round simulation, with polls ranging from 41.0% to 44.0% (consistent with the well-documented polling bias that underestimated Leave support).

**Table 14a: EnKF Brexit — Polling Observations**

| Round | Poll (%) | Sample Size |
|---|---|---|
| 1 | 41.0 | 1000 |
| 2 | 42.0 | 1000 |
| 3 | 43.0 | 1000 |
| 4 | 43.0 | 1000 |
| 5 | 44.0 | 1000 |
| 6 | 44.0 | 1000 |

The EnKF ingests these observations while propagating the opinion dynamics model forward. Despite the polls systematically underestimating the final outcome by ~8 pp, the dynamics model—which captures mechanisms such as late-breaking opinion shifts and shy voter effects through the herd and anchor forces—produces a final prediction of 50.1%, within 1.8 pp of the ground truth.

**Table 14b: EnKF vs Baselines**

| Method | Final Prediction (%) | Error (pp) | Uses Dynamics | Updates Params |
|---|---|---|---|---|
| Last available poll | 44.0 | 7.9 | No | No |
| Running poll average | 42.8 | 9.1 | No | No |
| EnKF (state only, θ fixed) | 50.1 | 1.8 | Yes | No |
| EnKF (state + params) | 50.1 | 1.8 | Yes | Yes |

The EnKF with dynamics model reduces prediction error by 77% compared to the last-available-poll baseline (1.8 pp vs 7.9 pp) and 80% compared to the running average (1.8 pp vs 9.1 pp). The dynamics model—even with frozen parameters—adds substantial value by propagating opinion evolution between observation points and capturing mechanisms that simple extrapolation misses.

State-only and state+params variants converge to the same prediction with 6 observations, which is expected: with abundant data, the state update dominates the parameter update. The value of joint parameter-state estimation would be more evident with sparse observations (1–2 polls), where the calibrated prior on θ provides more leverage.

It is important to note that the 1.8 pp EnKF error is achieved *with streaming polling data*—it is not directly comparable to the offline-only MAE of 12.6 pp, which uses no scenario-specific observations. The EnKF demonstrates what becomes possible when the digital twin receives live data, not a claim about the base model's accuracy.

[FIGURE 4: EnKF convergence on Brexit. Top panel: pro% prediction (mean ± 90% CI) over 9 rounds, with vertical markers at observation rounds. Horizontal line at GT = 51.89%. Bottom panel: CI width over rounds, showing contraction at each observation.]

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

**Financial domain performance.** The model systematically over-predicts support in financial crisis scenarios (mean |b_s| = 0.744 in logit space). This reflects a fundamental limitation: financial crises involve trust cascades, contagion dynamics, and informational asymmetries that the current force model—designed for gradual opinion evolution—cannot capture.

**Variational approximation quality.** The SVI posterior agrees with NUTS on the dominant parameters (α_herd, α_anchor: |Δμ|/σ_NUTS < 0.4) but shows concentration bias on weaker parameters (α_social: 0.85; α_event: 1.71). Uncertainty estimates on α_social and α_event are likely underestimated. This is a known limitation of mean-field variational families (Blei et al., 2017) that trades exactness for computational tractability.

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

---

## 9. Conclusion

We have presented DigitalTwinSim, a framework for transforming LLM-agent opinion simulations from uncalibrated narrative generators into quantitative digital twins with calibrated uncertainty. The key insight is that LLM-agent simulations, despite their structural misspecification, contain learnable signal about opinion dynamics mechanisms—but only when combined with principled Bayesian calibration and explicit model discrepancy.

The calibrated model achieves 12.6 pp MAE on verified held-out scenarios (19.2 pp on the full test set including one ambiguous scenario) with 85.7% coverage of 90% credible intervals. Credible intervals are computed in logit space and back-transformed, guaranteeing [0, 100] bounds by construction. Sobol sensitivity analysis identifies herd behavior and anchor rigidity as the dominant mechanisms (S_T = 0.55 and 0.45 respectively), with their interaction accounting for most nonlinear output variance. Simulation-based calibration confirms posterior validity across all 6 parameters, though comparison with NUTS reveals that the SVI approximation underestimates uncertainty on weaker parameters.

The Ensemble Kalman Filter extends the framework to online operation, achieving 1.8 pp error on the Brexit scenario with six streaming polling observations—a 77% improvement over the last-available-poll baseline. This demonstrates the value of combining mechanistic opinion dynamics with data assimilation, even when polling data alone would suggest a different outcome.

The framework has known limitations: sensitivity of the frozen λ_citizen parameter (up to 7.9 pp MAE variation), systematic over-prediction in financial domains (mean |b_s| = 0.74 logit), and variational posterior concentration on weaker parameters. These are reported transparently and motivate concrete next steps: promoting λ_citizen to calibrable, running NUTS refinement of the SVI posterior, and calibrating the regime-switching extension for crisis dynamics.

The framework's modular design—force-based dynamics, hierarchical Bayesian calibration, online assimilation, regime switching—allows each component to be extended independently. The immediate priorities are λ_citizen calibration and NUTS-initialized uncertainty refinement, both of which can be pursued without modifying the core architecture.

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

$$\text{logit}(p) = \beta_0 + \beta_{\text{shock}} (m_t - \tau_{\text{shock}}) + \beta_{\text{vel}} (v_t - \tau_{\text{vel}}) + \beta_{\text{trust}} (1 - \text{trust}) + \beta_{\text{recovery}} \cdot r_t + \beta_{\text{momentum}} \cdot p_{t-1} \quad \text{(Eq. B.2)}$$

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

The 42 empirical scenarios spanning 10 domains are available in the project repository under `calibration/empirical/scenarios/`. Each scenario includes agent definitions, event timelines, polling trajectories (where available), and ground-truth final outcomes with provenance metadata.

---

## References

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337–351.

Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.

Deffuant, G., Neau, D., Amblard, F., & Weisbuch, G. (2000). Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3(01n04), 87–98.

DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345), 118–121.

Evensen, G. (2003). The Ensemble Kalman Filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343–367.

Gao, C., Lan, X., Li, N., Yuan, Y., Ding, J., Zhou, Z., ... & Li, Y. (2023). Large language models empowered agent-based modeling and simulation: A survey and perspectives. *arXiv preprint arXiv:2312.11970*.

Grazzini, J., Richiardi, M. G., & Tsionas, M. (2017). Bayesian estimation of agent-based models. *Journal of Economic Dynamics and Control*, 77, 26–47.

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

Thiele, J. C., Kurth, W., & Grimm, V. (2014). Facilitating parameter estimation and sensitivity analysis of agent-based models: A cookbook using NetLogo and R. *Journal of Artificial Societies and Social Simulation*, 17(3), 11.
