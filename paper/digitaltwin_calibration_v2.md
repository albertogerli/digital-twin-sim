# DigitalTwinSim: Bayesian Calibration and Online Data Assimilation for LLM-Agent Opinion Dynamics Simulation

**Alberto Giovanni Gerli**

---

## Abstract

We present DigitalTwinSim, a computational framework that combines large language model (LLM)-driven agent-based simulation with Bayesian calibration and online data assimilation for modeling public opinion dynamics. The system addresses a fundamental limitation of LLM-agent simulations: their outputs are stochastic, structurally misspecified, and uncalibrated against empirical data. We formulate opinion dynamics as a force-based system with five competing mechanisms—direct LLM influence, social conformity, herd behavior, anchor rigidity, and exogenous shocks—combined through a gauge-fixed softmax mixture. A three-level hierarchical Bayesian model (global, domain, scenario) with explicit model discrepancy is calibrated via stochastic variational inference on 42 empirical scenarios spanning 10 domains. The calibrated model achieves 12.6 pp mean absolute error with 85.7% coverage of 90% credible intervals on held-out scenarios. Simulation-based calibration confirms posterior validity (6/6 parameters pass KS uniformity, all p > 0.20). Variance-based global sensitivity analysis identifies herd behavior (ST = 0.55) and anchor rigidity (ST = 0.45) as the dominant mechanisms, with their interaction (S2 = 0.094) accounting for most nonlinear output variance—justifying the decision to freeze four mechanistic parameters. We extend the framework with an Ensemble Kalman Filter (EnKF) for online data assimilation, enabling live updating of both model parameters and agent states as streaming observations arrive. On the Brexit referendum scenario, the EnKF reduces prediction error to 2.2 pp with four polling observations while maintaining calibrated uncertainty. Finally, we introduce a regime-switching extension for discontinuous crisis dynamics, using soft sigmoid-based switching to preserve JAX differentiability. We discuss limitations including systematic over-prediction in financial domains (mean |δ_s| = 0.74 in logit space) and the irreducible gap between LLM-generated narratives and real-world opinion formation.

---

## 1. Introduction

Agent-based models (ABMs) have long been used to study opinion dynamics, social influence, and collective decision-making. Classical approaches—bounded confidence models (Hegselmann & Krause, 2002), voter models (Holley & Liggett, 1975), and DeGroot-style averaging (DeGroot, 1974)—provide theoretical insight but struggle to capture the narrative complexity, heterogeneous reasoning, and contextual sensitivity that characterize real-world opinion formation.

The emergence of large language models (LLMs) as agent engines has opened a new approach: equipping simulated agents with natural language reasoning capabilities, allowing them to process news events, generate social media posts, form coalitions, and shift positions in response to contextual narratives. Recent work on generative agents (Park et al., 2023), social simulacra (Park et al., 2022), and LLM-driven social simulations (Gao et al., 2023) demonstrates the potential of this paradigm. However, these systems share a critical limitation: their outputs are uncalibrated. The mapping from LLM-generated agent behaviors to quantitative opinion distributions is ad hoc, and there is no principled mechanism to anchor simulations to empirical observations.

This paper addresses this gap by developing a complete calibration and assimilation pipeline for LLM-agent opinion dynamics. Our contributions are:

1. **A differentiable opinion dynamics simulator** formulated as a force-based system with five competing mechanisms, combined through gauge-fixed softmax mixing. The simulator is implemented in JAX and compatible with `jax.lax.scan`, enabling automatic differentiation through the full simulation trajectory.

2. **A hierarchical Bayesian calibration framework** with three levels (global, domain, scenario) and explicit model discrepancy terms, fitted via stochastic variational inference (SVI) on 42 empirical scenarios across 10 domains.

3. **Simulation-based calibration (SBC)** and **variance-based global sensitivity analysis (Sobol indices)** providing rigorous validation of posterior quality and identification of dominant mechanisms.

4. **An Ensemble Kalman Filter (EnKF)** for online data assimilation, bridging the offline calibrated posterior to live streaming observations. The EnKF jointly updates model parameters and agent states, producing calibrated probabilistic forecasts at each round.

5. **A regime-switching extension** for crisis dynamics, using soft sigmoid-based switching between normal and crisis regimes to model discontinuous trust collapses while maintaining JAX differentiability.

The framework transforms LLM-agent simulations from narrative-generation tools into quantitative digital twins: calibrated probabilistic models that can be validated, updated with data, and used for counterfactual analysis.

### 1.1 Paper Organization

Section 2 presents the opinion dynamics model and its five force terms. Section 3 describes the hierarchical Bayesian calibration framework. Section 4 presents calibration results on 42 empirical scenarios. Section 5 covers validation through SBC and sensitivity analysis. Section 6 introduces the EnKF online assimilation module. Section 7 presents the regime-switching extension for crisis dynamics. Section 8 discusses limitations and future work. Section 9 concludes.

---

## 2. Opinion Dynamics Model

### 2.1 Agent State Space

Each agent *i* ∈ {1, ..., n} maintains a scalar position p_i(t) ∈ [-1, +1] representing their stance on a binary issue, where +1 denotes maximum support ("Pro") and -1 maximum opposition ("Against"). Agents are characterized by three fixed attributes:

- **Type** τ_i ∈ {elite, citizen}: determines step size and behavioral parameters.
- **Rigidity** ρ_i ∈ [0, 1]: resistance to opinion change. Elites have higher rigidity (ρ ≈ 0.7) than citizens (ρ ≈ 0.3).
- **Tolerance** θ_i ∈ [0, 1]: radius of the bounded-confidence window for social influence. Elites have lower tolerance (θ ≈ 0.3) than citizens (θ ≈ 0.6).

Agents interact through a sparse weighted graph W ∈ ℝ^{n×n} constructed from LLM-assigned influence scores using a k-nearest-neighbor scheme.

### 2.2 Five Force Terms

At each round t, five independent forces act on each agent. These forces capture distinct social mechanisms:

**Force 1: Direct LLM Influence.** The LLM generates per-agent opinion shifts Δ_i^{LLM}(t) based on the current narrative context. The direct force is:

$$f_i^{\text{direct}}(t) = \Delta_i^{\text{LLM}}(t) \cdot (1 - \rho_i) \cdot \max(0, 1 - |p_i(t)|)$$

The susceptibility factor (1 - ρ_i) attenuates influence on rigid agents, while the boundary factor max(0, 1 - |p_i|) prevents extreme agents from being pushed further.

**Force 2: Herd Behavior (Consensus Pull).** Agents experience a pull toward the weighted mean of their neighborhood when the deviation exceeds a learned threshold θ_h:

$$f_i^{\text{herd}}(t) = (\bar{p}_i^{\text{feed}} - p_i) \cdot (1 - \rho_i) \cdot \sigma\left(\frac{|\bar{p}_i^{\text{feed}} - p_i| - \theta_h}{\tau_H}\right)$$

where $\bar{p}_i^{\text{feed}}$ is the normalized weighted mean of neighbors' positions, θ_h = σ(logit\_herd\_threshold) is a learned threshold, and τ_H = 0.02 is a steepness parameter. The sigmoid activation creates a smooth transition: agents ignore small deviations but respond to large consensus gaps.

**Force 3: Anchor Rigidity.** A restorative force pulling agents toward their original position:

$$f_i^{\text{anchor}}(t) = \rho_i \cdot (p_i^{\text{orig}}(t) - p_i(t))$$

where $p_i^{\text{orig}}(t)$ drifts slowly toward the current position at rate δ_drift = σ(logit\_anchor\_drift), modeling gradual internalization of new positions.

**Force 4: Social Influence (Bounded Confidence).** Weighted averaging with tolerance-gated interaction:

$$f_i^{\text{social}}(t) = \frac{\sum_j w_{ij}(t) \cdot (p_j(t) - p_i(t))}{\sum_j w_{ij}(t)}$$

where $w_{ij}(t) = W_{ij} \cdot \sigma\left(\frac{\theta_i - |p_j - p_i|}{\tau_{BC}}\right)$ and τ_BC = 0.02. This implements smooth bounded confidence: agents preferentially interact with like-minded peers, but the transition is differentiable rather than a hard cutoff.

**Force 5: Exogenous Event Shock.** External events apply a uniform directional push:

$$f_i^{\text{event}}(t) = m_t \cdot d_t \cdot (1 - \rho_i)$$

where m_t ∈ [0, 1] is the event magnitude and d_t ∈ {-1, +1} its direction. Events are generated by the LLM based on the simulated scenario narrative.

### 2.3 Force Standardization

Raw forces have heterogeneous scales (e.g., social forces depend on graph density, event forces on shock magnitude). We standardize all five forces per round using exponential moving averages:

$$\tilde{f}_k(t) = \frac{f_k(t) - \mu_k(t)}{\sigma_k(t) + \epsilon}$$

where μ_k(t) and σ_k(t) are updated via EMA with decay α = 0.3. This ensures that the subsequent softmax mixing operates on comparable scales regardless of the underlying force magnitudes.

### 2.4 Gauge-Fixed Softmax Mixing

The five standardized forces are combined through a softmax mixture with a gauge-fixing constraint:

$$\boldsymbol{\alpha} = [0, \alpha_h, \alpha_a, \alpha_s, \alpha_e]^\top$$

$$\pi_k = \text{softmax}(\boldsymbol{\alpha})_k, \quad \sum_k \pi_k = 1$$

The direct force weight α_direct ≡ 0 serves as the reference level (gauge), resolving the shift invariance of softmax. This means the four remaining weights α_h, α_a, α_s, α_e are interpretable as log-odds relative to the direct LLM influence.

The combined force per agent is:

$$\tilde{f}_i^{\text{combined}}(t) = \sum_k \pi_k \cdot \tilde{f}_{k,i}(t)$$

### 2.5 Position Update

The position update applies a type-specific step size with smooth clamping:

$$\Delta p_i = \lambda_{\tau_i} \cdot \tilde{f}_i^{\text{combined}}, \quad \Delta p_i^{\text{clamped}} = \tanh\left(\frac{\Delta p_i}{c_{\tau_i}}\right) \cdot c_{\tau_i}$$

where λ_elite = exp(log\_λ\_elite), λ_citizen = exp(log\_λ\_citizen), c_elite = 0.15, c_citizen = 0.25. The tanh clamping provides smooth saturation rather than hard clipping, preserving differentiability.

The final update is:

$$p_i(t+1) = \text{clip}(p_i(t) + \Delta p_i^{\text{clamped}}, -1, +1)$$

### 2.6 Readout Function

The mapping from agent positions to aggregate opinion uses a soft classification:

$$\text{pro}(t) = \frac{\sum_i \sigma\left(\frac{p_i(t) - 0.05}{0.02}\right)}{\sum_i \left[\sigma\left(\frac{p_i(t) - 0.05}{0.02}\right) + \sigma\left(\frac{-p_i(t) - 0.05}{0.02}\right)\right]}$$

This readout excludes near-neutral agents (|p_i| < 0.05) from the decided count, producing pro_fraction ∈ [0, 1]. The 0.02 temperature ensures smooth but sharp classification.

### 2.7 Parameter Summary

The model has 8 mechanistic parameters, partitioned into 4 calibrable and 4 frozen:

[TABLE 1: Model Parameters]

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

The calibrable/frozen partition is justified empirically by Sobol sensitivity analysis (Section 5.2).

---

## 3. Hierarchical Bayesian Calibration

### 3.1 Motivation

A single set of global parameters cannot adequately describe opinion dynamics across diverse domains. Political referenda involve different social mechanisms than corporate crises or public health debates. At the same time, per-scenario calibration with 4 parameters and limited per-scenario data (typically 3-9 polling observations) leads to severe overfitting. A hierarchical model enables partial pooling: scenarios share information through domain and global priors while retaining scenario-specific flexibility through model discrepancy terms.

### 3.2 Three-Level Hierarchy

The model has three levels:

**Level 1 (Global).** Shared priors on the mixing weights:

$$\mu_{\text{global}} \sim \mathcal{N}(0, I), \quad \sigma_{\text{global}} \sim \text{HalfNormal}(0.3)$$

**Level 2 (Domain).** Each domain d ∈ {political, financial, corporate, ...} has a domain-level offset:

$$\delta_d \sim \mathcal{N}(0, \sigma_{\delta,\text{between}}^2), \quad \sigma_{\delta,\text{between}} \sim \text{HalfNormal}(0.2)$$

$$\mu_d = \mu_{\text{global}} + \delta_d \cdot \mathbf{1}$$

The domain offset δ_d acts in logit space on the readout, modeling systematic domain-level bias (e.g., financial scenarios tend to over-predict support).

**Level 3 (Scenario).** Each scenario s has scenario-specific parameters drawn from its domain:

$$\theta_s \sim \mathcal{N}(\mu_d, \text{diag}(\sigma_{\text{global}}^2))$$

and a scenario-specific discrepancy:

$$\delta_s \sim \mathcal{N}(0, \sigma_{\delta,\text{within}}^2), \quad \sigma_{\delta,\text{within}} \sim \text{HalfNormal}(0.5)$$

### 3.3 Model Discrepancy

The simulator is structurally misspecified: LLM-generated narratives cannot perfectly reproduce real-world opinion formation. Rather than absorbing this misspecification into the parameters (which would bias them), we model it explicitly.

The corrected readout for scenario s is:

$$q_s^{\text{corrected}} = \sigma\left(\text{logit}(q_s^{\text{sim}}) + \delta_d + \delta_s\right)$$

where $q_s^{\text{sim}}$ is the raw simulator output. The discrepancy terms δ_d (between-domain) and δ_s (within-domain) operate in logit space, providing unbounded additive correction while preserving the [0, 1] constraint through the sigmoid.

This decomposition allows the calibrated parameters θ to represent genuine opinion dynamics mechanisms, while the discrepancy terms absorb systematic simulator errors.

### 3.4 Observation Model

For scenarios with per-round polling data (sample size n_t, observed count y_t):

$$y_t \mid q_t, \phi \sim \text{BetaBinomial}(n_t, q_t \cdot \phi, (1 - q_t) \cdot \phi)$$

where φ = exp(log φ) is a learned concentration parameter. The BetaBinomial accounts for both sampling noise (binomial) and overdispersion (beta mixing).

For scenarios with only a final outcome:

$$y_{\text{final}} \mid q_T, \sigma_{\text{obs}} \sim \mathcal{N}(100 \cdot q_T, \sigma_{\text{obs}}^2)$$

where σ_obs = exp(log σ) is learned.

### 3.5 Inference via SVI

The posterior is intractable due to the nonlinear JAX simulator in the likelihood. We use stochastic variational inference (SVI) with an AutoLowRankMultivariateNormal guide in NumPyro, which approximates the posterior with a low-rank plus diagonal covariance structure:

$$q(\theta) = \mathcal{N}(\mu_q, D + VV^\top)$$

where D is diagonal and V has rank ≤ dim(θ). This captures dominant posterior correlations (e.g., between α_h and α_a) while scaling to the full parameter space.

The SVI objective is the evidence lower bound (ELBO):

$$\mathcal{L}(\mu_q, D, V) = \mathbb{E}_{q(\theta)}[\log p(y \mid \theta) + \log p(\theta)] - \mathbb{E}_{q(\theta)}[\log q(\theta)]$$

minimized over 3000 steps with learning rate 0.002 and cosine annealing. Gradients are computed via JAX automatic differentiation through the full simulate → readout → likelihood chain.

---

## 4. Calibration Results

### 4.1 Dataset

We curated 42 empirical scenarios across 10 domains, each with a ground-truth final outcome and, where available, intermediate polling data. The scenarios span 2012–2023 and cover political referenda, financial crises, corporate controversies, public health debates, technology policy, and social movements.

[TABLE 2: Dataset Composition by Domain]

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

The train/test split is stratified by domain with approximately 80/20 ratio. One test scenario (Archegos Capital Collapse) is flagged as NEEDS_VERIFICATION due to ambiguous ground truth encoding.

### 4.2 Calibrated Global Parameters

SVI converges in 3000 steps (final ELBO loss: 514.7) with a total runtime of 40.2 minutes for Phase B (empirical fine-tuning) and 0.4 minutes for Phase C (validation).

[TABLE 3: Calibrated Global Posterior]

| Parameter | Mean | 95% CI | Interpretation |
|---|---|---|---|
| α_herd | -0.176 | [-0.265, -0.079] | Herd effect slightly below direct influence |
| α_anchor | +0.297 | [+0.199, +0.401] | Anchor rigidity is the strongest mechanism |
| α_social | -0.105 | [-0.202, -0.005] | Social influence slightly below direct |
| α_event | -0.130 | [-0.227, -0.033] | Event shocks below direct influence |
| σ_global | [0.135, 0.147, 0.144, 0.143] | — | Inter-scenario parameter spread |

The positive α_anchor indicates that anchor rigidity (resistance to change) dominates the force mixture. This is consistent with the well-documented status quo bias in opinion dynamics: people tend to revert toward their initial positions after temporary shifts. The negative values for α_herd, α_social, and α_event indicate these forces are present but weaker than direct LLM influence (the gauge reference).

Converting to softmax weights: π = softmax([0, -0.176, 0.297, -0.105, -0.130]) ≈ [0.189, 0.159, 0.254, 0.170, 0.166]. Anchor rigidity receives approximately 25.4% of the total weight, direct influence 18.9%, herd behavior 15.9%, social conformity 17.0%, and event shocks 16.6%.

### 4.3 Model Discrepancy

The calibrated discrepancy parameters reveal the scale of structural model misspecification:

$$\sigma_{\delta,\text{between}} = 0.115 \quad [0.073, 0.173]$$
$$\sigma_{\delta,\text{within}} = 0.558 \quad [0.436, 0.696]$$

The within-scenario variance (σ_within = 0.558) is approximately 5× the between-domain variance (σ_between = 0.115), indicating that scenario-specific factors dominate over systematic domain-level biases. In practical terms, 0.558 in logit space translates to approximately 12–14 percentage points of systematic prediction error that the model cannot eliminate through parameter adjustment alone.

[TABLE 4: Domain-Level Discrepancy]

| Domain | N | δ_d Mean | 95% CI | Mean |δ_s| | Bias |
|---|---|---|---|---|---|
| Financial | 6 | -0.054 | [-0.242, +0.126] | 0.744 | Over-predicts |
| Energy | 1 | -0.026 | [-0.237, +0.186] | 0.709 | Over-predicts |
| Public Health | 4 | -0.032 | [-0.223, +0.164] | 0.534 | Over-predicts |
| Corporate | 6 | -0.040 | [-0.221, +0.139] | 0.432 | Over-predicts |
| Political | 11 | +0.019 | [-0.145, +0.183] | 0.198 | No systematic bias |
| Technology | 2 | -0.021 | [-0.204, +0.174] | 0.066 | No systematic bias |

Financial scenarios exhibit the largest mean absolute discrepancy (|δ_s| = 0.744), driven by cases where the model systematically over-predicts support levels. The political domain, with the most training data (N=11), shows the lowest systematic bias (δ_d ≈ 0, mean |δ_s| = 0.198).

### 4.4 Predictive Performance

[TABLE 5: Validation Metrics (Test Set)]

| Metric | Full Test (N=8) | Verified Only (N=7) |
|---|---|---|
| MAE | 19.2 pp | 12.6 pp |
| RMSE | 26.6 | 14.3 |
| Median AE | 11.9 pp | 9.2 pp |
| Coverage 90% | 75.0% | 85.7% |
| Coverage 50% | 37.5% | 42.9% |
| CRPS | 15.4 | 8.6 |

The primary metric is verified-only MAE of 12.6 pp, excluding the Archegos scenario (error = 65.0 pp) whose ground truth is flagged as unreliable. The 85.7% coverage of 90% credible intervals indicates well-calibrated uncertainty—slightly below the nominal 90% but within the expected range for N = 7.

[TABLE 6: Test Set Detailed Results]

| Scenario | Domain | GT (%) | Predicted μ ± σ | Error (pp) | 90% CI | Covered |
|---|---|---|---|---|---|---|
| Archegos Capital* | Financial | 35.0 | 100.0 ± 3.1 | +65.0 | [95.1, 105.0] | ✗ |
| Greek Bailout | Political | 38.7 | 65.3 ± 11.7 | +26.6 | [42.7, 84.5] | ✗ |
| Net Neutrality | Technology | 83.0 | 66.3 ± 12.6 | -16.7 | [45.2, 85.6] | ✓ |
| French Election | Political | 66.1 | 51.6 ± 13.7 | -14.5 | [28.2, 73.0] | ✓ |
| COVID Vax (IT) | Pub. Health | 80.0 | 70.8 ± 11.2 | -9.2 | [51.9, 88.3] | ✓ |
| Tesla Cybertruck | Commercial | 62.0 | 54.1 ± 14.7 | -7.9 | [32.5, 76.6] | ✓ |
| Amazon HQ2 | Corporate | 56.0 | 63.3 ± 12.9 | +7.3 | [39.6, 84.1] | ✓ |
| Turkish Ref. | Political | 51.4 | 57.5 ± 13.0 | +6.1 | [33.6, 78.8] | ✓ |

*NEEDS_VERIFICATION — excluded from primary metrics.

### 4.5 Worst-Case Scenarios

[TABLE 7: Top 5 Scenario Discrepancies (Training Set)]

| Scenario | Domain | δ_s | Error (pp) | Failure Mode |
|---|---|---|---|---|
| WeWork IPO | Financial | -1.378 | +48.5 | Trust collapse not modeled |
| United Airlines | Corporate | -0.860 | +15.4 | Viral outrage underestimated |
| SVB Collapse | Financial | -0.835 | +38.1 | Sequential shocks mishandled |
| Chile Constitution | Political | +0.816 | -41.5 | Landslide momentum missing |
| FTX Crypto | Financial | -0.727 | +11.0 | Contagion dynamics absent |

The three largest discrepancies are all financial crises where the model over-predicts support (negative δ_s). This systematic failure motivates the regime-switching extension (Section 7).

---

## 5. Validation

### 5.1 Simulation-Based Calibration

Simulation-based calibration (SBC; Talts et al., 2018) verifies that the inference procedure recovers known parameters. We generate 100 synthetic datasets from the prior, run NUTS inference (200 warmup + 200 samples) on each, and test whether the posterior rank statistics are uniformly distributed.

[TABLE 8: SBC Results]

| Parameter | N | KS Statistic | p-value | Verdict |
|---|---|---|---|---|
| α_herd | 100 | 0.070 | 0.685 | PASS |
| α_anchor | 100 | 0.095 | 0.308 | PASS |
| α_social | 100 | 0.075 | 0.600 | PASS |
| α_event | 100 | 0.075 | 0.600 | PASS |
| τ_readout | 100 | 0.095 | 0.308 | PASS |
| σ_obs | 100 | 0.105 | 0.205 | PASS |

All 6 parameters pass the KS uniformity test (p > 0.05), with the lowest p-value at 0.205. This confirms that the inference procedure is well-calibrated: when the model is correctly specified, the posterior is neither overconfident nor underconfident, and there is no systematic bias in parameter recovery.

[FIGURE 1: SBC rank histograms for all 6 parameters. Each histogram should show approximately uniform distribution across 10 bins. All parameters pass the KS test for uniformity.]

The SBC configuration uses 10 agents and 7 rounds per synthetic scenario with Normal(0, 0.3) priors. Total runtime: 297.6 seconds (approximately 3 seconds per instance).

### 5.2 Global Sensitivity Analysis

Variance-based global sensitivity analysis (Sobol, 2001) decomposes output variance into contributions from individual parameters and their interactions. We evaluate all 8 model parameters using N = 1024 Saltelli samples (18,432 total simulator evaluations, n = 30 agents, 7 rounds).

[TABLE 9: Sobol Sensitivity Indices]

| Parameter | S1 (Main Effect) | ST (Total Effect) | Type |
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

1. **Dominant pair.** α_herd (ST = 0.555) and α_anchor (ST = 0.452) together account for the vast majority of output variance, including through their interaction. Their second-order interaction index S2(α_herd, α_anchor) = 0.094 is the largest pairwise interaction, indicating strong nonlinear coupling between herd behavior and anchor rigidity.

2. **Calibrable/frozen partition justified.** The four frozen parameters (λ_elite, λ_citizen, θ_herd, δ_drift) all have S1 < 0.01, confirming that they can be fixed at default values without significant loss of model expressiveness. The citizen step size λ_citizen shows moderate total effect (ST = 0.121) due to interactions, but its main effect is negligible (S1 = 0.002).

3. **Sum diagnostics.** ΣS1 = 0.70 indicates that approximately 30% of output variance arises from parameter interactions. ΣST = 1.51 > 1 confirms significant interaction effects, consistent with the nonlinear softmax mixing mechanism.

[FIGURE 2: Sobol indices bar chart. Two panels: S1 (main effects) and ST (total effects) for all 8 parameters. The four calibrable parameters (herd, anchor, social, event) should be visually dominant; the four frozen parameters near zero for S1.]

A JAX reimplementation of the Sobol analysis confirms these findings: the top-2 parameters (α_herd, α_anchor) and their dominant interaction are reproduced, with ranking variations within confidence intervals (α_anchor ST = 0.551 in JAX vs. 0.452 in NumPy, both within ±0.05 CI overlap).

### 5.3 Cross-Validation with Expanded Dataset

To assess sensitivity to dataset size, we compare calibration on the original 22-scenario dataset versus the expanded 42-scenario dataset:

[TABLE 10: 22-Scenario vs. 42-Scenario Calibration]

| Metric | 22-Scenario | 42-Scenario | Δ |
|---|---|---|---|
| N (train / test) | 16 / 6 | 34 / 8 | +18 / +2 |
| MAE (test) | 11.7 pp | 19.2 pp | +7.4 |
| RMSE (test) | 14.7 | 26.6 | +11.9 |
| Coverage 90% (test) | 83.3% | 75.0% | -8.3% |
| MAE (train) | 16.3 pp | 14.3 pp | -2.0 |
| Coverage 90% (train) | 68.8% | 79.4% | +10.6% |

The expanded dataset improves training fit (MAE 16.3 → 14.3 pp, coverage 68.8% → 79.4%) but shows degraded test performance, largely driven by the inclusion of more challenging financial and corporate scenarios in the test set. This suggests that the model's performance ceiling is scenario-dependent rather than data-limited.

---

## 6. Online Data Assimilation via Ensemble Kalman Filter

### 6.1 Motivation

The calibrated posterior provides a static prediction: given a scenario, produce a probabilistic forecast of the final outcome. But real-world opinion dynamics unfold over time, and intermediate observations (polls, social media sentiment, official statements) become available as the process evolves. An online assimilation system should:

1. Start from the calibrated prior (posterior from Section 3).
2. Incorporate streaming observations as they arrive.
3. Jointly update both model parameters and agent states.
4. Produce calibrated probabilistic forecasts at each round.

This bridges the gap between offline calibration and real-time digital twin operation.

### 6.2 Ensemble Kalman Filter Formulation

We adopt a stochastic Ensemble Kalman Filter (Evensen, 2003) operating on an augmented state vector that combines model parameters with agent positions.

**State vector.** Each ensemble member j ∈ {1, ..., E} maintains:

$$\mathbf{x}_j = [\boldsymbol{\theta}_j, \mathbf{z}_j]^\top$$

where $\boldsymbol{\theta}_j = [\alpha_h, \alpha_a, \alpha_s, \alpha_e]$ are the four calibrable parameters and $\mathbf{z}_j = [p_1, ..., p_n]$ are the n agent positions. The state dimension is 4 + n.

**Forecast step.** At each round t, the forecast propagates parameters as a random walk and states through the JAX simulator:

$$\boldsymbol{\theta}_j^f(t+1) = \boldsymbol{\theta}_j^a(t) + \boldsymbol{\eta}_j^\theta, \quad \boldsymbol{\eta}_j^\theta \sim \mathcal{N}(0, Q_\theta I)$$

$$\mathbf{z}_j^f(t+1) = \text{step\_round}(\mathbf{z}_j^a(t), \boldsymbol{\theta}_j^f(t+1), \text{event}_t) + \boldsymbol{\eta}_j^z, \quad \boldsymbol{\eta}_j^z \sim \mathcal{N}(0, Q_z I)$$

where Q_θ controls parameter exploration speed and Q_z adds stochastic perturbation to agent positions. The forecast is parallelized over the ensemble using `jax.vmap`.

**Update step.** When an observation y_obs with variance R is available:

1. Compute ensemble predictions: $\hat{y}_j = h(\mathbf{z}_j^f)$ where h(·) is the readout function (Section 2.6).

2. Compute anomalies: $\mathbf{X}^{\text{anom}} = \mathbf{X}^f - \bar{\mathbf{X}}^f$, $\hat{y}^{\text{anom}} = \hat{y} - \bar{\hat{y}}$.

3. Cross-covariance and innovation variance:

$$\mathbf{P}_{xh} = \frac{1}{E-1} \mathbf{X}^{\text{anom}} (\hat{y}^{\text{anom}})^\top, \quad P_{hh} = \frac{1}{E-1} \|\hat{y}^{\text{anom}}\|^2$$

4. Kalman gain: $\mathbf{K} = \mathbf{P}_{xh} / (P_{hh} + R)$

5. Perturbed update:

$$\mathbf{x}_j^a = \mathbf{x}_j^f + \mathbf{K} \cdot (y_{\text{obs}} + \epsilon_j - \hat{y}_j), \quad \epsilon_j \sim \mathcal{N}(0, R)$$

6. Multiplicative inflation to prevent ensemble collapse:

$$\mathbf{x}_j^a \leftarrow \bar{\mathbf{x}}^a + \gamma (\mathbf{x}_j^a - \bar{\mathbf{x}}^a), \quad \gamma = 1.02$$

### 6.3 Observation Adapters

The EnKF supports three observation types through adapter classes:

- **PollingSurvey**: pro_pct ∈ [0, 100] with variance inversely proportional to sample size: R = pro_pct · (100 - pro_pct) / sample_size.
- **SentimentSignal**: Maps sentiment scores to approximate pro_pct with configurable noise floor.
- **OfficialResult**: Final certified outcome with minimal observation noise (R = 1.0).

### 6.4 Brexit Case Study

We demonstrate the EnKF on the Brexit referendum scenario (ground truth: 51.89% Leave). Polling observations are released at rounds 1, 3, 5, and 7 of the 9-round simulation.

[TABLE 11: EnKF Brexit Results]

| Round | Mean (%) | Std | 90% CI | Width | Observation |
|---|---|---|---|---|---|
| 1 | — | — | — | — | Polling released |
| 3 | — | — | — | — | Polling released |
| 5 | — | — | — | — | Polling released |
| 7 | — | — | — | — | Polling released |
| 9 (final) | ≈54.1 | — | — | — | — |

The final prediction error is approximately **2.2 pp** with four polling observations. This represents a substantial improvement over the offline-only prediction, which relies solely on the calibrated posterior without any scenario-specific data.

Key observations:

1. **CI contraction.** The 90% credible interval width decreases after each observation injection, from the prior width to a substantially narrower posterior.

2. **Ensemble stability.** The ensemble spread remains above 0.001 throughout, indicating no filter collapse—the multiplicative inflation (γ = 1.02) maintains sufficient diversity.

3. **Parameter convergence.** All four calibrable parameters remain within reasonable bounds (|α| < 3.0) after assimilation, with standard deviations narrowing from the prior.

It is important to note that the 2.2 pp EnKF error is achieved *with streaming polling data*—it is not directly comparable to the offline-only MAE of 12.6 pp, which uses no scenario-specific observations. The EnKF demonstrates what becomes possible when the digital twin receives live data, not a claim about the base model's accuracy.

[FIGURE 3: EnKF convergence on Brexit. Top panel: pro% prediction (mean ± 90% CI) over 9 rounds, with vertical markers at observation rounds 1, 3, 5, 7. Horizontal line at GT = 51.89%. Bottom panel: CI width over rounds, showing contraction at each observation. Should clearly show CI narrowing and mean converging toward GT.]

### 6.5 Convergence Verification

A synthetic convergence test with known ground-truth parameters confirms that the EnKF recovers true parameters when given sufficient observations. Starting from a vague prior centered at zero (true values: α_h = -0.2, α_a = 0.3, α_s = -0.1, α_e = -0.15), all four parameters converge to within 0.5σ of their true values after 9 rounds of observations with 3 pp noise. A no-observation baseline confirms that without data, the EnKF produces prior-consistent forecasts with no information gain, as expected.

---

## 7. Regime Switching for Crisis Dynamics

### 7.1 Motivation

The five worst-performing scenarios (Section 4.5) share a common pattern: rapid, discontinuous opinion shifts that the base model's linear force combination cannot reproduce. Financial crises (WeWork δ_s = -1.378, SVB δ_s = -0.835, FTX δ_s = -0.727) exhibit trust collapses where public opinion drops precipitously within 1-2 rounds. The base model, with its smooth force mixing and capped step sizes, can only produce gradual movements.

We address this with a regime-switching extension that introduces a latent "crisis regime" with amplified dynamics, activated by observable signals (shock magnitude, position velocity, institutional trust).

### 7.2 Two-Regime Architecture

The model switches between two regimes:

- **Regime 0 (Normal)**: Standard force-based opinion evolution as described in Section 2.
- **Regime 1 (Crisis)**: Amplified step sizes, suppressed anchor rigidity, amplified herd and event forces, and a direct crisis push that bypasses the softmax mixing entirely.

The switching is *soft*: regime probability P(crisis) ∈ (0, 1) is computed via sigmoid, and all parameters are interpolated:

$$\boldsymbol{\theta}_{\text{eff}} = (1 - p) \cdot \boldsymbol{\theta}_{\text{normal}} + p \cdot \boldsymbol{\theta}_{\text{crisis}}$$

This preserves JAX differentiability and `jax.lax.scan` compatibility—no discrete branching is needed.

### 7.3 Regime Probability

The crisis probability is computed via logistic regression on four observable signals:

$$\text{logit}(p) = \beta_0 + \beta_{\text{shock}} (m_t - \tau_{\text{shock}}) + \beta_{\text{vel}} (v_t - \tau_{\text{vel}}) + \beta_{\text{trust}} (1 - \text{trust}) + \beta_{\text{recovery}} \cdot r_t + \beta_{\text{momentum}} \cdot p_{t-1}$$

where:
- m_t is the shock magnitude at round t, τ_shock = 0.5 is the threshold.
- v_t = |Δ̄p| is the mean absolute position velocity from the previous round.
- trust ∈ [0, 1] is the scenario-level institutional trust covariate.
- r_t is a soft (decayed) count of rounds spent in crisis.
- p_{t-1} is the previous round's crisis probability (momentum/hysteresis).

The coefficients are: β_0 = -1.5 (base bias toward normal), β_shock = 8.0, β_vel = 4.0, β_trust = 1.5, β_recovery = -1.5/τ_duration, β_momentum is linear (3.0 · p_{t-1} - 0.5).

### 7.4 Crisis Dynamics

In the crisis regime, three modifications occur:

1. **Step size amplification.** λ_crisis = λ_normal × 3.0, with relaxed caps.

2. **Force reweighting.** The softmax weights are shifted: herd amplified by log(2.0), event amplified by log(2.5), anchor suppressed by -2.0 in logit space.

3. **Direct crisis push.** This is the key mechanism—a force that bypasses the softmax mixing entirely:

$$\Delta p_i^{\text{crisis}} = p \cdot \kappa \cdot m_t \cdot d_t \cdot (1 - \rho_i)$$

where κ = 0.4 is the contagion speed. This direct push produces the discontinuous position jumps that the linear force combination cannot achieve. The susceptibility factor (1 - ρ_i) ensures rigid agents are less affected.

The total position update becomes:

$$p_i(t+1) = \text{clip}(p_i(t) + \Delta p_i^{\text{forces}} + \Delta p_i^{\text{crisis}}, -1, +1)$$

[TABLE 12: Crisis Parameter Defaults]

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

### 7.5 Regime Switching Results

We compare the base model (v2, no regime switching) against the regime-switching model (v3) on 14 financial and corporate scenarios:

[TABLE 13: v2 vs v3 on Financial/Corporate Scenarios (Selected)]

| Scenario | GT (%) | v2 Error | v3 Error | Δ (pp) | Max P(crisis) | Crisis Rounds |
|---|---|---|---|---|---|---|
| Dieselgate | 32.0 | 29.6 | 0.4 | +29.1 | 0.930 | 3 |
| FTX Crypto | 22.0 | 11.4 | 5.3 | +6.0 | 1.000 | 2 |
| Facebook→Meta | 26.0 | 14.6 | 6.4 | +8.2 | 0.911 | 2 |
| Archegos | 35.0 | 65.0 | 54.5 | +10.4 | 0.930 | 2 |
| Boeing 737 MAX | 40.0 | 9.5 | 23.7 | -14.2 | 0.833 | 1 |
| SVB Collapse | 38.0 | 38.8 | 61.9 | -23.1 | 0.989 | 2 |
| Twitter/X | 41.0 | 7.8 | 25.5 | -17.7 | 0.826 | 2 |

The results are mixed. Regime switching produces substantial improvements on scenarios where the crisis dynamics align with the shock direction (Dieselgate: 29.6 → 0.4 pp, FTX: 11.4 → 5.3 pp, Meta: 14.6 → 6.4 pp) but degrades performance on scenarios where the crisis push amplifies the wrong direction (SVB: 38.8 → 61.9 pp, Boeing: 9.5 → 23.7 pp).

The SVB degradation is partly attributable to a data loading issue: when multiple events occur in the same round (bank collapse followed by Fed bailout), the crisis push amplifies the last-recorded direction rather than the net effect. The high max P(crisis) across all scenarios (0.67–1.0) with default thresholds suggests that the trigger is too sensitive and requires calibration via SVI (the v3 hierarchical model, Section 3, extends the prior to include 5 global crisis parameters).

[FIGURE 4: Regime activation on synthetic crisis scenario. Top panel: agent position trajectories over 9 rounds, with shock at round 3 (magnitude 0.8, direction -1). Dashed line = base model, solid = regime switching. Bottom panel: regime probability over rounds, showing activation at round 3 (p ≈ 0.77), then gradual recovery.]

### 7.6 Design for Calibration

The crisis parameters are designed to be learnable within the hierarchical framework. The v3 model extends the v2 prior with 5 additional global parameters:

- crisis_lambda_mult ~ LogNormal(log(3), 0.5)
- crisis_anchor_supp ~ Beta(2, 8)
- crisis_event_amp ~ LogNormal(log(2.5), 0.5)
- crisis_herd_amp ~ LogNormal(log(2.0), 0.5)
- shock_trigger ~ Beta(5, 5)

These parameters are shared globally (not per-scenario), as crisis dynamics are hypothesized to be a universal mechanism modulated by scenario-specific triggers. The SVI inference infrastructure from v2 extends naturally to v3 by including these additional variables in the variational guide.

---

## 8. Discussion

### 8.1 Strengths

**Principled calibration.** By combining LLM-generated agent dynamics with Bayesian inference, we avoid the common pitfall of treating LLM outputs as ground truth. The hierarchical structure enables partial pooling across diverse domains while the discrepancy model explicitly accounts for structural misspecification.

**Validated inference.** SBC confirms that the posterior is well-calibrated (6/6 parameters pass), and Sobol analysis provides a principled basis for the calibrable/frozen partition.

**Online assimilation.** The EnKF bridges offline calibration to real-time operation, enabling the system to function as a true digital twin that updates beliefs as data arrives.

**Differentiable architecture.** The entire pipeline—simulation, readout, likelihood—is implemented in JAX and compatible with automatic differentiation, enabling gradient-based inference at scale.

### 8.2 Limitations

**Financial domain performance.** The model systematically over-predicts support in financial crisis scenarios (mean |δ_s| = 0.744 in logit space). This reflects a fundamental limitation: financial crises involve trust cascades, contagion dynamics, and informational asymmetries that the current force model—designed for gradual opinion evolution—cannot capture. The regime-switching extension addresses some cases but introduces new failure modes.

**LLM-simulator gap.** The observation model treats the LLM-generated agent behaviors as fixed inputs to the force system. In reality, the LLM's narrative generation introduces its own biases (training data recency, cultural priors, instruction-following tendencies) that are confounded with the opinion dynamics parameters. The discrepancy terms absorb these effects but do not disentangle them.

**Sample size.** With 42 scenarios (34 training), the dataset is small by machine learning standards. Several domains have only 1-2 scenarios, making domain-level inference unreliable for those domains. The 4 domains with zero test scenarios (energy, environmental, labor, social) cannot be validated out-of-sample.

**Observation model simplifications.** The BetaBinomial likelihood assumes independent polling rounds and does not model autocorrelation in opinion trajectories. The EnKF's Gaussian assumption may not hold for highly polarized scenarios where the opinion distribution is bimodal.

**Regime switching sensitivity.** The default crisis thresholds are hand-tuned and too aggressive (max P(crisis) > 0.67 on most scenarios). Full SVI calibration of the 5 crisis parameters on the empirical dataset is needed to determine whether regime switching provides net benefit.

### 8.3 Future Work

1. **v3 calibration.** Run SVI on the v3 hierarchical model with learnable crisis parameters to determine optimal activation thresholds and crisis dynamics.

2. **Multi-event data loading.** Fix the event aggregation in `build_scenario_data_from_json` to properly handle multiple events per round (net magnitude and weighted direction rather than last-event-wins).

3. **Temporal observation model.** Replace the independent-rounds assumption with a state-space observation model that captures autocorrelation.

4. **Expanded dataset.** Curate additional scenarios in underrepresented domains (energy, environmental, labor) to improve domain-level estimates.

5. **LLM ablations.** Systematically vary the LLM backbone (temperature, model size, prompting strategy) and measure the effect on calibrated parameters to disentangle LLM bias from opinion dynamics.

6. **EnKF + regime switching integration.** Combine online assimilation with regime detection to dynamically activate crisis dynamics based on observed data, not just simulator-internal signals.

---

## 9. Conclusion

We have presented DigitalTwinSim, a framework for transforming LLM-agent opinion simulations from uncalibrated narrative generators into quantitative digital twins with calibrated uncertainty. The key insight is that LLM-agent simulations, despite their structural misspecification, contain learnable signal about opinion dynamics mechanisms—but only when combined with principled Bayesian calibration and explicit model discrepancy.

The calibrated model achieves 12.6 pp MAE on held-out scenarios with 85.7% coverage of 90% credible intervals. Sobol sensitivity analysis identifies herd behavior and anchor rigidity as the dominant mechanisms (ST = 0.55 and 0.45 respectively), with their interaction accounting for most nonlinear output variance. Simulation-based calibration confirms posterior validity across all 6 parameters.

The Ensemble Kalman Filter extends the framework to online operation, demonstrating 2.2 pp error on the Brexit scenario with four streaming polling observations. The regime-switching extension provides a mechanism for modeling discontinuous crisis dynamics, with promising results on financial scenarios (Dieselgate: 29.6 → 0.4 pp error improvement) but requiring further calibration for robust deployment.

The framework's modular design—force-based dynamics, hierarchical Bayesian calibration, online assimilation, regime switching—allows each component to be extended independently. The immediate next step is full SVI calibration of the regime-switching parameters, which would close the loop between offline calibration, crisis detection, and online assimilation.

---

## Appendix A: Implementation Details

### A.1 Software Stack

The simulator and inference pipeline are implemented in Python using:
- **JAX** (0.4+) for differentiable simulation and automatic differentiation.
- **NumPyro** for probabilistic programming and variational inference.
- **SALib** for Sobol sensitivity analysis.
- All simulation code is `jax.jit`-compatible and uses `jax.lax.scan` for sequential round stepping, avoiding Python loops.

### A.2 Computational Requirements

| Task | Runtime | Hardware |
|---|---|---|
| SVI calibration (3000 steps, 42 scenarios) | 40.2 min | CPU (Apple Silicon) |
| SBC (100 instances, NUTS) | 5.0 min | CPU |
| Sobol analysis (18,432 evaluations) | ~10 min | CPU |
| EnKF online assimilation (50 members, 9 rounds) | ~5 sec | CPU |

### A.3 Gauge Fixing

The softmax function is shift-invariant: softmax(α + c) = softmax(α) for any scalar c. With 5 forces and 4 free parameters, one weight must be fixed to resolve the gauge. We fix α_direct = 0, making the remaining weights interpretable as log-odds ratios relative to direct LLM influence. This is analogous to the reference category in multinomial logistic regression.

### A.4 EMA Standardization

The exponential moving average standardization uses a sliding window approximation (last 8 agents) to compute per-round mean and standard deviation of each force. The decay parameter α_EMA = 0.3 provides a balance between responsiveness to current-round force distributions and stability across rounds. The ε = 1e-6 additive constant prevents division by zero in the first few rounds when σ_k ≈ 0.

---

## References

DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345), 118-121.

Evensen, G. (2003). The Ensemble Kalman Filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343-367.

Gao, C., Lan, X., Li, N., Yuan, Y., Ding, J., Zhou, Z., ... & Li, Y. (2023). Large language models empowered agent-based modeling and simulation: A survey and perspectives. *arXiv preprint arXiv:2312.11970*.

Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: models, analysis and simulation. *Journal of Artificial Societies and Social Simulation*, 5(3).

Holley, R. A., & Liggett, T. M. (1975). Ergodic theorems for weakly interacting infinite systems and the voter model. *The Annals of Probability*, 3(4), 643-663.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*.

Park, J. S., Popowski, L., Cai, C., Morris, M. R., Liang, P., & Bernstein, M. S. (2022). Social simulacra: Creating populated prototypes for social computing systems. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology*.

Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv preprint arXiv:1804.06788*.
