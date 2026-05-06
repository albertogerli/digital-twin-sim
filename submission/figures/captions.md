# Figure captions

## Figure 1 — Cost vs shock-magnitude (log-log)
**File:** `fig01_cost_vs_shock_loglog.{png,pdf}`

Reference dataset N=40 historical operational-risk incidents (1998–2024) plotted on log-log axes by category.
Dashed line: overall power-law fit $\hat{c}(s) = \beta \cdot s^\gamma$ obtained by log-log Huber regression on the
full corpus. Fitted parameters: $\beta \approx \text{€1.16B}$, $\hat{\gamma} = 3.36$, $R^2_{\log} = 0.72$.
Lehman 2008 (banking-US, top-right) is the dominant high-leverage observation.

## Figure 2 — Leave-one-out hit-rate by model specification
**File:** `fig02_loo_hit_rates_by_mode.{png,pdf}`

Hit-rate within ±50% / ±100% / ±200% under three competing model specifications evaluated by leave-one-out
cross-validation on the N=40 corpus. M1 = linear pooled (single $\alpha$ across all categories);
M2 = linear per-category ($\alpha_k$ refit on each held-out's own bucket); M3 = per-category power-law
$\beta_k \cdot s^{\gamma_k}$ (the proposed estimator). M3 attains 80% within ±100%; M1 attains 35%, M2 attains 40%.

## Figure 3 — Per-category convexity exponent $\hat{\gamma}_k$
**File:** `fig03_per_category_gamma.{png,pdf}`

Power-law exponent $\hat{\gamma}_k$ per category with empirical 5°/95° pairs-bootstrap confidence interval
(N=2000 replicates per category). Vertical dashed line at $\gamma = 1$ (linear baseline). Every category
satisfies $\hat{\gamma}_k > 1$, with point estimates ranging from 1.65 (energy, n=3) to 3.92 (banking-US, n=7).
The hypothesis $H_0: \gamma = 1$ is rejected at $p < 0.01$ on every category with $n \geq 4$.

## Figure 4 — Hidden Markov regime posterior on monthly log(VIX)
**File:** `fig04_hmm_regime_posterior.{png,pdf}`

VIX month-end series 1997-01 to 2025-12 ($T = 348$ observations) with the 2-state Gaussian HMM regime
posterior $P(z_t = \text{high} \mid x_{1:T})$ overlaid (right axis, red). Pink shading is proportional to the
posterior. Five reference incidents (LTCM 1998, Argentine default 2001, Lehman 2008, Brexit 2016, SVB 2023)
are annotated. The HMM cleanly separates a low-volatility regime ($\hat{\mu}_0 = \log 14.7$) from a
high-volatility regime ($\hat{\mu}_1 = \log 24.9$); both are ~96% persistent.

## Figure 5 — Residual diagnostic: linear vs power-law
**File:** `fig05_residuals_vs_shock.{png,pdf}`

Per-incident percent prediction error under the overall linear fit (left, $\hat{c} = \alpha s$ with $\hat{\alpha} \approx \text{€14B}$/unit) and the overall power-law fit (right, $\hat{c} = 1{,}158 \cdot s^{3.36}$).
Light-green band marks $\pm 100\%$ (the tier-correct prediction range for DORA reporting). The linear model
exhibits systematic over-prediction for $s < 1.5$ (Tercas, ENI Gabon, TIM downgrade) reaching $> +400\%$;
the power-law splits its errors symmetrically around zero.

## Figure 6 — Reference dataset breakdown by category
**File:** `fig06_calibration_dataset_breakdown.{png,pdf}`

Number of incidents per category (solid bars, left axis) and median cost per category in €M (hatched bars,
right axis, log scale). The corpus is dominated numerically by EU and US banking incidents (8 + 7 = 15)
but spans seven categories with $n \geq 3$ each. Median cost spans three orders of magnitude across
categories, from €1,600M (telco) to €27,500M (sovereign).
