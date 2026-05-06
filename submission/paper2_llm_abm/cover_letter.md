# Cover Letter — Submission to JASSS / Journal of Computational Social Science

**Date:** 6 May 2026

**To:** Editor-in-Chief, *Journal of Artificial Societies and Social Simulation* (JASSS)
*(or alternative venue: Journal of Computational Social Science, Computational Economics)*

Dear Editor,

I am writing to submit the manuscript "**Calibrated LLM-Agent Models of Public Opinion: A Bayesian Framework with Contamination Auditing and a Reproducible Skill Floor**" for consideration.

The manuscript addresses two methodological problems that, in my view, are limiting the scientific value of large-language-model-driven agent-based simulations of opinion dynamics. Reported predictive performance is uncalibrated (no principled mechanism anchors simulator outputs to observational data) and contaminated (LLMs have been pre-trained on narratives describing the same historical events that retrospective benchmarks use). The paper closes both gaps with the joint application of Bayesian inverse-problem machinery (calibration, sensitivity analysis, simulation-based calibration) and contamination-audit machinery (probing, blinding, apples-to-apples sim-lift) to the LLM-agent ABM setting.

The submission makes three methodological contributions:

1. A **JAX-differentiable five-force opinion-dynamics simulator** combining direct LLM influence, social conformity, herd behaviour, anchor rigidity, and exogenous shocks through a gauge-fixed softmax mixture. The simulator is compatible with `jax.lax.scan` and supports automatic differentiation through the full trajectory.

2. A **three-level hierarchical Bayesian calibration** (global / domain / scenario) with explicit readout-discrepancy terms, fit by stochastic variational inference (SVI) on N=42 historical scenarios across ten domains. Performance metrics: 17.6 percentage-point mean absolute error on held-out scenarios, 87.5% nominal coverage of 90% credible intervals, simulation-based calibration confirming well-specification under NUTS reference, Sobol global sensitivity analysis identifying herd behaviour and anchor rigidity as the dominant mechanisms ($S_T = 0.55, 0.45$).

3. A **four-axis data-contamination audit and deterministic blinding protocol** that, to my knowledge, are the first such instruments applied jointly to LLM-agent ABMs. The probe (outcome / trajectory / events / actors) measures per-scenario LLM prior knowledge; the blinding protocol replaces titles, country names, dates, and agent names with templates while preserving every numeric field the simulator consumes. On 11 high-leakage political-referendum scenarios the probe finds mean contamination 0.721; the blinding protocol drops it to 0.000 across all four axes. An apples-to-apples sim-lift evaluation under both arms finds that the calibrated framework matches but does not beat naive persistence on this subset, ruling out memorization-driven retrospective skill claims.

A **null-baseline benchmarking layer** (Diebold–Mariano with Harvey–Leybourne–Newbold small-sample correction against four standard forecasters) is documented in Section 7 and released as a reproducible package. The headline finding is consistent with the macroeconomic forecasting experience: naive persistence is a strong baseline that is hard to beat at the trajectory level.

The manuscript's framing — that LLM-agent ABMs are scenario-exploration and counterfactual tooling rather than out-of-the-box forecasters — is novel for the LLM-agent simulation literature, which has largely sidestepped both the contamination and the null-baseline questions. I believe this framing, combined with the contamination-audit machinery, can set a useful methodological standard for predictive-skill claims in this rapidly growing literature.

The manuscript is 28 pages including references. Code and data are released at `https://github.com/albertogerli/digital-twin-sim`.

I confirm that:

- The work is original and has not been previously published.
- The work is not under consideration at any other venue.
- All authors (single author) have agreed to this submission.
- There are no conflicts of interest to declare.
- Funding: Tourbillon Tech Srl computational resources; no external grant.

I would like to suggest the following candidate reviewers:

- **Sven Banisch** (Universität Bielefeld) — opinion dynamics, learning from social feedback
- **Joon Sung Park** (Stanford) — generative agents, LLM-driven social simulations
- **Inbal Magar** (Technion) — LLM data contamination audits
- **Peter Druckman** (Northwestern) — political opinion dynamics, communication research

I respectfully decline reviewers from Università degli Studi di Milano (joint affiliation) or Tourbillon Tech Srl (sole funding source).

I look forward to the editor's and reviewers' feedback.

Sincerely,

**Alberto Giovanni Gerli**
Tourbillon Tech Srl, Padova, Italy
Dipartimento di Scienze Cliniche e di Comunità, Università degli Studi di Milano, Italy
alberto@albertogerli.it
