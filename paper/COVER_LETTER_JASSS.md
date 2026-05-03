# Cover Letter — JASSS Submission

**To:** Editorial Office, *Journal of Artificial Societies and Social Simulation*

**Re:** *"A Calibrated LLM-Conditioned Agent-Based Model of Public Opinion Dynamics with Null-Baseline Benchmarking, Data-Contamination Auditing, and Online Assimilation"* — original submission, **revision v2.8** (May 2026)

Dear Editor,

I am submitting the manuscript above for consideration as an original research article in JASSS.

**What the paper does.** It presents a single coherent framework that combines (i) a force-based opinion-dynamics ABM whose per-agent reactive shifts are emitted by a large language model, (ii) hierarchical Bayesian calibration with explicit Kennedy–O'Hagan model discrepancy on 42 empirical scenarios across ten domains, (iii) an Ensemble Kalman Filter that augments parameters and states for online assimilation against streaming polling data, (iv) a four-axis LLM data-contamination probe that quantifies, per scenario, what the model knows about title, country, dates, agents, trajectory, and outcome before any simulation runs, (v) a deterministic title/country/date/agent blinding protocol that drops that probe from index 0.721 to 0.000 on the high-leak political-referendum subset, and (vi) an apples-to-apples sim-lift evaluation that runs the full simulator under both contaminated and blinded variants on the eleven highest-leak scenarios and compares each trajectory against persistence / AR(1) / linear-trend / running-mean baselines via Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction.

**The headline empirical reading is a pre-registered null result.** Under blinding, the simulator does *not* beat persistence in trajectory space at $p < 0.05$ on any of the eleven scenarios; the blinded mean DTW (0.100) is within 0.004 of the contaminated mean (0.096), which rules out LLM memorisation as the source of any retrospective skill but also bounds how much skill there was to be claimed in the first place. We report this transparently as the load-bearing measurement of the paper rather than as a secondary check, and we adjust the framing throughout: the manuscript avoids the term "digital twin" in the predictive sense and locates the framework's operational value in the EnKF online regime and in the benchmark-integrity instrumentation rather than in an open-loop retrospective forecasting claim.

**Why this fits JASSS.** The contribution sits at three intersections that JASSS has historically been the natural home for: (a) ABM methodology — we provide a full ODD protocol description (Appendix G; Grimm et al. 2010, 2020), Sobol sensitivity analysis, simulation-based calibration, and a permutation-invariance audit; (b) calibration discipline for ABMs — we follow the Grazzini–Richiardi–Tsionas framing and the Platt comparative literature, with an explicit hierarchical decomposition that partitions cross-scenario heterogeneity from scenario-specific discrepancy; (c) the methodological challenge of using LLMs *inside* social simulators without inheriting their data-contamination biases — to our knowledge this is the first paper to operationalise that audit as a falsifiable, reproducible protocol with a blinded A/B that shows the contamination signal collapsing under the transformation.

**Novelty over prior LLM-agent work.** Park et al. (2023, 2022) and Argyle et al. (2023) demonstrate that LLMs can populate generative agents and reproduce survey distributions. Horton (2023), Gao et al. (2023), and the wider survey literature catalogue applications. None of these works (a) calibrate the LLM-conditioned dynamics under a hierarchical Bayesian likelihood, (b) audit the LLM's data contamination on the empirical evaluation corpus, or (c) report a null-baseline benchmark that the simulator does not pass. We see (c) as a feature rather than a limitation: the field needs honest measurements of what LLM-agent simulations cannot do, alongside honest measurements of what they can.

**Reproducibility.** A pinned-version reproducibility README (`paper/REPRODUCIBILITY.md`), a CITATION.cff at the repository root, and the canonical git commit (`f3bf60e5ed58f582598fad50abd5f0c51bd86238` for v2.7; the v2.8 supersession lives at HEAD of `main`) accompany the submission. Every numerical artifact in the paper is regenerable from one of seven documented commands. A Zenodo archive will be minted at acceptance.

**Note on v2.8 (this revision).** Between v2.7 and v2.8 the simulator code path was refactored across thirteen targeted sprints (country-alias normalisation, realism-gate fix, agent-prompt and engine improvements; full catalogue in `docs/SPRINT_1-13_CHANGELOG.md`). The hierarchical SVI calibration was re-fit on the **same 42 empirical scenarios** with **identical hyperparameters** (3000 SVI steps, lr 0.005, seed 42). Test-set MAE moved from **19.18 to 17.56 pp** (−1.62), 90% credible-interval coverage moved from **75.0% to 87.5%** (+12.5), final SVI loss from 514.7 to 493.8. All inference machinery (NUTS validation, Sobol sensitivity, EnKF, contamination probe, blinding protocol, null-baseline benchmark) is unchanged from v2.7; the headline scientific claim — that the calibrated ABM matches but does not dominate naive persistence in retrospective trajectory space — is unaffected. See §10 of the manuscript for the full diff and per-domain / per-scenario breakdown.

**Format.** The manuscript is structured for the JASSS main + online supplement convention. Sections 1–9 plus an abridged Table 2 constitute the main submission; Appendices A–G constitute the online supplementary material, with explicit `<!-- SI-BEGIN/END -->` markers in the source markdown so a single source produces both PDFs. Build instructions are in the reproducibility document.

**Conflicts of interest, prior submission.** The paper has not been submitted elsewhere. The author has no financial conflicts of interest. The Gemini API used for the LLM substrate is a paid service; total cost across all experiments reported in the paper is documented in §6.10.4 and Appendix D.

I would be glad to suggest reviewers from the ABM-calibration, opinion-dynamics, and data-assimilation communities upon request.

Sincerely,

Alberto Gerli
Tourbillon Tech Srl, Padova
Università degli Studi di Milano
alberto@albertogerli.it
