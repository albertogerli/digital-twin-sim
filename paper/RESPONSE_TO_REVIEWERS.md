# Response to Reviewers

**Manuscript:** *A Calibrated LLM-Conditioned Agent-Based Model of Public Opinion Dynamics with Null-Baseline Benchmarking, Data-Contamination Auditing, and Online Assimilation*

**Note on framing.** This document is a *self-prepared anticipation* of reviewer concerns from an internal pre-submission audit, written in the format of a reply letter so that the reviewers' own report can be addressed against a structured baseline. Each item below begins with the concern (paraphrased), our response, and the precise location in the revised manuscript where the response is implemented. Line numbers refer to the revised markdown source `paper/digitaltwin_calibration_v2.7.md` at commit `f3bf60e`. **The v2.8 supersession** (May 2026; see §10 of the manuscript and `docs/SPRINT_1-13_CHANGELOG.md`) re-fits the SVI calibration after thirteen sprints of simulator hardening — test MAE 19.18 → 17.56 pp, cov₉₀ 75.0% → 87.5% — but does not change any of the responses below; the inference machinery is unchanged.

---

## Concern 1 — Inconsistent scenario count (42 / 43 / 44 across the manuscript)

**Concern.** The previous draft reported corpus size as 42, 43, and 44 in different places, with no single canonical statement of how the splits are constructed.

**Response.** The corpus is canonically described as **43 in-scope scenarios** = **42 SVI corpus** + **1 eval-only v2.7 addition** (`CORP-2020-BOEING_737_MAX_RETURN_TO_SERVICE`). The SVI calibration (§4–§5) and the multi-modal extension (§5.8) are fit on the 42 scenarios; the v2.5 predictive-skill benchmarks (§6.6–§6.8), the contamination probe (§6.10), and the blinding A/B (§6.10.3) use the full 43. We have unified the count to this scheme everywhere in the paper.

**Location.** §5.1 (Table 2 caption and footnote, line 295); §6.9.1 ("43-Scenario", line 872); §6.10.x throughout; Appendix C.1 caption (line 1432) plus row 7a inserted for the Boeing RTS scenario; Appendix C.1 final summary line (1487).

---

## Concern 2 — 1176-word abstract organised by version-history paragraphs (v2.5 / v2.6 / v2.7)

**Concern.** The abstract was unfit for purpose: a chronological dump of three version-history paragraphs at 1176 words.

**Response.** Rewritten from scratch as a single 407-word abstract (within JASSS guidelines). The new abstract surfaces the three load-bearing measurements: the contamination collapse (0.721 → 0.000 under blinding), the apples-to-apples sim-lift result (mean DTW 0.0997 blinded vs 0.096 contaminated, $\delta = -0.0036$, with the simulator winning on $0/11$ scenarios at $p < 0.05$), and the EnKF online demonstration (1.8 pp final error, declassed as in-sample). All v-cronology paragraphs have been removed.

**Location.** §Abstract (line 12).

---

## Concern 3 — No apples-to-apples benchmark (sim vs. simulator under contamination control)

**Concern.** Previous drafts reported the simulator's performance and reported the contamination probe's performance, but never showed the simulator running in both arms with matched ground-truth comparison.

**Response.** §6.10.4 ("Apples-to-Apples Sim-Lift: Blinded vs. Contaminated Trajectory Skill") was added in this version. It executes 22 full simulations (11 high-leak scenarios × {contaminated, blinded}) via `benchmarks.sim_lift_runner --all`, grades each trajectory with DTW (path-length-normalised), KS, RMSE, and terminal error, and runs Diebold–Mariano (with HLN small-sample correction) against the best of {persistence, AR(1), linear trend, running mean} per scenario. Results are reported as Tables 22a (headline) and 22b (per-scenario verdict). The headline is a *transparent null*: blinded mean DTW 0.0997 vs contaminated 0.096 (delta $-0.0036$, sim survives blinding); on DM the simulator wins $0/11$ in either arm — the calibrated ABM matches but does not dominate persistence on this subset.

**Location.** §6.10.4 (lines 1038–1095 of the revised source); abstract (line 12); §9 Conclusion (line 1275); item 13 of §8.3 future-work list (now closed; line 1259).

---

## Concern 4 — Archegos exclusion appears post-hoc

**Concern.** The previous draft excluded the Archegos scenario (test-set error 65 pp) from the "verified-only" headline metric without making clear that the exclusion was determined before, not after, looking at the model's error.

**Response.** §5.1 now contains a *pre-registered data-quality protocol*: a five-criterion checklist applied uniformly to every corpus scenario at v2.1 corpus-construction time, recorded in `calibration/empirical/scenarios_v2.1/_quality_audit.json` before any SVI run. The criteria are: (1) primary-source ground truth; (2) polling reliability with named pollsters / sample sizes; (3) pro/against label coherence; (4) round-event causal chain; (5) aggregate quality_score $\geq 70$. Archegos fails criteria (1) and (2) (only a Reuters retrospective for ground truth, LLM-estimated polling without verified sample sizes), with quality_score = 67 — the only scenario in the 43-scenario corpus below the threshold. The exclusion would have been made identically had the v2.5 calibration assigned an Archegos error of 0.5 pp. Both verified-only and full-set metrics are reported throughout for transparency.

**Location.** §5.1, lines 315–321 (the "Pre-registered data-quality protocol" paragraph and numbered checklist).

---

## Concern 5 — LLM determinism: single-seed runs without variance bands

**Concern.** The simulator uses LLM completions at temperatures 0.6–0.85; the manuscript reports single-seed numbers without quantifying how much they would change under a different LLM seed.

**Response.** A "Stochasticity bound on the 22-run batch" paragraph has been added to §6.10.4. It separates the two stochastic sources (asyncio task-completion order and LLM sampling) and provides analytic bounds: (i) asyncio order is provably eliminated by the per-round EMA standardisation, with permutation-invariance confirmed empirically at $< 6 \times 10^{-8}$; (ii) LLM sampling stochasticity is bounded above by the per-step clamp $|\Delta p_i| \leq c_{\tau_i}$, yielding a worst-case cumulative DTW envelope of $\leq 0.035$ on a 9-round trajectory — *larger* than the contaminated-blinded delta of $-0.0036$, which strengthens the "no detectable memorisation" reading because the delta sits inside the noise envelope. We also note that the contamination probe runs at temperature 0.0, so the contamination-side claims of §6.10 are exposed to zero LLM stochasticity.

We acknowledge that a direct empirical multi-seed estimate would require an additional $\$25$–$35$ / 12 h compute budget and is in scope for a future revision. We do not claim the present paper resolves this; we claim the bound above is sufficient to support the conclusions of §6.10.4 as stated.

**Location.** §6.10.4 stochasticity-bound paragraph (lines 1083–1089).

---

## Concern 6 — Brexit EnKF demonstration is in-sample (Brexit is in the SVI training set)

**Concern.** The 1.8 pp error reported in §7.4 is on Brexit, but Brexit is part of the SVI training corpus. The number is therefore not an out-of-sample forecasting result.

**Response.** §7.4 has been retitled "Brexit Demonstration (In-Sample)" and now opens with a "Scope of this section" blockquote that explicitly states the in-sample status, confirms Brexit is row 28 of Appendix C.1 (Train split), states that the prior round-0 prediction is a sample from the training-conditioned posterior rather than an open-loop forecast, distinguishes the demonstration's purpose (assimilation mechanics, CI behaviour, comparison vs. naive baselines on the same data stream) from a predictive-skill claim, and cross-references §6.10.4 as the bound on plausible open-loop EnKF gain on unseen political referendums.

**Location.** §7.4 opener (line 1156, with the prominent blockquote and explicit caveat).

---

## Concern 7 — Missing ODD protocol description (JASSS-mandatory for ABMs)

**Concern.** JASSS requires an Overview / Design concepts / Details (ODD) description for every ABM submission (Grimm et al. 2010, 2020). The previous draft did not have one.

**Response.** Appendix G, "ODD Protocol Description", has been added. It covers: G.1 Overview (Purpose and Patterns; Entities, State Variables, and Scales; Process Overview and Scheduling, including a phase-by-phase pseudocode of one round); G.2 Design Concepts (Basic principles, Emergence, Adaptation, Objectives, Learning, Prediction, Sensing, Interaction, Stochasticity, Collectives, Observation); G.3 Details (Initialisation, Input Data, Submodels — including a parameter / equation / calibration table — and a final paragraph on initialisation/input/submodel determinism).

**Location.** Appendix G (lines 1731–1900). New references to Grimm et al. 2006, 2010, 2020 added to the bibliography.

---

## Concern 8 — Manuscript carries too many side-modules for a single article

**Concern.** Sections covered v2.5 benchmarks, v2.6 Layer 0 / realism gate, v2.7 contamination + blinding + sim-lift, multi-modal financial calibration, and the live-market layer. The manuscript was unwieldy for a single submission.

**Response.** A formal main / supplement split has been inserted. Sections 1–9 plus the abridged Table 2 (training/test domain composition) constitute the main JASSS submission. Appendices A (Implementation), B (Regime Switching, preliminary), C (Full corpus list), D (v2.5 benchmark reproducibility), E (Layer 0), F (Contamination/blinding/metrics), G (ODD) constitute the Online Supplementary Material. The split is implemented with `<!-- SI-BEGIN -->` and `<!-- SI-END -->` markers in the source, and `paper/REPRODUCIBILITY.md` §3.7 documents the awk-based extraction commands that produce the main and supplement PDFs from the single source.

**Location.** Cut-line comment block + new "Online Supplementary Material" header (around line 1294 of the revised source).

---

## Concern 9 — Reproducibility package missing (pinned versions, exact commands)

**Concern.** The previous draft listed software dependencies but did not pin versions, did not specify a canonical commit, and did not give a clean command list to regenerate paper numbers.

**Response.** `paper/REPRODUCIBILITY.md` has been added (182 lines). It pins Python and the full JAX / NumPyro / SALib / dtaidistance / pandas / scipy stack at the versions used; states the canonical commit `f3bf60e5ed58f582598fad50abd5f0c51bd86238`; gives the deterministic `pip install` line; lists the LLM model identifier and temperatures (with the contamination/blinded arms forced to 0.0); and provides command blocks tagged §3.1–§3.7 — one per paper section — showing exactly how to regenerate calibration tables, sensitivity analyses, benchmarks, the contamination probe, the blinding A/B, the apples-to-apples sim-lift, the EnKF demonstration, and the paper PDFs (main / supplement / combined). A `CITATION.cff` at the repository root captures author / version / commit / archive identifiers; a Zenodo DOI will be inserted at acceptance.

**Location.** `paper/REPRODUCIBILITY.md`; `CITATION.cff` (repository root).

---

## Concern 10 — Cross-domain heterogeneity not characterised

**Concern.** Domain-level posteriors and discrepancy biases are reported, but the paper does not surface a clear cross-domain comparison of where the model works well versus poorly.

**Response.** §5.4 ("Discrepancy Decomposition") and §5.5 ("Worst-Performing Scenarios") report per-domain $\sigma_d$, per-domain mean $b_d$ and within-domain spread $\sigma_{b,\text{within}}$. The new abstract surfaces this directly: "The hierarchical model partitions cross-scenario heterogeneity (domain-level $\sigma_d$, scenario-level $b_s$) and reports residual structure transparently." The financial-domain over-prediction is highlighted as an open finding (mean $|b_s| = 0.74$ logit). The contamination probe of §6.10.1 shows that high-leak scenarios are exclusively political referendums and national-level elections, which gives a separate per-domain reading: corporate / financial domains are intrinsically less contaminable in the LLM prior, motivating the future-work item 15 (sim-lift on the low-contamination domain subset). We note that a fuller per-domain breakdown of sim-lift skill would itself require a 10–15 scenario sim run in each of the three lowest-contamination domains; that batch is queued under future-work item 15 with an explicit \$1–3 / run / 4–6 h compute budget.

**Location.** §5.4–§5.5; §6.10.1 contamination breakdown by domain; abstract; §8.3 future-work item 15.

---

## Concern 11 — Blinded full-sim incomplete in v2.7-pre

**Concern.** The pre-revision draft promised a 22-run blinded full-sim batch but did not include it. Without that batch the contamination probe of §6.10.3 was orphaned: it showed the LLM's prior knowledge collapsed, but the simulator's behaviour under the same blinding was untested.

**Response.** Resolved. The 22-run batch was executed (via `benchmarks.sim_lift_runner --all`, ~$8–11 in Gemini flash-lite tokens, 4–8 h wall-clock). All 22 trajectory JSONs are committed under `outputs/sim_lift/trajectories/` and the grader output is `outputs/sim_lift.md`. The headline (Tables 22a/22b of §6.10.4) is the load-bearing pre-registered null result described in Concern 3 above.

**Location.** §6.10.4; `outputs/sim_lift/trajectories/*.json`; `outputs/sim_lift.md`.

---

## Other changes made in this revision (not in the 11-item list)

- **Title softened.** Replaced "Bayesian Calibration, Null-Baseline Benchmarking, and Online Data Assimilation for LLM-Agent Opinion Dynamics Simulation" with "A Calibrated LLM-Conditioned Agent-Based Model of Public Opinion Dynamics with Null-Baseline Benchmarking, Data-Contamination Auditing, and Online Assimilation". Removed the keyword "digital twin"; added an explicit disclaimer in §1 contributions ("We avoid the phrase 'digital twin' in the predictive sense …") and four further softening edits at lines 84, 921, 1098, 1183.
- **§9 Conclusion rewritten** to lead with the calibrated-ABM framing and to state explicitly that §6.10.4 results bound the open-loop EnKF claim of §7.
- **§8.3 future-work list updated.** Item 13 was the "remaining v2.7 deliverable" pre-revision; it is now flagged as completed in §6.10.4 with the actual outcome reported. Items 14 and 15 are restated to reflect the post-revision state.

---

## Summary table of revision deltas

| Concern | Status in v2.7-pre | Status in v2.7 | Word delta |
|---|---|---|---|
| 1. Scenario count | mixed 42/43/44 | unified 43 (= 42 + 1) with footnote | +120 |
| 2. Abstract | 1176 w, 3 v-paragraphs | 407 w, single block | $-769$ |
| 3. Apples-to-apples | promised, not run | §6.10.4 with 22 runs | $+1700$ |
| 4. Archegos exclusion | one-liner | pre-registered 5-criterion protocol | $+450$ |
| 5. LLM determinism | not addressed | analytic bound paragraph | $+330$ |
| 6. EnKF in-sample | implicit | explicit blockquote caveat in §7.4 | $+220$ |
| 7. ODD protocol | absent | full Appendix G | $+2400$ |
| 8. Side-modules | unwieldy single doc | main + SI split | $+150$ |
| 9. Reproducibility | minimal | full README + CITATION.cff | new file 182 lines |
| 10. Cross-domain | uneven | abstract + §5.4–5.5 + future-work 15 | $+200$ |
| 11. Blinded full-sim | promised, not run | run, graded, integrated | $+1700$ (overlaps with 3) |

The v2.7 source was the single authoritative version at original submission; the v2.8 supersession (`paper/digitaltwin_calibration_v2.8.md`) is the current authoritative version. Previous v2.x markdown files in `paper/` are kept for historical reference only and are not part of this submission.

---

## Addendum — v2.8 supersession (May 2026)

After original submission and reviewer dialogue, the simulator code path
was refactored across thirteen targeted sprints (catalogue in
`docs/SPRINT_1-13_CHANGELOG.md`):

- **Sprint 7** (the largest contributor): country-alias normalisation
  (UK ↔ GB, USA ↔ US) in the stakeholder relevance scorer. The v2.7 LLM
  scope detector emitted ISO-3166 short forms ("UK", "USA") while the
  stakeholder DB stores alpha-2 codes ("GB", "US"); previously this
  silently dropped Boris Johnson on Brexit briefs and Trump / Biden /
  Harris on US-politics briefs.
- **Sprints 1, 6, 8-13**: thread-safe SQLite + budget caps; realism-gate
  composition hints; tighter per-domain few-shot exemplars; engine
  off-by-one in event override; programmatic realism-gate override path;
  E2E test harness exercising 56 paper-scenario briefs.

The v2-discrepancy hierarchical SVI calibration was re-fit on the **same
42 empirical scenarios** with **identical hyperparameters** (3000 SVI
steps, lr 0.005, seed 42). Results:

| Group   | N  | MAE pre→post (pp)         | cov₉₀ pre→post           | CRPS Δ |
|---------|----|---------------------------|--------------------------|--------|
| OVERALL | 42 | 15.22 → **14.65** (−0.57) | 78.6% → **83.3%** (+4.8) | −0.34  |
| TRAIN   | 34 | 14.29 → **13.97** (−0.32) | 79.4% → **82.4%** (+2.9) | −0.10  |
| TEST    |  8 | 19.18 → **17.56** (−1.62) | 75.0% → **87.5%** (+12.5)| −1.35  |

Final SVI loss: 514.74 → 493.79.

**This addendum does not invalidate any v2.7 response above.** The
inference machinery (NUTS validation, SBC, Sobol, EnKF on Brexit,
contamination probe, blinding protocol, null-baseline benchmark) is
unchanged — every reviewer concern that targeted those parts of the
paper still has the same response. The only revisions touching v2.8
numbers are §3 Test Set Performance and §5.6 Multi-modal Comparison,
where the v2.7 column is preserved alongside the new v2.8 column for
direct comparability. The headline scientific claim (calibrated ABM
matches but does not dominate naive persistence in retrospective
trajectory space; operational value is in EnKF online assimilation +
benchmarking instrumentation) is unaffected.

The v2.7 PDF and `f3bf60e` artifact bundle remain available alongside
v2.8 for any reviewer who prefers to review against the originally
submitted version.
