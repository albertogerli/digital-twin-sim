# Submission packages

Two paper submissions, each with markdown source, LaTeX source, compiled PDF,
and cover letter. Figures are shared and re-generated from a single Python
script that operates on the production calibration code.

## Layout

```
submission/
├── Makefile                                — build everything: make / make figures / make pdfs / make clean
├── README.md                               — this file
│
├── figures/                                — shared, regenerated from data
│   ├── generate_figures.py                 — entry point (operates on core/dora/economic_impact.py)
│   ├── captions.md                         — caption per figure
│   ├── fig01_cost_vs_shock_loglog.{png,pdf}
│   ├── fig02_loo_hit_rates_by_mode.{png,pdf}
│   ├── fig03_per_category_gamma.{png,pdf}
│   ├── fig04_hmm_regime_posterior.{png,pdf}
│   ├── fig05_residuals_vs_shock.{png,pdf}
│   └── fig06_calibration_dataset_breakdown.{png,pdf}
│
├── paper1_oprisk/                          — Paper #1: op-risk power-law model
│   ├── paper.md                            — source (markdown)
│   ├── paper.tex                           — pandoc-generated LaTeX
│   ├── paper.pdf                           — compiled (xelatex, 29 pages, ~150KB)
│   ├── cover_letter.md                     — source
│   └── cover_letter.pdf                    — compiled (~40KB)
│
└── paper2_llm_abm/                         — Paper #2: LLM-agent ABM + contamination audit
    ├── paper.md
    ├── paper.tex
    ├── paper.pdf                           — compiled (xelatex, 18 pages, ~125KB)
    ├── cover_letter.md
    └── cover_letter.pdf
```

## Paper #1 — Op-risk power-law

**Title:** A Power-Law Cost Model for Operational-Risk Incident Reports: Calibration on 40 Historical Incidents and Implications for DORA Compliance

**Target venue:** *Journal of Operational Risk* (alt: *Quantitative Finance*, *Journal of Risk*).

**Headline claims:**

- Curated open dataset N=40 historical op-risk incidents 1998–2024 across 7 categories.
- Power-law cost model β·s^γ with category-specific γ̂ ranging 1.65–3.92, all > 1.
- LOO hit-rate within ±100% = 80% under power-law vs 35% linear pooled.
- Six-layer diagnostic stack (bootstrap, HC3, HMM, Hill, 2SLS-IV, fragility).
- Released as open benchmark.

**Status:** submission-ready.

## Paper #2 — LLM-agent ABM + contamination audit

**Title:** Calibrated LLM-Agent Models of Public Opinion: A Bayesian Framework with Contamination Auditing and a Reproducible Skill Floor

**Target venue:** *Journal of Artificial Societies and Social Simulation* (JASSS) (alt: *Journal of Computational Social Science*, *Computational Economics*).

**Headline claims:**

- Hierarchical Bayesian calibration on N=42 historical scenarios; 17.6 pp held-out MAE; 87.5% nominal CI coverage.
- Four-axis contamination probe + deterministic blinding protocol; mean leakage 0.721 → 0.000 on 11 high-leakage scenarios.
- Apples-to-apples sim-lift under contaminated and blinded arms: framework matches but does not beat persistence on retrospective political-referendum subset.
- Null-baseline benchmark released as a reproducible skill floor.

**Status:** submission-ready.

## Build

```bash
# One-time setup (TinyTeX is non-sudo, ~150MB)
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
brew install pandoc

cd submission
make            # regenerate everything
make pdfs       # papers + cover letters only
make figures    # figures only
make clean      # nuke generated artefacts
```

## Reproducibility

Every figure is derived from the same `core/dora/economic_impact.py` and
`shared/dora_reference_incidents.json` that the production estimator uses,
so the submission package and the production system are guaranteed to be
in sync. The `Makefile` is stateless: `make clean && make` rebuilds
the entire submission package from sources.
