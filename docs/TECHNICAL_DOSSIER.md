# DigitalTwinSim — Dossier tecnico completo

**Versione 0.8 · Maggio 2026**

Documento di riferimento per la trasformazione in brochure tecnica. Copre
tutto quello che il sistema fa, come lo fa, con quali equazioni e quali
coefficienti calibrati. La struttura segue l'architettura reale del
codice (`core/`, `briefing/`, `api/`, `frontend/`) — ogni sezione
include i path dei file rilevanti per chi vuole verificare.

Ogni numero in questo documento è verificabile su Git. Le scelte di
modellazione sono documentate nel paper accademico
(`paper/digitaltwin_calibration_v2.8.md`, 1988 righe) e nei JSON di
calibrazione (`shared/*.json`).

---

## Indice

1. [Posizionamento tecnico (cosa è, cosa non è)](#1-posizionamento-tecnico)
2. [Architettura a cinque strati](#2-architettura-a-cinque-strati)
3. [Multi-agent layer: tier, filtri, stakeholder graph](#3-multi-agent-layer)
4. [Opinion dynamics: il modello matematico](#4-opinion-dynamics)
5. [Calibrazione Bayesiana gerarchica (SVI / NUTS / EnKF)](#5-calibrazione-bayesiana)
6. [FinancialTwin ALM: fisica del bilancio](#6-financialtwin-alm)
7. [Calibrazione empirica del motore finanziario (v0.8)](#7-calibrazione-empirica)
8. [Backtest: corpus, validation, hit-rate](#8-backtest)
9. [Compliance enclave: BYOD + DORA + self-calibration](#9-compliance-enclave)
10. [Validation onesta e limiti dichiarati](#10-validation-onesta)
11. [Stack tecnologico + infrastruttura](#11-stack-tecnologico)
12. [Numerical fingerprint (one-page summary)](#12-numerical-fingerprint)

---

## 1. Posizionamento tecnico

### 1.1 Cosa il sistema effettivamente computa

DigitalTwinSim è un **gemello digitale narrativo + finanziario** che,
dato un brief in linguaggio naturale (es. *"Sella riduce il pass-through
del taglio BCE di 25 bps"*), produce in 8-15 minuti:

1. **Una traiettoria di posizioni** per N stakeholder reali (politici,
   regulator, peer CEO, giornalisti, cluster cittadini) sull'asse
   semantico [-1, +1] del topic, evolvente per 5-9 round.

2. **Un trail di post / dichiarazioni / commenti** che ciascuno
   stakeholder pubblicherebbe, con engagement metrics (like, repost,
   reply) e classificazione di canale (TV / press / social / forum).

3. **Una traiettoria di KPI bancari** (NIM, CET1, LCR, deposit balance,
   loan demand, BTP-Bund spread, FTSE MIB) coerente con vincoli
   regolatori reali (EBA Risk Dashboard, ECB DFR, Banca d'Italia).

4. **Un impatto azionario per-ticker** su ~190 titoli globali (T+1, T+3,
   T+7) con direzione (long/short), intervalli di confidenza, pair
   trade derivato empiricamente.

5. **Un report HTML stampabile** + dashboard interattiva + JSON
   tracciabile + (su request) export DORA XML per submission
   regolatoria.

### 1.2 Cosa il sistema NON è

- **Non è un modello previsionale.** Il paper accademico (v2.8, 8
  scenari held-out) misura il sistema contro forecaster naive
  (persistence / running-mean / OLS / AR(1)) e trova: *"the calibrated
  ABM matches but does not dominate naive persistence in retrospective
  trajectory space"*. Il valore operativo è in scenario exploration +
  EnKF online assimilation, non in open-loop forecast.

- **Non sostituisce il modello ALM core di una banca.** Affianca come
  what-if narrative layer.

- **Non è un'IA generativa pura.** È un sistema **ibrido fisico-narrativo**
  dove l'LLM gioca solo dove la narrativa lo richiede; i KPI bancari
  sono governati da fisica deterministica (CIR, deposit beta,
  elasticità, regressioni event-study calibrate).

### 1.3 Innovazioni tecniche distintive

1. **Mixture softmax gauge-fixed** di 5 forze in opinion dynamics
   (sezione 4) — replica gli studi classici di Hegselmann-Krause /
   DeGroot ma con calibrazione Bayesiana gerarchica sui coefficienti.

2. **Calibrazione empirica end-to-end del motore finanziario** (sezione 7)
   — sostituisce 4 costanti euristiche con coefficienti derivati da
   event study reali su 95 scenari × ~4 ticker.

3. **Cross-sector VAR(1) contagion network** (sezione 7.4) — 39 spillover
   edges direzionali next-day estratti da OLS pooled su 2.177 giorni
   di trading.

4. **Three-layer stakeholder filtering** (sezione 3.2) — score deterministico
   + LLM realism gate + Chain-of-Thought audit per i marginali. Audit
   trail di compliance per ogni inclusione/esclusione.

5. **BYOD enclave + DORA export + continuous self-calibration** (sezione 9)
   — i tre moat strategici per la vendita enterprise.

---

## 2. Architettura a cinque strati

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Briefing Layer  (briefing/)                                 │
│    Brief LLM-driven scoping → scope strutturato                │
│    Layer 1 (relevance score) → Layer 2 (realism gate)          │
│    Layer 3 (CoT reasoning audit per casi marginali)            │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Multi-Agent Layer  (core/agents/)                           │
│    Tier 1 — Elite (8-14 agenti): exec, peer CEO, ABI, BCE      │
│    Tier 2 — Institutional (6-10): media, lobby, tribunali      │
│    Tier 3 — Citizen Clusters (5-8): retail, PMI, millennials   │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Opinion Engine  (core/simulation/opinion_dynamics_v2.py)    │
│    5-force softmax mixture, gauge-fixed                        │
│    Forze: direct LLM | herd | anchor | social | event_shock    │
│    Calibrazione: NumPyro SVI hierarchical + EnKF data assim    │
└────────────────────────────────────────────────────────────────┘
        ↓ in lockstep ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. FinancialTwin ALM  (core/financial/twin.py)                 │
│    Stato: NIM, CET1, LCR, deposit balance, loan demand,        │
│    duration gap, hedging P&L, breach detection                 │
│    7 paesi (IT/DE/FR/ES/NL/US/GB) con dati live ECB/FRED/BoE   │
│    Coupling bidirezionale opinion ↔ financial                  │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Reporting & Audit  (core/simulation/reporting.py)           │
│    Report HTML print-to-PDF · dashboard React · JSON           │
│    Reasoning trace per audit compliance                        │
│    DORA XML export · BYOD audit log JSONL                      │
└────────────────────────────────────────────────────────────────┘
```

Ogni strato è **indipendente e testabile**. Il backend Python espone
30+ endpoint REST + Server-Sent Events per streaming round-by-round.
Il frontend Next.js consuma via fetch/EventSource.

---

## 3. Multi-Agent Layer

### 3.1 Tre tier di agenti

**Tier 1 — Elite agents** (`core/agents/elite_agent.py`):
- 8-14 agenti per simulazione
- Public figures reali con bio, key_traits, party_or_org, posizioni
  iniziali sui topic
- Output per round: 1-3 post strutturati con position shift, channel,
  engagement target
- Memoria persistente: ogni agente ricorda le proprie posizioni e
  citazioni passate, ricevute come context al round successivo

**Tier 2 — Institutional agents** (`core/agents/institutional_agent.py`):
- 6-10 organizzazioni (media outlet, lobby, tribunali, regulator)
- Behaviour archetype-driven (rational analyst, populist activist,
  bureaucratic process, etc.)
- Stesso output schema degli elite ma con voce istituzionale

**Tier 3 — Citizen swarm** (`core/agents/citizen_swarm.py`):
- 5-8 cluster demografici (retail urban, PMI provinciali, pensionati,
  Gen Z, etc.)
- Simulazione aggregata per cluster (no per-individuo)
- Output: sentiment medio + posizione drift + post virali rappresentativi

**Totale tipico**: 20-32 entità simulate per scenario, distribuite sui tre tier.

### 3.2 Three-layer stakeholder filtering

Il problema "chi commenta credibilmente questa decisione?" è la prima
fonte di errore in ogni multi-agent system. Senza filtering rigoroso il
sistema produce roster con Trump che commenta il pricing di Sella.

**Layer 1 — Score deterministico** (`briefing/relevance_score.py`):
- 6 componenti pesate:
  - `country_match` (peso 0.25): country dello stakeholder vs
    `scope.geography`
  - `sector_match` (peso 0.20): sector vs `scope.industries`
  - `semantic_similarity` (peso 0.20): cosine similarity tra brief e
    stakeholder bio via Gemini text-embedding-004 (768-dim, multilingual)
  - `brief_mention` (peso 0.15): named entity match nel brief
  - `archetype_whitelist` / `denylist` (peso 0.15): explicit allow/deny
    per (scope_tier, archetype)
  - `global_figure_penalty` (peso 0.05): foreign head-of-state on
    non-political brief → 1.0 penalty; in-country head-of-state on
    non-political → 0.7
- Threshold conservativo: 0.40
- Latenza: <1ms per stakeholder
- 22 unit test su 5 brief diversi (banking IT, IT/DE cross-border,
  banking DE, banking US, US politics)

**Layer 2 — LLM realism gate** (`briefing/realism_gate.py`):
- Audit Gemini STRICT in modalità REJECT-by-default
- Prompt con esempi nominati di global figure da rifiutare
- Cache verdict per (brief_hash, agent_id) — riutilizzo cross-sim
- Sempre attivo (synthesised default scope se Layer-0 fail)

**Layer 3 — Chain-of-Thought reasoning audit** (`briefing/reasoning_audit.py`):
- SOLO sui marginali in zona [0.30, 0.60] (~5-10 candidati per sim)
- Output strutturato: 4 step (topics_in_brief → agent_expertise →
  overlap_assessment → decision_rationale) + verdict + confidence
- **Trace verbatim del LLM salvato in JSON tracciabile** (compliance audit)

### 3.3 Stakeholder graph

DB di **744 stakeholder reali** organizzati per paese (IT/DE/FR/ES/UK/US),
categoria (politician, ceo, journalist, magistrate, central_bank,
industry_association, consumer_advocacy, regulator, …), tier (1-3).

Schema per stakeholder:
```yaml
id: carlo_messina
country: IT
archetype: bank_ceo
party_or_org: Intesa Sanpaolo
role: CEO
bio: "..."  # ~200 words
key_traits: [pragmatic, atlanticist, ESG-forward]
positions:
  - topic_tag: monetary_policy_dovish
    value: -0.42
    confidence: 0.78
relationships:
  - target_id: ignazio_visco
    strength: 0.65
    context: regulatory dialogue
affiliated_tickers: [ISP.MI]
```

Per il dominio finance Italia: 20 stakeholder verificati (Patuelli/ABI,
Panetta/Bankitalia, Codacons, Findomestic, Agos, Compass,
Carlo Messina/Intesa, Orcel/UniCredit, Cottarelli, De Bortoli,
Fubini, Savona/Consob, Signorini/IVASS, MEF, Altroconsumo).

---

## 4. Opinion dynamics

### 4.1 Il modello matematico

L'evoluzione delle posizioni segue una **mixture softmax di 5 forze**
gauge-fixed (`core/simulation/opinion_dynamics_v2.py`).

Per ogni agente i, al round t+1:

```
Δp_i = λ · Σ_k w_k(t) · F_k_i(t)
```

dove:
- `λ` è lo step size (calibrato ~0.15 per political, ~0.08 per finance)
- `w_k(t) = softmax(α_k + β_k · context_k(t))` sono i pesi normalizzati
  delle 5 forze (somma a 1, gauge-fixed)
- `F_k_i(t)` sono le 5 forze standardizzate (z-score) per l'agente i

### 4.2 Le 5 forze

**Forza 1 — `direct` (LLM-emitted shift)**
- Lo shift che l'LLM genera direttamente quando processa l'evento di round
- Standardizzato per agente: `(raw_shift - μ_agent) / σ_agent`
- Coefficient calibrato (political domain): `α_direct ≈ 0` (gauge fix)

**Forza 2 — `herd` (alignment to mean sentiment)**
- `F_herd_i = sentiment_mean(t) - p_i(t)`
- Tendenza ad allinearsi al sentiment medio del feed
- Coefficient: `α_herd = -0.176` (negativo = anti-herd in equilibrio)

**Forza 3 — `anchor` (rigidity toward initial position)**
- `F_anchor_i = (p_original_i - p_i(t)) · rigidity_i`
- Tendenza a tornare verso la posizione iniziale (più alta per agenti
  rigidi: politicians, regulator)
- Coefficient: `α_anchor = +0.297` (dominante)

**Forza 4 — `social` (weighted influence from followed agents)**
- `F_social_i = Σ_j (relationship_weight_ij · p_j(t)) - p_i(t)`
- Influenza ponderata dagli agenti che `i` segue
- Coefficient: `α_social = -0.105`

**Forza 5 — `event_shock` (exogenous event reaction)**
- `F_event_i = event_direction · event_magnitude · agent_relevance_i`
- Reazione all'evento esogeno emesso dall'event injector
- Coefficient: `α_event = -0.130`

### 4.3 Gauge fixing

Il softmax ha un grado di libertà ridondante (sommare una costante a
tutti i `α_k` non cambia i pesi). Il gauge fix forza `α_direct = 0`,
trasformando le altre 4 in **deviazioni relative dal direct shift**.
Questo è cruciale per l'identificabilità nella calibrazione Bayesiana.

### 4.4 Bounded confidence — perché NON l'abbiamo usato

Modelli di bounded confidence (Hegselmann-Krause, Deffuant-Weisbuch)
assumono un **threshold ε** sotto cui due agenti si influenzano e
sopra cui si ignorano. L'abbiamo provato in v1 e poi scartato in v2
perché:

1. Empiricamente, lo stakeholder graph di un brief reale ha relationship
   weights che variano in [0, 1] in modo continuo — il threshold
   crisp non riflette la realtà.
2. La soglia ε è notoriamente non-calibrabile via gradient (è
   non-differenziabile).
3. Il nostro `α_social` con weighting continuo fa il lavoro che
   bounded confidence farebbe, in modo differenziabile e calibrabile
   Bayesiano.

Il modello v2 può essere visto come una **generalizzazione continua**
del bounded confidence dove il "threshold" è implicito nel relationship
weight.

### 4.5 Standardizzazione con EMA

Le 5 forze hanno scale fisiche diverse. Standardizziamo via
exponential moving average per agente, con cold-start passthrough nei
primi 3 round (`core/simulation/opinion_dynamics_v2.py:Standardizer`).

```python
μ_t = (1 - α_ema) · μ_{t-1} + α_ema · F_t
σ_t = sqrt((1 - α_ema) · σ²_{t-1} + α_ema · (F_t - μ_t)²)
F_standardized = (F_t - μ_t) / max(σ_t, ε)
```

con `α_ema = 0.15`. Il test di permutation invariance conferma che
l'ordering degli agenti non altera le forze standardizzate
(max abs difference < 6×10⁻⁸ sotto random permutation).

### 4.6 Bounds

Posizioni clamped in `[-1, +1]` ad ogni step. Gli shock estremi
(es. `event_shock` con magnitude 1.0) vengono assorbiti dal cap senza
overflow, verificato in `tests/test_opinion_dynamics_v2.py`.

---

## 5. Calibrazione Bayesiana

### 5.1 Modello gerarchico a tre livelli

I coefficienti `α_k` non sono fissi: sono **distribuzioni** fittate
Bayesiane su 42 scenari empirici across 10 domini (2011-2023).

Struttura:
```
α_k_global ~ Normal(μ_prior_k, σ_prior_k)
α_k_domain ~ Normal(α_k_global, σ_domain_k)   [10 domini]
α_k_scenario ~ Normal(α_k_domain, σ_scenario_k) [42 scenari]
```

Con readout discrepancy esplicito (Kennedy-O'Hagan):
```
b_d ~ Normal(0, σ_b_d)   # bias per domain
b_s ~ Normal(0, σ_b_s)   # bias per scenario
```

### 5.2 Inference — SVI vs NUTS

**SVI (Stochastic Variational Inference)** in produzione:
- AutoLowRankMultivariateNormal guide
- 3000 step a learning rate 0.005
- Final loss 493.79 (post-Sprint 15 recalibration)
- Tempo: ~8 min su CPU singola
- Vantaggio: 10× più veloce di NUTS, ottimo per re-calibration ricorrente

**NUTS (Hamiltonian MC)** per validation:
- 4 chain × 1000 sample, 1000 warmup
- R-hat < 1.01 su tutti i parametri principali
- Usato per Simulation-Based Calibration (SBC)
- 6/6 test SBC confermano well-specification (KS uniformity p > 0.20)

**Trade-off documentato nel paper §6.2**: SVI sottostima l'incertezza
sui parametri deboli (α_social, α_event) di un fattore 5-13× rispetto a
NUTS. È un finding centrale del paper, non un side caveat — bound la
reliability di tutti gli intervalli di credibilità SVI.

### 5.3 Sensitivity analysis (Sobol)

Variance-based sensitivity analysis su 5000 sample LHS:
- `herd` factor: total Sobol index `S_T = 0.55` (dominante)
- `anchor` factor: `S_T = 0.45`
- altre forze: `S_T < 0.20`

Conferma che il sistema è governato principalmente da herd dynamics +
anchor rigidity, coerente con la letteratura sociologica.

### 5.4 EnKF per online assimilation

L'Ensemble Kalman Filter (`core/orchestrator/`) augment lo stato come
[θ, z] dove θ sono i pesi delle forze e z sono le posizioni. Permette
data assimilation di osservazioni reali (sondaggi, dati di mercato)
senza re-fit completo.

Su un caso study Brexit (in-sample): final-round error ridotto a 1.8 pp
(77% improvement vs last-poll baseline) con 6 polls assimilati.

**Out-of-sample validation**: pending (uno dei limiti dichiarati nel paper).

---

## 6. FinancialTwin ALM

### 6.1 Variabili di stato

Per ogni round t (`core/financial/twin.py`):

| Variabile | Significato | Aggiornamento |
|---|---|---|
| `NIM_t` | Net Interest Margin (%) | f(rate_path, deposit_β, asset_repricing) |
| `CET1_t` | Common Equity Tier 1 ratio (%) | CET1_{t-1} − ΔRWA_t / capital + retained_earnings |
| `LCR_t` | Liquidity Coverage Ratio (%) | f(deposit_outflow, HQLA, contractual_inflows) |
| `deposit_balance_t` | Total deposits (€M) | (1 - runoff_rate) · deposit_balance_{t-1} |
| `loan_demand_t` | Loan demand (€M) | f(elasticity, rate_change, sentiment) |
| `duration_gap_t` | Asset-liability duration mismatch | exogenous baseline + repricing |
| `breach_flags_t` | Set of regulatory thresholds breached | {LCR < 100%, CET1 < min, NIM ≤ 0} |

### 6.2 Parametri di default (fonte: pubblicazioni 2024-2025)

| Parametro | IT default | Fonte |
|---|---|---|
| Deposit β sight | 0.45 | ECB Economic Bulletin 2025 |
| Deposit β term | 0.75 | ECB Economic Bulletin 2025 |
| Consumer loan elasticity | -1.7 (median IT -2.14) | Bonaccorsi/Magri micro-data IT |
| NIM baseline | 1.70% (EU Q2 2025: 1.58%) | EBA Risk Dashboard Q2 2025 |
| CET1 baseline | 16% | EBA Risk Dashboard Q3 2025 (EU 16.3%) |
| LCR baseline | 170% | EBA Risk Dashboard Q3 2025 (EU 160.7%) |
| Mortgage var/fix mix | 50/50 stock, 20/80 nuovi | Banca d'Italia 2025 |
| Deposit runoff cap | 4%/round (no panic) | letteratura empirica IT |

### 6.3 Country dispatch automatico

`core/financial/country_params.py` mappa il `country` field del brief a
parametri specifici per 7 paesi:

| Paese | Deposit β sight/term | CET1 baseline | Note distintive |
|---|---|---|---|
| IT | 0.45 / 0.75 | 16.0% | Default canonico |
| DE | 0.30 / 0.65 | 18.9% | Sparkasse drag β |
| FR | 0.40 / 0.70 | 17.0% | Livret A regulation |
| ES | 0.55 / 0.80 | 13.9% | Post-2012, mortgage 70% var |
| NL | 0.50 / 0.75 | 19.4% | Mortgage 95% fixed |
| US | 0.40 / 0.85 | 14.3% | Mortgage securitised, Fed Funds |
| GB | 0.35 / 0.75 | 17.9% | Ring-fencing, BoE Bank Rate |

### 6.4 Modello stocastico tassi (CIR 1-factor)

```
dr = κ(θ - r) dt + σ√r dW
```

Default Euribor 3M area euro 2024-2026:
- κ = 0.30 (mean-reversion speed)
- θ = 2.5% (long-run mean)
- σ = 1.5% (volatility)
- r₀ = 2.4% (initial)

Discretizzazione Euler-Maruyama con clamp positivo. Feller condition
`2κθ > σ²` rispettata (2 × 0.30 × 0.025 = 0.015 vs 0.015² = 0.000225).
Test `tests/test_financial_twin.py::test_cir_stays_positive_over_long_run`
verifica che il path resta positivo su 1000 simulazioni × 100 step.

### 6.5 Coupling bidirezionale opinion ↔ financial

**Opinion → Financial** (weighted by stakeholder exposure):
- Negative depositor-weighted opinion → ↑ deposit runoff
- Negative borrower-weighted opinion → ↓ loan demand
- Competitor agents con posizione negativa verso bank → ↑ market share grab

**Financial → Opinion** (`FeedbackSignals`):
- `nim_anxiety`, `cet1_alarm`, `runoff_panic`, `competitor_pressure`,
  `rate_pressure`
- Categorici (non numerici) — iniettati come context nei prompt agente
  del round successivo
- L'LLM legge "[CET1 alarm 0.74]" e aggiorna la sua posizione
- **Cruciale per BYOD**: i numeri esatti di CET1/LCR non lasciano mai il
  perimetro del cliente; solo i feedback signal categorici raggiungono
  l'LLM provider

### 6.6 Stress templates regolatori

`core/financial/stress_templates.py`:
- **EBA Baseline 2025**: ciclo normale, parametri standard
- **EBA Adverse 2025**: shock +250 bps cumulativo, deposit β
  0.55/0.85, elasticità -2.1, CET1 alarm 13%, breach detection
  LCR < 100% / CET1 < min
- Test `test_eba_adverse_triggers_breach_at_round_1` verifica che il
  template adverse triggera breach come atteso

### 6.7 Live data ingestion

Connettori opzionali, free public endpoints (no enterprise tier $5K/mese):
- **ECB Statistical Data Warehouse**: Euribor 3M, ECB DFR, BTP/Bund 10Y
  yield → spread
- **FRED (St. Louis Fed)**: Effective Fed Funds Rate, US 10Y Treasury,
  UK 10Y Gilt
- **Bank of England Statistical Database**: BoE Bank Rate
- Cache 24h TTL, fallback graceful su default literature-based se rete
  unreachable. Auto-refresh al boot del FinancialTwin per ogni sim.

---

## 7. Calibrazione empirica del motore finanziario (v0.8)

Quattro costanti precedentemente euristiche del motore di impatto
finanziario sono state sostituite con coefficienti **calibrati su event
study reali** del corpus storico (95 scenari × ~4 ticker, 310
osservazioni event×ticker valide dopo noise floor 20bps).

| Costante legacy | Valore euristico | Sostituzione empirica |
|---|---|---|
| `panic_mult = exp(cri × 1.5)` | 2.3x → 4.0x via cri | **median per CRI bin** (cri 0.4–0.7: 2.7x · 0.7–0.85: 5.5x · ≥0.85: 10.9x) |
| `recovery_factor = 0.5 + 0.1×i` | 0.5–0.7 (recovery) | **T+3/T+1 empirica** per (intensity_bin × sector), 23 celle |
| `escalation_factor = 1.0 + 0.1×(i−2)` | 1.0–1.6 (escalation) | come sopra |
| `t7 = t3 × 1.3 + ...` | 1.3 (persistenza) | **T+7/T+1 empirica** per (intensity_bin × sector) |
| `CRISIS_PAIR_TRADES[topic]` | dizionario cablato | **derivato da matrice correlazione globale** |

### 7.1 Matrice di correlazione globale

`shared/correlation_matrix.json`, costruito da
`scripts/compute_correlation_matrix.py`:

- 180 ticker (190 dell'universe meno 10 delisted/illiquid)
- 2018-01-01 → today (2.177 trading days)
- Pearson correlation su daily log returns
- Top-K (K=8) edges per node con threshold |r| ≥ 0.30
- 14 community via Louvain (resolution 1.1) sulla matrice |r|-pesata

Esempi di community detection interpretable:
- **Community 0** (37 nodes): Italian banks/insurance (UCG, ISP, BMPS,
  BAMI, MB, BPE, FBK, G, UNI, …)
- **Community 1** (36 nodes): US financials + media (JPM, BAC, GS, MS,
  C, WFC, DIS, …)
- **Community 3** (18 nodes): US Big Tech + TSMC (AAPL, MSFT, GOOGL,
  AMZN, META, NVDA, TSLA, TSM, …)
- **Community 4** (17 nodes): Japanese stocks (Toyota, Sony, MUFG,
  Hitachi, Tokyo Electron, …)
- **Community 5** (13 nodes): Global energy majors (XOM, CVX, ENI, TTE,
  SHEL, BP, RIO, …)

### 7.2 Sector beta empirici per (country, sector)

`shared/sector_betas_empirical.json`, costruito da
`scripts/recalibrate_sector_betas.py`:

Per ogni (country, sector) bucket:
1. Pool daily log-returns dei ticker membri
2. OLS regression vs country index (^GSPC, ^FTSE, ^GDAXI, ^FCHI,
   FTSEMIB.MI, ^N225, ^HSI, ^SSMI, ^AEX, ^IBEX, 000001.SS):

```
r_bucket(t) = α + β · r_country_index(t) + ε(t)
```

86 celle across 20 paesi. Notable:
- **US tech β = 1.26** (15 ticker, R² = 0.85 — fit più robusto)
- **IT banking β = 1.22** (7 ticker, R² = 0.72) — era 1.85 hand-coded,
  sopravvalutato del 50%
- **GB defense β = 1.65** (n=1, R² = 0.26)
- **FR defense β = 1.45** (R² = 0.53)
- **DE banking β = 1.36** (R² = 0.45)
- **CN banking β = 0.57** (defensive SOE behaviour, R² = 0.19)
- **JP banking β = 0.83** (n=1, R² = 0.38)

`MarketContext.get_beta()` consulta empirico → fallback statico.

### 7.3 Impulse response empirica T+3/T+1, T+7/T+1

`shared/impulse_response_coefficients.json`, costruito da
`scripts/calibrate_impulse_response.py`:

Per ogni (intensity_bin, sector) bucket, weighted-mean ratio dei
realized log-returns con weight = |t1| (per dare peso alle big-shock
observations):

```
ρ_3 = E[t3 / t1]   bin × sector
ρ_7 = E[t7 / t1]   bin × sector
```

Bin: low (< 2 intensity) / mid (2-4) / high (≥ 4). 23 celle + 3 pooled
ALL fallback per bin.

**Headline finding**: la V-shape recovery del legacy
(`recovery_factor = 0.5 + 0.1·i`) è **empiricamente sbagliata**. Real
markets post-shock NON tornano del 50% in 3 giorni — continuano a
trendare:

- low intensity: T+3/T+1 ≈ **1.03** (legacy diceva 0.5 — off by 2x)
- mid intensity: T+3/T+1 ≈ **1.09**
- high intensity: T+3/T+1 ≈ **1.37**, T+7/T+1 ≈ **1.28**

### 7.4 Panic multiplier mediano per CRI bin

`shared/panic_multiplier_calibration.json`, costruito da
`scripts/calibrate_panic_multiplier.py`:

Per ogni (event × ticker) del corpus:
1. Compute |realized T+1| (yfinance log-return × 100)
2. Compute |linear-prediction T+1| usando le metriche reali del
   corpus (engagement, wave, neg_inst_pct, neg_ceo, polar_vel) — NOT
   neutral defaults
3. Form ρ = |realized| / |predicted|
4. Per CRI bin: median ρ

Risultato:

| CRI bin | n | median ρ | mean ρ | legacy `exp(cri·1.5)` |
|---|---|---|---|---|
| mid (0.4–0.7) | 28 | **2.68x** | 4.82x | 2.28x ✓ vicino |
| high (0.7–0.85) | 52 | **5.54x** | 10.82x | 3.20x ↑70% sotto |
| extreme (≥0.85) | 49 | **10.86x** | 42.07x | 3.97x ↑170% sotto |

Median (non mean) preferita perché distribuzione heavy-tailed (Lehman,
COVID dominano la mean).

### 7.5 Cross-sector VAR(1) contagion network

`shared/sector_contagion_var.json` + UI `SectorContagionGraph`,
costruito da `scripts/build_sector_contagion_var.py`:

Per ogni (sector_i, sector_j) ordered pair:

```
r_j(t) = α_ij + β_ij · r_i(t-1) + ε(t)
```

su sector basket returns (equally-weighted) 2018-2025. 39 spillover
edges direzionali significativi (|β| ≥ 0.05, |t| ≥ 2.0). Esempi:

- **tech → infrastructure** β = +0.12 (t = 5.7) — capex propagation
- **tech → automotive** β = +0.12 (t = 4.5) — semis / EV exposure
- **food_consumer → luxury** β = +0.13 (t = 3.3) — consumer cycle
- **healthcare → media** β = -0.14 (t = -2.5) — defensive rotation
- **real_estate → luxury** β = +0.10 (t = 4.5) — wealth effect
- **banking → infrastructure** β = +0.07 (t = 3.9)

### 7.6 Pair trade derivation empirica

`core/orchestrator/correlation_lookup.py`:

```python
def derive_pair_trade(seed_tickers, k_short=4, k_long=4, min_corr=0.30):
    # SHORT leg: top-K positively correlated globals to seed basket
    # LONG leg:  top-K negatively correlated globals
    ...
```

Esempio shock banche italiane → top correlated empirici:
`BPE.MI / BNP.PA / GLE.PA / MB.MI / DBK.DE / G.MI / PST.MI / UNI.MI`
(tutte r ≥ +0.61). Sostituisce il vecchio `CRISIS_PAIR_TRADES` cablato
a mano.

---

## 8. Backtest

### 8.1 Corpus

`backtest_scenarios.py` + `backtest_scenarios_global.py`:
- **60 eventi italiani** (2011-2025): Berlusconi convictions, Renzi
  referendum, Conte/Mattarella, COVID Italian lockdown, Spread Crisis
  2011, Draghi resignation, EU Budget Standoff Conte vs Brussels,
  bank tax 2023, Monte Paschi bail-out, ECB hikes, Greek Referendum,
  Brexit IT impact, Credit Suisse / Italian Contagion, ILVA, ENI
  Corruption, Vivendi vs Mediaset, Generali Board Fight, ECB First
  Hike Post-COVID, …

- **35 eventi globali** (`backtest_scenarios_global.py`, generato da
  `scripts/build_global_corpus.py`): Lehman 2008, SVB 2023, Trump
  Tariff 2019, COVID Lockdown US 2020, FTX collapse, Trump 2024
  re-election, Brexit Vote 2016, Truss Mini-Budget 2022, Macron 2017,
  Le Pen 1st-Round Surge, Credit Suisse / UBS rescue 2023, EU Energy
  Crisis 2022, Fukushima 2011, HK Extradition Bill 2019, Evergrande
  2021, China Tech Crackdown 2021, Korea Martial Law 2024, Bolsonaro
  2018, Lula 2022, Milei 2023, Russia Invades Ukraine 2022,
  Israel-Hamas 2023, Saudi-Russia Oil Price War 2020, Boeing 737 MAX
  2019, VW Dieselgate 2015, Meta Q4 2022 Crash, Nvidia AI Earnings
  2023, OpenAI ChatGPT Launch, Tesla Q3 2018, Archegos 2021, Fed First
  Hike 2022, ECB First Hike 2022, BoJ YCC Tweak 2022, Apple App Store
  Ruling 2021, Disney+ Subscriber Miss 2022.

**95 scenari totali**. Ogni scenario ha:
- name, date_start, date_end, brief (Italian text)
- topics, sectors lists
- engagement_score, contagion_risk, active_wave, polarization,
  polarization_velocity, negative_institutional_pct, negative_ceo_count
- verify_tickers (basket di validazione)
- expected_directions {ticker: "up"|"down"|"flat"}
- notes (free-text validation)

### 8.2 Hit-rate validato (corpus globale)

`scripts/build_global_corpus.py` valida ogni scenario via yfinance:

| Country | n events | Hit-rate T+1 |
|---|---:|---:|
| GB | 2 | 100% (6/6) |
| HK | 1 | 100% (3/3) |
| JP | 2 | 100% (7/7) |
| AR | 1 | 100% (2/2) |
| CN | 2 | 88% (7/8) |
| US | 17 | 74% (43/58) |
| CH | 1 | 50% (2/4) |
| DE | 4 | 56% (9/16) |
| FR | 2 | 40% (2/5) |
| KR | 1 | 33% (1/3) |
| BR | 2 | 17% (1/6) |
| **TOTAL** | **35** | **70.3% (83/118)** |

I miss restano in corpus come **disagreement verificabili** (es. ChatGPT
launch non ha tankato Google nella prima settimana — il consenso comune
era sbagliato).

### 8.3 Validation A/B PRE/POST empirical wiring

`scripts/validate_empirical_wiring.py` runs the full scorer twice
(once with empirical JSONs present, once with them temporarily
renamed → falls back to heuristics) on the same corpus, compares MAE.

Risultato (n=40 valid event×ticker observations):

| Metric | PRE (empirico) | POST (heuristic) | Δ |
|---|---:|---:|---:|
| MAE T+1 | 1.92pp | 1.96pp | -0.05pp |
| MAE T+3 | 2.87pp | 3.36pp | **-0.50pp ✓** |
| MAE T+7 | 8.08pp | 5.25pp | **+2.83pp ✗** |
| Direction accuracy | **58%** (15/26) | 54% (14/26) | **+3.8pp ✓** |

Empirico vince su direzione (+4pp) e MAE T+3, perde su MAE T+7 a causa
della distribuzione bimodale (eventi flash mean-revertono, eventi
slow-burn accumulano). Honest finding documentato in
`docs/EMPIRICAL_VALIDATION_REPORT.md`.

### 8.4 Backtest dashboard `/backtest`

KPI strip:
- Events tested: **64** (2019-2025) — corpus subset usato per il
  dashboard
- Direction hit-rate: **61%** (148/242 ticker × event)
- Mean abs. error T+1: **2.14%**
- Sharpe (paper): **1.84** vs benchmark 0.94
- R² (sentiment → price): **0.41** across 249 ticker

Hit-rate by domain:

| Dominio | Hit-rate | MAE | N |
|---|---:|---:|---:|
| Labor / Industrial | 83% | 1.2 | 8 |
| Banking / Financial | 78% | 2.7 | 7 |
| Immigration / Social | 100% | 0.6 | 2 |
| Healthcare / COVID | 70% | 2.8 | 2 |
| Media / Tech | 67% | 1.0 | 3 |
| Energy / Environment | 67% | 2.3 | 3 |
| Fiscal / EU Standoff | 56% | 2.6 | 12 |
| Corporate / Other | 54% | 1.7 | 6 |
| Political Crisis | 48% | 2.7 | 15 |

UI: `/backtest` page con scatter plot Predicted vs Actual T+1,
calibration chart, ticker leaderboard, scenario table, **GlobalContagionGraph**
(180 ticker D3 Canvas + Louvain coloring), **SectorContagionGraph**
(VAR(1) directed edges).

---

## 9. Compliance enclave

### 9.1 BYOD enclave (`core/byod/`)

**Architettura**: il `FinancialTwin` computa LCR/NIM/CET1/etc. in
locale ed emette `FeedbackSignals` **categorici** (non numerici) ai
prompt agente. Il sanitizer è defense-in-depth contro future code
regression.

**Detector regex** (6 categorie):

| Category | Match pattern (esempio) | Replacement |
|---|---|---|
| `currency` | `€2.4M`, `$500K`, `2.5 milioni di euro` | `[currency-amount]` |
| `financial_metric` | `LCR 168%`, `CET1 14.2%`, `NIM 1.85%` | threshold-aware: `[LCR healthy: above 100%]` / `[LCR breaching: below 100%]` / `[CET1 healthy: above 12%]` / `[CET1 tight: below 12%]` / `[NIM above EBA median]` |
| `client_id` | `client-12345`, `customer ID xxx` | `[client-id]` |
| `iban` | `IT60 X05428 11101 ...` | `[IBAN]` |
| `benchmark_value` | `Euribor 3M at 2.4%`, `BTP-Bund 180bps` | `[Euribor at level]` / `[BTP-Bund spread at level]` |
| `large_amount_in_context` | `deposit balance reached 12,500,000` | `[large-amount]` |

**4 mode** (env var `BYOD_MODE`):
- `OFF` (default): passthrough
- `LOG`: detect + audit, no modify
- `STRICT`: detect + redact + audit (production)
- `BLOCK`: detect + raise `BYODLeakError` (paranoid)

**Audit log** JSONL append-only a `outputs/byod_audit.jsonl`:
```json
{"ts":"2026-05-04T11:23:45Z","site":"llm:agent_round.elite","mode":"STRICT",
 "raw_chars":4823,"sanitized_chars":4791,
 "patterns":[{"category":"financial_metric","count":1}],"tenant":"sella-prod"}
```

**Hook** in `BaseLLMClient.generate()` pre-flight (prima del semaforo).

**Test coverage**: 20 unit test per detector + mode + threshold label +
non-financial preservation (poll percentage, polarization score, agent
position non vengono redatti).

### 9.2 DORA Major Incident Report (`core/dora/`)

**Schema** Pydantic approssima EBA/EIOPA/ESMA Joint Committee Final
Report JC 2024-43 (Luglio 2024) per Reg. (EU) 2022/2554 Art. 19-20.

**Classification helper** (`core/dora/classification.py`): mapping
deterministico simulator metrics → 7 livelli qualitativi DORA Annex I:

```python
classify_from_simulation(
    customers_affected=420_000,    # → "high" (bucket [10k, 1M))
    economic_impact_eur=2_400_000, # → "1m-10m" band
    countries_affected=2,          # → "national"
    polarization_peak=7.5,         # → "high" reputational
    viral_posts_count=12,          # → "high" reputational
    data_records_lost=0,           # → "low" data
    affected_core_functions=2,     # → "critical"
    downtime_hours=4.5,            # > 2h → is_major
)
```

**XML emitter** (`core/dora/exporter.py`): zero-dependency (solo
`xml.etree.ElementTree` stdlib), namespace
`urn:eu:europa:dora:incident:report:1.0`, output pretty-printed.
Three ReportType: INITIAL (24h), INTERMEDIATE (72h), FINAL (1 month).

**Test**: 15 unit test per classification thresholds, schema construction,
XML well-formedness, namespace presence, mitigation/communications
sections, final-report extras only when ReportType=FINAL.

**Endpoint API** `/api/compliance/dora/export/{sim_id}` → download XML
ready per upload portale Banca d'Italia.

### 9.3 Continuous self-calibration (`core/calibration/`)

**Pipeline ricorrente**:
1. `fetch_recent_news(watchlist)` via yfinance Ticker.news
2. `infer_crisis_metrics(headlines)` heuristic CRI/intensity da
   keyword positivi/negativi (IT + EN)
3. `predict_returns(tickers, cri, intensity)` lightweight (no LLM)
   usando β empirici + impulse response + panic mult
4. `record_forecast(run)` SQLite registry idempotente
5. T+1/T+3/T+7 trading days dopo: `evaluate_pending(horizon)` confronta
   con realized yfinance, scrive drift log JSONL

**CLI scheduler**:
```bash
python scripts/continuous_calibration.py forecast
python scripts/continuous_calibration.py evaluate --horizon 1
python scripts/continuous_calibration.py evaluate --horizon 3
python scripts/continuous_calibration.py evaluate --horizon 7
python scripts/continuous_calibration.py report
```

**Default watchlist** (configurabile): UCG.MI, ISP.MI, ENI.MI, ENEL.MI,
STLAM.MI, G.MI.

**Costo**: $0/mese (lightweight, no LLM call). $15/mese se `--llm`
flag abilitata (1 sim/giorno × $0.50 LLM × 30).

**Test coverage**: 14 unit test (heuristic CRI, predict, SQLite
round-trip + idempotency, evaluate_pending mocked yfinance, drift log
row, running summary).

**Frontend**: tab "Self-calibration" in `/compliance` con loop status,
MAE per horizon (color-coded direction acc T+1: green ≥60%, amber
50-60%, red <50%), per-ticker breakdown, ultime 30 evaluations.

---

## 10. Validation onesta

### 10.1 Paper finding (v2.8)

Dal paper `paper/digitaltwin_calibration_v2.8.md`:

> "The calibrated ABM matches but does not dominate naive persistence
> in retrospective trajectory space, and the framework's operational
> value is strongest under EnKF online assimilation rather than as an
> open-loop retrospective forecaster."

**Numeri held-out (8 scenari test)**:
- MAE: **17.6 pp** (full) / **12.6 pp** (verified, escludendo Archegos
  che ha ground truth flagged unreliable)
- RMSE: 24.7 / 14.3
- Coverage 90%: **87.5%** test / 79.4% train (ben calibrato)
- Median AE: 11.9 pp / 9.2 pp
- CRPS: 15.4 / 8.6

**Null-baseline benchmark** (Sezione 6.6 paper): contro 4 standard
forecaster (persistence, running-mean, OLS linear trend, AR(1)) su 43
scenari empirici, con Diebold-Mariano + Harvey-Leybourne-Newbold
small-sample correction:
- Naive persistence è strong baseline: mean RMSE 0.038 (3.8 pp)
- OLS linear trend beats persistence con p < 0.05 su solo 4-6 / 43
  scenari
- Domain skill eterogeneo: political persistence RMSE 0.012,
  financial 0.063 (5× spread)

### 10.2 Empirical wiring validation (sezione 8.3)

Ricalibrazione end-to-end del motore finanziario migliora direction
accuracy (+3.8pp) ma regressa T+7 MAE (+2.83pp) per eventi slow-burn.
Causa identificata: distribuzione bimodale che la mediana per bin perde.
Honest finding documentato, fix bimodal per T+7 in roadmap v0.9.

### 10.3 Diagnostic OLS finding

`scripts/calibrate_intensity_formula.py` ha rivelato che le metriche
`wave` e `neg_inst_pct` del corpus conflate due dimensioni distinte:
"velocità evento" e "depth crisi". L'OLS ritorna segni invertiti su
questi parametri. Il file `intensity_formula_coefficients.json` è
marcato `_status: "diagnostic-only"` e NON viene consumato in
produzione. Pubblicato come finding scientifico, non silenziato.

### 10.4 Limiti dichiarati (compliance honesty box)

1. **Non è un modello previsionale** — gli scenari sono plausibili,
   non profezie.
2. **Calibrazione finance ha RMSE 0.063** vs 0.012 del political domain.
3. **L'LLM è non-differenziabile** — la calibrazione Bayesiana usa uno
   shadow simulator differenziabile per JAX gradient flow.
4. **Stakeholder DB curato manualmente** — 744 stakeholder reali ma con
   coverage variabile per dominio. Richiede ~5h/mese manutenzione.
5. **Live data dipende da endpoint pubblici** — ECB SDW e BoE possono
   avere downtime; fallback graceful su default ma in quei casi i
   numeri non sono "live".
6. **Compliance regolatoria** — il sistema NON ha (ancora) certificazione
   ESMA/Bankitalia. Strumento di **pre-deliberation**, non di
   **regulatory submission** (ad eccezione del DORA XML che è formato
   conformante ma non ancora XSD-validato).
7. **BYOD sanitizer è regex-based**, non semantico. Non cattura
   "twelve million euros" scritto a parole. Per paranoid deployments
   serve aggiungere semantic NER pass.
8. **Self-calibration loop**: il moat data network effect richiede 12+
   mesi di operatività continua. Oggi infrastruttura pronta, history
   da costruire.

---

## 11. Stack tecnologico

| Layer | Tecnologia | Versione |
|---|---|---|
| Backend | Python | 3.11 |
| API | FastAPI + asyncio + uvicorn | latest |
| LLM provider | Gemini 2.0 Flash Lite (default), Claude Haiku, OpenAI gpt-5.4-mini | configurabili via env |
| Calibrazione Bayesian | JAX + NumPyro | latest |
| Data assimilation | Custom EnKF (JAX) | — |
| Embeddings | Gemini text-embedding-004 (768-dim, multilingual) | — |
| Persistenza | PostgreSQL (Railway) + SQLite locale per audit/drift logs + volume disco per checkpoint | — |
| Market data | yfinance + ECB SDMX + FRED + BoE | latest |
| Frontend | Next.js 14 (App Router) + TypeScript + D3.js + Recharts + Framer Motion | 14.2.35 |
| Auth | HMAC-SHA256 signed cookies via Web Crypto (Edge Runtime) | — |
| Tests | pytest + asyncio-pytest, ~200 unit test, ~30s full suite | latest |
| Deploy | Railway (backend, single worker, PostgreSQL) + Vercel (frontend) | — |
| Cost tracking | UsageStats per-component, $/sim ~ $0.13–0.50 | — |

---

## 12. Numerical fingerprint

One-page reference dei numeri chiave. Verificabile su Git.

### 12.1 Calibrazione opinion dynamics (paper v2.8)

| Parametro | Valore | Note |
|---|---|---|
| α_direct | 0 | gauge fix |
| α_herd | -0.176 | anti-herd in equilibrio |
| α_anchor | +0.297 | dominante |
| α_social | -0.105 | weak |
| α_event | -0.130 | weak |
| Sobol S_T herd | 0.55 | dominante |
| Sobol S_T anchor | 0.45 | secondario |
| Test MAE | 17.6 pp | full N=8 |
| Test MAE verified | 12.6 pp | excl. Archegos |
| Coverage 90% test | 87.5% | well-calibrated |
| SVI under-dispersion | 5-13× narrower than NUTS | weak parameters |

### 12.2 FinancialTwin defaults (IT)

| Parametro | Valore | Fonte |
|---|---|---|
| Deposit β sight | 0.45 | ECB |
| Deposit β term | 0.75 | ECB |
| Consumer loan elasticity | -1.7 | Banca d'Italia |
| NIM baseline | 1.70% | EBA |
| CET1 baseline | 16% | EBA |
| LCR baseline | 170% | EBA |
| CIR κ | 0.30 | mean-reversion speed |
| CIR θ | 2.5% | long-run mean |
| CIR σ | 1.5% | volatility |

### 12.3 Calibrazione empirica (v0.8, Sprint 71-83)

| Artefatto | Cardinalità |
|---|---|
| Cross-correlation matrix | 180 ticker × 2.177 days, Pearson |
| Louvain communities | 14 |
| Sector beta cells | 86 (across 20 paesi) |
| Impulse response cells | 23 + 3 pooled fallback |
| Panic mult bins | 3 (mid/high/extreme) + 1 fallback (low) |
| VAR(1) spillover edges | 39 directional |
| Backtest corpus | 95 events (60 IT + 35 global) |
| Validation observations | 310 (event × ticker) |
| Hit-rate T+1 (global corpus) | 70.3% |

### 12.4 Compliance moats (Sprint 84-94)

| Moat | Files | Tests |
|---|---|---|
| BYOD sanitizer | core/byod/sanitizer.py | 20 |
| DORA exporter | core/dora/{schema,exporter,classification}.py | 15 |
| Self-calibration | core/calibration/continuous.py + scripts/continuous_calibration.py | 14 |

### 12.5 Validation A/B empirico vs heuristic

| Metric | Empirico | Heuristic | Δ |
|---|---:|---:|---:|
| MAE T+1 | 1.92pp | 1.96pp | -0.05 |
| MAE T+3 | 2.87pp | 3.36pp | -0.50 ✓ |
| MAE T+7 | 8.08pp | 5.25pp | +2.83 ✗ |
| Direction acc | 58% | 54% | +3.8pp ✓ |

### 12.6 Cost economics

| Voce | Costo |
|---|---|
| Per simulazione (5 round, 25 agenti) | $0.13 – $0.50 |
| Infrastruttura cloud (startup) | $20 – $100 / mese |
| Self-calibration loop (lightweight) | $0 / mese |
| Self-calibration loop (--llm) | ~$15 / mese |
| Per simulation enterprise tier (Strategic SLA) | bundled in licenza |

---

## Appendice A — File path per verifica

Per ogni claim del documento, il file Python rilevante (per la
brochure: si può rimuovere questa appendice):

- Opinion dynamics: `core/simulation/opinion_dynamics_v2.py`
- Calibrazione SVI: `calibration/hierarchical_model_v2.py`,
  `calibration/sprint15_recalibrate.py`
- Sobol sensitivity: `calibration/sobol_sensitivity.py`
- EnKF: `core/orchestrator/enkf.py`
- FinancialTwin: `core/financial/twin.py`,
  `core/financial/country_params.py`
- Stress templates: `core/financial/stress_templates.py`
- CIR: `core/financial/rates.py`
- Stakeholder filtering: `briefing/relevance_score.py`,
  `briefing/realism_gate.py`, `briefing/reasoning_audit.py`
- Stakeholder graph: `stakeholder_graph/`
- Sector betas empirici: `scripts/recalibrate_sector_betas.py` →
  `shared/sector_betas_empirical.json`
- Correlation matrix: `scripts/compute_correlation_matrix.py` →
  `shared/correlation_matrix.json`
- Impulse response: `scripts/calibrate_impulse_response.py` →
  `shared/impulse_response_coefficients.json`
- Panic multiplier: `scripts/calibrate_panic_multiplier.py` →
  `shared/panic_multiplier_calibration.json`
- VAR contagion: `scripts/build_sector_contagion_var.py` →
  `shared/sector_contagion_var.json`
- BYOD: `core/byod/sanitizer.py`, `docs/BYOD_ARCHITECTURE.md`
- DORA: `core/dora/{schema,exporter,classification}.py`,
  `docs/DORA_EXPORT_SCOPE.md`
- Self-calibration: `core/calibration/continuous.py`,
  `scripts/continuous_calibration.py`
- Validation A/B: `scripts/validate_empirical_wiring.py` →
  `docs/EMPIRICAL_VALIDATION_REPORT.md`
- Backtest corpus: `backtest_scenarios.py`,
  `backtest_scenarios_global.py`
- Backtest engine: `backtest_financials.py`
- Paper: `paper/digitaltwin_calibration_v2.8.md` (1988 lines), PDF
  generato da `paper/digitaltwin_calibration_v2.8.tex`

## Appendice B — Documentazione collaterale

- `paper/digitaltwin_calibration_v2.8.md` — paper accademico completo
  (JASSS submission)
- `paper/COVER_LETTER_JASSS.md` — cover letter
- `paper/RESPONSE_TO_REVIEWERS.md` — response addendum v2.8
- `paper/REPRODUCIBILITY.md` — pinned-version reproducibility README
- `docs/RELAZIONE_BANCA_SELLA.md` — relazione tecnica e commerciale
  (v0.8, 580 righe)
- `docs/BYOD_ARCHITECTURE.md` — BYOD enclave architecture
- `docs/BYOD_DATA_FLOW_AUDIT.md` — audit dei 17 LLM call site
- `docs/DORA_EXPORT_SCOPE.md` — scope decision DORA MVP
- `docs/EMPIRICAL_VALIDATION_REPORT.md` — validation A/B
- `CITATION.cff` — citation metadata (Zenodo-ready)
- `docs/SPRINT_1-13_CHANGELOG.md` — changelog sprint 1-13

---

**Fine documento.** Per domande tecniche specifiche, consultare i file
elencati in Appendice A. Ogni numero qui presente è verificabile
empiricamente eseguendo gli script in `scripts/` su un clone del repo.
