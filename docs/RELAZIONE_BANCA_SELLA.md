# DigitalTwinSim — Piattaforma di simulazione multi-agente per il settore bancario

**Relazione tecnica e commerciale per Banca Sella**
*Maggio 2026 · v0.8*

> **Aggiornamento v0.8 (questa revisione)**: il motore di impatto finanziario è stato
> ricalibrato end-to-end su 95 eventi storici reali (corpus IT + global) con dati
> yfinance verificati. Le quattro costanti precedentemente euristiche (`panic_mult`,
> `recovery_factor`, `escalation_factor`, `t7 = t3 × 1.3`) sono state sostituite con
> coefficienti empirici per (intensity_bin × sector). I beta settoriali sono ora
> stimati per (country × sector) via OLS su 2.177 giorni di trading. Pair trades non
> più cablati: derivati dalla matrice di correlazione globale 180 ticker. Sezione 4.8
> ha i dettagli tecnici e i numeri di validation. Frontend non emette più stime
> "fake" lato client quando il backend è offline.

---

## 1. Executive summary

DigitalTwinSim è una piattaforma di **simulazione multi-agente con accoppiamento finanziario reale** progettata per stress-testare decisioni strategiche bancarie in scenari plausibili e auditable, prima che la decisione venga presa.

A differenza di un framework di stress test regolamentare (statico, retrospettivo) e a differenza di una ricerca di mercato (lenta, costosa, post-hoc), DigitalTwinSim genera **5-9 round narrativi** in 8-15 minuti che mostrano **come reagirebbero realmente** regulator, competitor, consumatori, media e mercato a una decisione strategica — con **delta KPI quantitativi** (NIM, CET1, LCR, deposit runoff, loan demand) coerenti con i vincoli ALM italiani 2025.

Ogni decisione è **tracciabile**: ogni esclusione/inclusione di stakeholder è documentata da un trace di reasoning step-by-step verbatim del modello, ogni numero finanziario è ricondotto a un parametro deterministico calibrato su benchmark EBA/ECB pubblici. **Compliance-ready** per pre-deliberation use case.

**Costo per simulazione completa**: $0.20–0.50 (LLM + compute). **Tempo**: 8–15 minuti. **Output**: report HTML stampabile in PDF, dashboard interattiva, JSON tracciabile.

---

## 2. Cosa è (e cosa non è)

### 2.1 È

- Un **gemello digitale narrativo + finanziario** di un settore di mercato (banking, insurance, asset management)
- Un sistema che modella **52 stakeholder reali** (politici, regulator, CEO peer, giornalisti, consumer voice, cluster cittadini) con posizioni dinamiche su un'asse semantica [-1, +1]
- Un motore di **opinion dynamics calibrato Bayesiana** (5 forze, mixture softmax, gauge-fixed) accoppiato a un **financial twin ALM** con vincoli regolatori reali
- Una **piattaforma what-if** per testare "Strategia A vs Strategia B" su un ventaglio di outcome quantificati

### 2.2 Non è

- **Non è un modello previsionale**: non predice la verità futura, esplora scenari plausibili. Identici parametri restituiscono ensemble di traiettorie con intervalli di confidenza al 90%.
- **Non sostituisce il modello ALM core di Banca Sella**: lo affianca come "what-if narrative layer". Il vostro IRRBB / capital model interno resta l'autorità in compliance.
- **Non è un'IA generativa pura**: è un sistema **ibrido fisico-narrativo** dove l'LLM gioca solo dove la narrativa lo richiede, mentre i KPI bancari sono governati da fisica deterministica (Hull-White-style, deposit beta, elasticità).

---

## 3. Use case bancari concreti

### 3.1 Pricing & repricing decisions
*"Cosa succede se a fronte del taglio BCE manteniamo lo spread anziché passare il taglio per intero?"*
→ DigitalTwinSim simula 5 mesi di reazione: Findomestic/Agos rispondono, Codacons si attiva, Bankitalia commenta, retail decide. KPI: deposit runoff cumulativo, NIM Δ, share of wallet.

### 3.2 Strategie di comunicazione in crisi
*"Come reagisce il mercato se diciamo X invece di Y nel comunicato stampa Q3?"*
→ Stress test reputazionale con Codacons/Altroconsumo, Corriere/Sole 24 Ore, citizen cluster.

### 3.3 Stress test regolatori "quotidiani"
*"Scenario EBA Adverse 2025: come si muove il nostro CET1 nei prossimi 6 mesi?"*
→ Template EBA precaricato (`adverse`, `baseline`) con CIR rates stocastico + opinion dynamics + financial twin.

### 3.4 M&A reaction modeling
*"Se annunciamo l'acquisizione di una fintech tedesca, come reagiscono BaFin, Bankitalia, peer?"*
→ Multi-country stakeholder graph (IT/DE/FR/ES/NL), ALM cross-jurisdiction.

### 3.5 Confronto A/B strategico
*"Strategia di pass-through completo vs loyalty program: quale ha miglior trade-off NIM/quota mercato?"*
→ Due simulazioni parallele, delta KPI affiancato, decisione data-driven.

### 3.6 Wargame tattico (modalità interattiva)
*Il decisore gioca in tempo reale: ad ogni round riceve la SITREP e può injectare una contromossa.*
→ Console wargame stile sala operativa, utile per board training.

---

## 4. Architettura tecnica

L'architettura è organizzata in **5 strati indipendenti** che girano in lockstep ad ogni round della simulazione.

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Briefing Layer                                              │
│    Brief LLM-driven scoping → scope strutturato                │
│    Layer 1 (relevance score) → Layer 2 (realism gate)          │
│    Layer 3 (CoT reasoning audit per casi marginali)            │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Multi-Agent Layer (52 agenti, 3 tier)                       │
│    Tier 1 — Elite (8-14): Sella exec, peer CEO, ABI, Bankitalia│
│    Tier 2 — Institutional (6-10): media, lobby, tribunali      │
│    Tier 3 — Citizen Clusters (5-8): retail, PMI, millennials   │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Opinion Engine (5-force softmax mixture, Bayesian-calibrated)│
│    Forze: direct LLM | herd | anchor | social | event_shock    │
│    Calibrazione: NumPyro SVI hierarchical + EnKF data assim    │
└────────────────────────────────────────────────────────────────┘
        ↓ in lockstep ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. FinancialTwin ALM (deterministic, country-aware)            │
│    Stato: NIM, CET1, LCR, deposit balance, loan demand,        │
│    duration gap, hedging P&L, breach detection                 │
│    Default IT/DE/FR/ES/NL/US/UK con dati live ECB/FRED/BoE     │
│    Coupling bidirezionale opinion ↔ financial                  │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Reporting & Audit Layer                                     │
│    Report HTML print-to-PDF · dashboard React · JSON tracciabile│
│    Reasoning trace per audit compliance                        │
└────────────────────────────────────────────────────────────────┘
```

### 4.1 Sistema di filtering stakeholder a 3 layer (oggi non esiste in nessun competitor noto)

Il problema "chi commenta credibilmente questa decisione?" è la **prima fonte di errore** in qualsiasi simulazione multi-agente. Senza un filtering rigoroso, il sistema produce roster con Donald Trump che commenta il pricing di Banca Sella — output non vendibile a un CFO.

DigitalTwinSim risolve con **3 layer in cascata**:

**Layer 1 — Score deterministico** (`briefing/relevance_score.py`)
- 6 componenti pesate: country_match, sector_match, semantic_similarity (via Gemini embeddings), brief_mention, archetype whitelist/denylist, global figure penalty
- Threshold 0.40 conservativo
- Penalty differenziata: foreign head-of-state (1.0) vs in-country head-of-state su brief non-politico (0.7)
- Latenza: <1ms per stakeholder, 0$ runtime
- 22 unit test su 5 brief diversi (banking IT, IT/DE cross-border, banking DE, banking US, US politics) per blindare la calibrazione

**Layer 2 — LLM realism gate** (`briefing/realism_gate.py`)
- Audit LLM (Gemini) STRICT in modalità REJECT-by-default
- Prompt con esempi nominati di global figure da rifiutare
- Cache verdict per (brief_hash, agent_id) — riutilizzo cross-sim
- Sempre attivo (synthesised default scope se Layer-0 fail)

**Layer 3 — Chain-of-Thought reasoning audit** (`briefing/reasoning_audit.py`)
- SOLO sui marginali in zona [0.30, 0.60] (~5-10 candidati per sim)
- Output strutturato: 4 step (topics_in_brief → agent_expertise → overlap_assessment → decision_rationale) + verdict + confidence + summary
- **Trace verbatim del LLM salvato in JSON tracciabile**
- Demo killer: il cliente vede *perché* il sistema ha deciso di includere Cottarelli ma non Draghi

### 4.2 Opinion Engine — fisica delle opinioni

Le posizioni di ogni agente evolvono nello spazio [-1, +1] tramite una **mixture softmax di 5 forze**, ognuna parametrizzata e calibrata Bayesiana:

| Forza | Significato fisico | Peso calibrato (esempio political domain) |
|---|---|---|
| `direct` | Risposta diretta dell'LLM al prompt | α_direct ≈ 0 (gauge-fix) |
| `herd` | Tendenza ad allinearsi al sentiment medio | α_herd = -0.176 |
| `anchor` | Rigidità verso posizione iniziale (rigidity-modulated) | α_anchor = +0.297 |
| `social` | Influenza ponderata dagli agenti che l'agente "segue" | α_social = -0.105 |
| `event_shock` | Reazione all'evento esogeno generato dall'event injector | α_event = -0.130 |

I coefficienti α sono **fittati Bayesiani con NumPyro SVI** (Stochastic Variational Inference) usando una architettura `AutoLowRankMultivariateNormal` su dati di calibrazione. Il sistema produce intervalli di confidenza al 90% gerarchici (3 livelli: global → domain → scenario), 5-13× più stretti rispetto a NUTS sui parametri deboli (α_social, α_event), in cambio di un bias residuo controllato.

Per la propagazione dell'incertezza in-sample è disponibile un **EnKF (Ensemble Kalman Filter)** con augmented state [θ, z] dove θ sono i pesi delle forze e z sono le posizioni — abilita data assimilation di osservazioni reali (sondaggi, dati di mercato) senza re-fit completo.

**Importante**: il `domain="financial"` ha attualmente RMSE 0.063 vs 0.012 del political — questo limite è dichiarato e gli sprint v3.0 stanno chiudendo il gap con coupling forte opinion↔financial.

### 4.3 FinancialTwin ALM — fisica del bilancio

Mentre l'opinion engine racconta la storia narrativa, il **FinancialTwin** mantiene in lockstep i KPI del bilancio bancario, con vincoli regolatori reali.

#### Parametri di default (fonte: pubblicazioni 2024-2025)

| Parametro | IT default | Fonte |
|---|---|---|
| Deposit β sight | 0.45 | ECB Economic Bulletin 2025 |
| Deposit β term | 0.75 | ECB Economic Bulletin 2025 |
| Consumer loan elasticity | -1.7 (median IT -2.14) | Bonaccorsi/Magri, micro-data IT |
| NIM baseline | 1.70% (EU Q2 2025: 1.58%) | EBA Risk Dashboard Q2 2025 |
| CET1 baseline | 16% | EBA Risk Dashboard Q3 2025 (EU 16.3%) |
| LCR baseline | 170% | EBA Risk Dashboard Q3 2025 (EU 160.7%) |
| Mortgage var/fix mix | 50/50 stock, 20/80 nuovi | Banca d'Italia 2025 |
| Deposit runoff cap | 4%/round (no panic) | letteratura empirica IT |

#### Country dispatch automatico (7 paesi)

| Paese | Deposit β sight/term | CET1 baseline | Note distintive |
|---|---|---|---|
| 🇮🇹 IT | 0.45 / 0.75 | 16% | Default canonico |
| 🇩🇪 DE | 0.30 / 0.65 | 18.9% | Sparkasse drag β |
| 🇫🇷 FR | 0.40 / 0.70 | 17% | Livret A regulation |
| 🇪🇸 ES | 0.55 / 0.80 | 13.9% | Post-2012, mortgage 70% var |
| 🇳🇱 NL | 0.50 / 0.75 | 19.4% | Mortgage 95% fixed |
| 🇺🇸 US | 0.40 / 0.85 | 14.3% | Mortgage securitised, Fed Funds |
| 🇬🇧 GB | 0.35 / 0.75 | 17.9% | Ring-fencing, BoE Bank Rate |

#### Coupling bidirezionale opinion ↔ financial

**Opinion → Financial** (weighted by stakeholder exposure):
- Negative depositor-weighted opinion → ↑ deposit runoff
- Negative borrower-weighted opinion → ↓ loan demand
- Competitor agents con posizione negativa verso Sella → ↑ market share grab

**Financial → Opinion** (FeedbackSignals):
- `nim_anxiety`, `cet1_alarm`, `runoff_panic`, `competitor_pressure`, `rate_pressure`
- Iniettati come context nei prompt agente del round successivo
- L'LLM legge "[Segnali stress finanziario] CET1 vicino al limite regolatorio (alarm 0.74)" e aggiorna la sua posizione

#### Modello di Contagio e Beta Empirici (vedi anche §4.7)

Per la modellazione dell'impatto sul mercato azionario e del pair trading, il sistema non utilizza correlazioni euristiche o qualitative, ma un approccio quantitativo rigoroso (calibrato end-to-end nelle sprint v0.8):

- **Sector Betas empirici geolocalizzati** (`shared/sector_betas_empirical.json`, 86 celle, 20 paesi): i beta non sono assunti "flat" a livello globale, ma fittati via OLS pooled su log returns 2018-2025 per ogni (country × sector). Esempio: IT banking β=1.22 (R²=0.72, 7 ticker), US tech β=1.26 (R²=0.85, 15 ticker), GB defense β=1.65, CN banking β=0.57 (defensivo SOE).

- **Matrice di correlazione globale** (`shared/correlation_matrix.json`, 180 ticker × 2.177 giorni): correlazioni di Pearson su daily log returns per derivare empiricamente le pair trade legs (sostituisce il dizionario `CRISIS_PAIR_TRADES` precedentemente cablato a mano). Esempio shock banche italiane → top correlati empirici: BPE.MI, BNP.PA, GLE.PA, MB.MI, DBK.DE.

- **Community Detection Louvain**: 14 community sul grafo delle correlazioni empiriche identifica cluster di mercato (banche italiane, US financials + media, US Big Tech + TSMC, Japan, energy majors, ecc.). Visualizzato come `GlobalContagionGraph` interattivo nel `/backtest`.

- **Cross-sector VAR(1) spillover network** (`shared/sector_contagion_var.json`): per ogni (sector_i, sector_j), regressione `r_j(t) = α + β·r_i(t-1)` su 2.177 giorni → 39 spillover edges direzionali significativi (|β|≥0.05, |t|≥2.0). Esempi economicamente interpretabili: tech→infrastructure +0.12 (t=5.7, capex propagation), tech→automotive +0.12 (semis/EV exposure), healthcare→media −0.14 (defensive rotation), real_estate→luxury +0.10 (wealth effect). UI: `SectorContagionGraph` nel `/backtest`.

#### Modello stocastico tassi (CIR 1-factor)

```
dr = κ(θ - r) dt + σ√r dW
```
Default Euribor 3M area euro 2024-2026: κ=0.30, θ=2.5%, σ=1.5%, r₀=2.4%.

Discretizzazione Euler-Maruyama, clamp positivo (Feller 2κθ > σ² rispettato).

#### Stress templates regolatori

- **EBA Baseline 2025**: ciclo normale, parametri standard
- **EBA Adverse 2025**: shock +250bps cumulativo, deposit β 0.55/0.85, elasticità -2.1, CET1 alarm 13%, breach detection LCR<100% / CET1<min

### 4.4 Live data ingestion

Connettori opzionali, **free public endpoints** (no enterprise tier $5K/mese):

- **ECB Statistical Data Warehouse**: Euribor 3M, ECB DFR, BTP/Bund 10Y yield → spread
- **FRED (St. Louis Fed)**: Effective Fed Funds Rate, US 10Y Treasury, UK 10Y Gilt
- **Bank of England Statistical Database**: BoE Bank Rate

Cache 24h TTL, fallback graceful su default literature-based se rete unreachable.
Auto-refresh al boot del FinancialTwin per ogni sim.

### 4.5 Stakeholder Graph

DB di **744 stakeholder reali** organizzati per paese (IT/DE/FR/ES/UK/US), categoria (politician, ceo, journalist, magistrate, central_bank, industry_association, consumer_advocacy, regulator…), tier (1-3), con campi:
- `country`, `archetype`, `party_or_org`, `role`, `bio`, `key_traits`
- `positions[]` (topic_tag, value ∈ [-1, +1], confidence)
- `relationships[]` (target_id, strength, context)
- `affiliated_tickers[]` per equity impact mapping

Per il dominio finance Italia abbiamo aggiunto un set domain-specific di 20 stakeholder verificati (Patuelli/ABI, Panetta/Bankitalia, Codacons, Findomestic, Agos, Compass, Carlo Messina/Intesa, Orcel/UniCredit, Cottarelli, De Bortoli, Fubini, Savona/Consob, Signorini/IVASS, MEF, Altroconsumo).

### 4.6 LLM agent engine

- **Provider**: Gemini Flash Lite (default), Claude haiku, OpenAI gpt-5.4-mini (configurabile via env)
- **Cost**: ~$0.16 per simulazione 5-round
- **Heterogeneity**: archetype-driven prompt templates (rational analyst, behavioral retail, populist activist…)
- **Memory**: ogni agente mantiene memoria delle proprie posizioni e citazioni passate, ricevuta come context al round successivo
- **Output strutturato**: response schema enforced (Gemini) per evitare prompt injection da viral text

### 4.7 Calibrazione empirica end-to-end del motore finanziario (NEW v0.8)

Quattro costanti precedentemente euristiche nel motore di impatto finanziario sono state sostituite con coefficienti **calibrati su event study reali** del corpus storico (95 scenari × ~4 ticker, 310 osservazioni event×ticker valide dopo noise floor 20bps su returns yfinance):

| Costante legacy | Valore euristico | Sostituzione empirica |
|---|---|---|
| `panic_mult = exp(cri × 1.5)` | 2.3x → 4.0x via cri | **median per CRI bin** (cri 0.4–0.7: 2.7x · 0.7–0.85: 5.5x · ≥0.85: 10.9x) |
| `recovery_factor = 0.5 + 0.1×i` | 0.5–0.7 (recovery) | **T+3/T+1 empirica** per (intensity_bin × sector), 23 celle |
| `escalation_factor = 1.0 + 0.1×(i−2)` | 1.0–1.6 (escalation) | come sopra |
| `t7 = t3 × 1.3 + ...` | 1.3 (persistenza) | **T+7/T+1 empirica** per (intensity_bin × sector) |
| `CRISIS_PAIR_TRADES[topic]` | dizionario cablato | **derivato da matrice correlazione globale** |

**Findings significativi:**

1. La V-shape recovery (legacy `0.5 + 0.1×i`) è **empiricamente sbagliata**: i mercati post-shock NON tornano del 50% in 3 giorni, continuano a trendare (T+3/T+1 mediano ≈ 1.0 anche a basse intensità).
2. I fat tails sono **veri** (non era ovvio): a CRI estremo (≥0.85) l'amplificazione mediana è 10.9x vs i 4.0x predetti da `exp(cri × 1.5)` legacy — sotto-predizione del 170%.
3. Le pair trade derivate dalla matrice di correlazione (es. IT banks → BNP.PA + GLE.PA + DBK.DE) sono economicamente più sensate dei pair handcoded.

**Validation A/B (PRE empirico vs POST heuristic-only) sul corpus completo:**

| Metric | PRE (empirico) | POST (heuristic) | Δ |
|---|---:|---:|---:|
| MAE T+1 | 1.92pp | 1.96pp | -0.05pp |
| MAE T+3 | 2.87pp | 3.36pp | **-0.50pp** ✓ |
| MAE T+7 | 8.08pp | 5.25pp | **+2.83pp** ✗ |
| Direction accuracy | **58%** | 54% | **+3.8pp** ✓ |

Empirico vince su direzione (+4pp, la metrica che conta di più per i decisori) e su T+3. Perde su T+7 in regime slow-burn-escalation: la mediana T+7/T+1 ≈ 1.3 cattura bene gli eventi flash che mean-revertono ma sotto-predice gli eventi che si accumulano (EU Standoff Oct 2018, Renzi Dec 2016, Ukraine Feb 2022). Honest follow-up: fit bimodale T+7 condizionato su event_speed (wave==1 vs wave≥2). Il caveat è documentato in `docs/EMPIRICAL_VALIDATION_REPORT.md`.

**Diagnostic OLS sulla formula di intensità.** Una regressione `log|t1_real| = α + β·log(eng) + β·log(cri) + ... + δ_wave` (n=310, R²=0.193) ha rivelato segni invertiti su `wave` e `neg_inst_pct` — non bug del fit ma *metric definition mismatch* del corpus: `wave` cattura "velocità escalation" non "magnitudine T+1" (wave-3 = crisi politiche lente come EU Standoff, wave-1 = flash event come Saudi-Russia oil −30%/giorno). Il file `shared/intensity_formula_coefficients.json` è marcato `_status: "diagnostic-only"` e NON wired in produzione: applicarlo invertirebbe la logica del simulatore. Documentato come finding scientifico, non come fix.

**Scripts riproducibili** (tutti runnable da clean checkout):

```
scripts/compute_correlation_matrix.py      # 180×180 Pearson + Louvain
scripts/recalibrate_sector_betas.py        # OLS β per (country×sector)
scripts/build_global_corpus.py             # +35 eventi non-IT, hit-rate 70%
scripts/calibrate_impulse_response.py      # T+3/T+1 e T+7/T+1 per bin
scripts/calibrate_panic_multiplier.py      # median fat-tail per CRI bin
scripts/calibrate_intensity_formula.py     # OLS diagnostic (non wired)
scripts/build_sector_contagion_var.py      # VAR(1) spillover per sector pair
scripts/validate_empirical_wiring.py       # PRE/POST validation harness
```

**Frontend hardening (parallelo).** `frontend/lib/generate-financial-impact.ts` (precedentemente fallback con SECTOR_BETAS hardcoded e `provenance: "client-fallback"`) **è stato disattivato**. Quando il backend non emette dati finanziari per uno scenario, il `FinancialImpactPanel` mostra esplicitamente "Backend financial scorer did not emit data — we deliberately do not fabricate numbers client-side" invece di servire silenziosamente stime inventate. Niente più rischio di numeri "fake" se Railway è offline.

### 4.8 Stack tecnologico

| Layer | Tecnologia |
|---|---|
| Backend | Python 3.11, FastAPI, asyncio, uvicorn |
| LLM | Gemini 2.0 Flash Lite (default) + Claude/OpenAI alternatives |
| Calibrazione Bayesian | JAX, NumPyro, NumPy |
| Data assimilation | Custom EnKF (JAX) |
| Embeddings | Gemini text-embedding-004 (768-dim, multilingual) |
| Persistenza | PostgreSQL (Railway), volume disco per checkpoint |
| Frontend | Next.js 14 (App Router), TypeScript, D3.js, Recharts |
| Deploy | Railway (backend, single worker, PostgreSQL) + Vercel (frontend) |
| Live data | urllib stdlib, cache 24h JSON su volume |
| Test | pytest, 35+ test su core financial + relevance score |

---

## 5. Output deliverables

### 5.1 Report HTML stampabile in PDF

Auto-generato a fine simulazione, contiene:
- **Cover page**: scenario, dominio, round, costo LLM, data
- **KPI dashboard**: polarizzazione finale, post totali, reazioni, custom metrics scenario-specific
- **Grafici inline SVG**: trend polarizzazione, sentiment per round (no dipendenze JS, stampa pulita)
- **Sintesi narrativa LLM**: executive summary in italiano impeccabile (headers tradotti)
- **Sezione Stato ALM** *(banking domain only)*: tabella per round con NIM, CET1, LCR, depositi, loan demand, runoff, breach flags, policy rate
- **Cronaca per round**: top-5 post virali per round con autore, piattaforma, engagement
- **Panel agenti élite**: posizione finale ed evoluzione

CSS print-friendly (`@page A4`, `page-break-inside: avoid`). Bottone "Stampa / Salva PDF" in alto a destra. Zero dipendenze server (no weasyprint, no playwright).

### 5.2 Dashboard interattiva web

- Live replay command-center con timeline scrubable
- Network graph D3 stakeholder con animazioni di posizione
- Indicatori real-time durante esecuzione (SSE streaming + REST polling fallback)
- Confronto A/B di scenari (finance impact ticker overlay)

### 5.3 JSON tracciabile (compliance-ready)

- Reasoning trace per ogni decisione marginale di filtering
- Realism gate verdict con rationale per ogni stakeholder
- Round results con financial twin state + feedback signals
- Calibration parameters source + version + 90% credible intervals

### 5.4 Wargame mode (interactive)

Console fullscreen stile sala operativa: ad ogni round complete il sistema pausa, mostra SITREP (3-5 threats identificate), il decisore può injectare una contromossa testuale che diventa l'evento del round successivo. Round completati persistono in Postgres → resume-from-checkpoint disponibile.

---

## 6. Vantaggi competitivi

### 6.1 vs framework di stress test regolamentare (Moody's GCorr, S&P scenario engines)

| | Stress test classici | DigitalTwinSim |
|---|---|---|
| Asse temporale | Statico (1 quadrimestre, scenario fisso) | Dinamico (5-9 round, narrative emergente) |
| Stakeholder modeling | Aggregato (settore) | Granulare (52 agenti nominati) |
| Reazione narrativa | Assente | LLM-driven con audit trace |
| Costo per scenario | $50K-200K consulenza | $0.20-0.50 + analyst time |
| Time to insight | 4-8 settimane | 8-15 minuti |

### 6.2 vs ricerca di mercato tradizionale

| | Survey research | DigitalTwinSim |
|---|---|---|
| Granularità | Cluster aggregati | Per-agente nominato |
| Variabili manipulabili | 2-3 (cost-prohibitive) | Illimitate (free what-if) |
| Reattività | Stima statica | Dinamica round-per-round |
| Scenari catastrofici | Non rilevabili (nessuno risponde "panico") | Modellabili con stress templates |
| Costo per scenario | €30K-150K | $0.50 |

### 6.3 vs altre piattaforme AI di simulazione

- **Adobe Walkme journey twin**: focus su customer journey, no financial coupling, no public-figure stakeholder
- **Palantir Foundry**: integration platform, non specifico simulation, richiede team data engineering interno
- **Replit/Vercel agent SDK**: framework generici, no domain modeling banking, no calibrazione Bayesiana

DigitalTwinSim è l'**unico sistema** che combina:
1. Roster di public-figure agents reali
2. Coupling fisica-narrativa con vincoli ALM
3. Audit trace LLM per ogni decisione di inclusione
4. Calibrazione Bayesiana con confidence intervals
5. Live data ingestion da fonti regolatorie

---

## 7. Vantaggi concreti per Banca Sella

### 7.1 Velocità decisionale
Una decisione di pricing che oggi richiede settimane di consulenza si valida in 15 minuti con 3 scenari paralleli.

### 7.2 Qualità del dato
Numeri ALM stanno **per costruzione** entro vincoli regolatori reali (deposit β IT, NIM EBA range). Niente più "deposit runoff 92% su +2% rate hike" da output LLM raw — quello è uno scenario di bank run, non di repricing.

### 7.3 Auditability compliance
Ogni esclusione di stakeholder ha un trace di reasoning step-by-step. Ogni numero finanziario è ricondotto a un parametro deterministico con fonte. La compliance review può ricostruire il ragionamento in 5 minuti.

### 7.4 Estensibilità
Stakeholder graph espandibile (Sella può aggiungere il proprio network: clienti corporate, distributori, partner fintech) senza modificare il core. Domain plugin pattern: aggiungere "asset_management", "private_banking", "merchant_acquiring" è 1 settimana di lavoro.

### 7.5 Cost economics
- Costo runtime per simulazione: $0.20-0.50
- Infrastruttura cloud: $20-100/mese (dimensione startup)
- Senza dipendenze enterprise (no Twitter/X $5K/mese, no Bloomberg terminal $24K/anno)

### 7.6 Confronto strategico data-driven
Lancia 2-3 sim parallele con strategie diverse → riceve delta KPI affiancato → decide con numeri non con opinioni. Use case A vs B documentato nel report HTML, presentabile in board meeting.

---

## 8. Modalità di utilizzo

### 8.1 SaaS multi-tenant (entry point)
- Web UI: incolla brief → ricevi report
- API REST: integrazione con tool interni Sella
- Costo: pay-per-simulation, ~$0.50 + setup

### 8.2 Single-tenant cloud (raccomandato per banche)
- Istanza dedicata Railway/AWS, accesso solo Sella
- Stakeholder graph custom (network Sella + universo standard IT)
- Persistenza PostgreSQL gestita
- Compliance review-friendly (data residency EU)
- Costo: ~$200-500/mese infrastructure + simulation costs

### 8.3 On-premise / private cloud (banche conservative)
- Docker container deployable su VMware/Kubernetes interni
- LLM provider configurabile (Gemini cloud / Claude / Azure OpenAI / on-prem Llama)
- Zero dato esce dal perimetro Sella
- Costo: setup engineering + ~$10K licenza annuale

---

## 9. Roadmap pubblica

### v0.7 (Aprile 2026, in produzione)
- ✅ FinancialTwin ALM con coupling debole+
- ✅ 7 paesi supportati (IT/DE/FR/ES/NL/US/GB)
- ✅ Live ECB/FRED/BoE ingestion
- ✅ EBA stress templates (baseline, adverse)
- ✅ Sistema 3-layer filtering con reasoning audit
- ✅ Semantic similarity via Gemini embeddings
- ✅ Report HTML stampabile + dashboard interattiva

### v0.8 (Maggio 2026, in produzione — questa revisione)
- ✅ Matrice correlazione globale 180 ticker × 2.177 giorni (Pearson + Louvain → 14 community)
- ✅ Sector beta empirici per (country × sector), 86 celle, 20 paesi (R² fino a 0.85 su US tech)
- ✅ Pair trades empirici da matrice di correlazione (sostituisce CRISIS_PAIR_TRADES cablato)
- ✅ Impulse response T+3/T+1 e T+7/T+1 calibrati empiricamente (23 celle, 310 osservazioni)
- ✅ Panic multiplier mediano per CRI bin (sostituisce `exp(cri × 1.5)`)
- ✅ Cross-sector VAR(1) spillover network (39 edges direzionali significativi)
- ✅ Corpus backtest esteso a 95 eventi (60 IT + 35 globali, hit-rate T+1 70%)
- ✅ Validation A/B PRE/POST documentata (`docs/EMPIRICAL_VALIDATION_REPORT.md`)
- ✅ Frontend: rimosso fallback "fake numbers" client-side
- ✅ UI `/backtest`: nuovi GlobalContagionGraph + SectorContagionGraph (D3 Canvas)

### v0.9 (4-6 settimane)
- Coupling forte: Δp_i = f(ΔP&L_individual_i) deterministico
- Hull-White 1-factor (richiede BTP zero curve)
- Reasoning trace HTML rendering
- Bimodal T+7 model (split per event_speed → fix del +2.83pp regression individuata in §4.7)
- Domain plugin insurance (Solvency II) e asset management

### v1.0 (3-4 mesi)
- Multi-instance scaling (Redis Streams per event queue)
- Real-time market data refresh durante sim (intra-day)
- Custom stakeholder graph editor per cliente
- Continuous self-calibration: pipeline daily news → shadow sim → 7d realised compare → EnKF state update (vedi §13)
- Export XBRL/DORA per compliance regolatoria (vedi §13)
- BYOD enclave: LLM riceve solo prompt narrativi, dati finanziari restano on-prem (vedi §13)

---

## 10. Limiti dichiarati (compliance honesty box)

1. **Non è un modello previsionale**: gli scenari sono plausibili, non profezie. Da usare come supporto a decisione, non sostituto del modello regolamentare interno.

2. **Calibrazione finance ha RMSE 0.063** vs 0.012 del political domain. Stiamo chiudendo il gap con coupling forte ma il sistema oggi performa meglio su narrative reazionali che su precision financial forecast.

3. **L'LLM è non-differenziabile**: il sistema NON è end-to-end differentiable nel forward pass. La calibrazione Bayesiana usa un shadow simulator differenziabile per JAX gradient flow.

4. **Stakeholder DB è curato manualmente**: 744 stakeholder reali ma con coverage variabile per dominio. Richiede manutenzione (~5h/mese) per restare aggiornato su uscite/cambi di carica.

5. **Live data dipende da endpoint pubblici**: ECB SDW e BoE possono avere downtime; il sistema cade graceful su default ma in quei casi i numeri non sono "live".

6. **Compliance regolatoria**: il sistema non ha (ancora) certificazione ESMA/Bankitalia. Per stress test regulatorie ufficiali resta lo strumento certificato interno. DigitalTwinSim è strumento di **pre-deliberation**, non di **regulatory submission**.

---

## 11. Onestà tecnica vs marketing competitor

Molti vendor in questo spazio promettono "world's first true Hybrid Physics + Narrative Digital Twin" e "end-to-end differentiable revolutionary system". DigitalTwinSim *è* hybrid physics + narrative, ma:

- **non lo chiamiamo "rivoluzionario"** (le banche non comprano rivoluzioni, comprano "auditable + back-tested + compliance-ready")
- **non promettiamo "end-to-end differentiable"** finché c'è un LLM nel loop (matematicamente impossibile)
- **non usiamo "AI magica" per le correlazioni di mercato**, ma applichiamo algoritmi di Louvain su matrici di Pearson derivate da 6 anni di log returns reali.
- **non promettiamo "live X/Twitter feed"** perché Twitter API enterprise costa $5K/mese e la maggior parte dei use case non lo richiede
- **dichiariamo limiti e RMSE**
- **dichiariamo le regressioni della validation**: la sostituzione delle costanti euristiche con coefficienti empirici (v0.8) **migliora** direction accuracy (+3.8pp) e MAE T+3 (-0.50pp) ma **peggiora** MAE T+7 di +2.83pp su eventi slow-burn — documentato in `docs/EMPIRICAL_VALIDATION_REPORT.md`. Avremmo potuto nascondere il dato; lo abbiamo invece pubblicato con la causa identificata (mediana T+7/T+1 ≈ 1.3 perde la struttura bimodale flash-vs-slow), il fix in roadmap (sprint v0.9), e il report scientifico onesto come parte della consegna.
- **dichiariamo i diagnostic falliti**: l'OLS sulla formula di intensità (Sprint 81) ha rivelato segni invertiti su `wave` e `neg_inst_pct`, dovuti a metric definition mismatch del corpus. Il file `intensity_formula_coefficients.json` è marcato `_status: "diagnostic-only"` e NON è in produzione. La scelta di pubblicarlo come finding scientifico documentato anziché silenziarlo è parte dell'audit-trail del processo.

Questa onestà tecnica è una scelta di posizionamento: vogliamo che il vostro risk officer e il vostro CIO firmino la deliberazione di acquisto senza controllare se il marketing claim regge a un peer review. Tutto quello che diciamo è verificabile su Git (`git log`, `docs/EMPIRICAL_VALIDATION_REPORT.md`, `shared/*.json`).

---

## 12. Prossimi passi proposti per Banca Sella

### Pilota di 4 settimane (€0 license, costo solo runtime)
1. **Settimana 1**: setup istanza dedicata + onboarding 2-3 use case Sella (es. mortgage pricing, comunicazione crisi reputazionale, M&A reaction)
2. **Settimana 2**: integration di 30-50 stakeholder Sella-specifici (network corporate, partner Hype, distributori)
3. **Settimana 3**: 5-10 simulazioni parallele su decisioni reali in pipeline, confronto output vs intuizione management
4. **Settimana 4**: review compliance, audit trace, sign-off per produzione

### Outcome atteso
- 1 decisione strategica validata con DigitalTwinSim entro fine pilota
- 1 report PDF presentabile a CDA con confronto A/B documentato
- Stack tecnico in single-tenant cloud Sella, owned & operated

### Investimento pilota
- Setup engineering: incluso
- Runtime: ~$50-100 per le simulazioni del pilota
- Licensing: zero per il pilota, da definire post-pilota in funzione di volume

---

## 13. Strategic moats (roadmap v1.0+ — sotto valutazione tecnica)

Tre asset strategici che trasformerebbero DigitalTwinSim da "tool" a "infrastruttura difendibile". Inclusi qui per trasparenza sulla direzione del prodotto, non come commitment di delivery a Sella.

### 13.1 Continuous self-calibration (Data Network Effect)

Pipeline notturna: ogni giorno alle 18:00 Reuters API → top-3 news → brief auto-generato → simulation ombra → previsione registrata. T+7 dopo, yfinance scarica i prezzi reali, computa l'errore vs previsione, aggiorna lo stato dell'EnKF e i coefficienti delle matrici di correlazione. Il modello impara dai propri errori in tempo reale.

Effort: 2-3 settimane (le primitive ci sono già: `correlation_lookup`, EnKF in `core/orchestrator/`, calibration scripts riproducibili). Restano da scrivere: news feed connector + scheduler + drift detector + safe rollback su regressione. Costo runtime: ~$15/mese (1 sim/giorno × $0.50 LLM × 30).

Moat: dopo 12 mesi di operatività, lo stato del filtro Bayesiano della piattaforma incorpora ~360 calibrazioni event-driven che un competitor che fork del codice oggi non recupera senza ri-running il loop. Tesla-Autopilot-style data moat.

### 13.2 Export DORA Major Incident Report — ✅ MVP IMPLEMENTATO (Maggio 2026, Sprint 88-90)

Modulo che prende l'output di una simulazione wargame di crisi operativa/cyber e genera **automaticamente il Major ICT-related Incident Report** richiesto da DORA (Regulation EU 2022/2554, Art. 19-20, in vigore dal 17 Gennaio 2025) nel formato XML prescritto da EBA/EIOPA/ESMA Joint Committee Final Report JC 2024-43 (Luglio 2024).

**Use case Sella, esempio concreto.** Il CRO simula un attacco DDoS al sistema di internet banking (`/wargame` in modalità "Cyber crisis"). Il sistema:
1. Esegue 5-9 round di simulazione con stakeholder reali (SOC, Banca d'Italia, BCE, stampa, clienti retail)
2. Tracciava metriche: clienti impattati, downtime ore, perdita economica stimata, polarization mediatica, viral posts
3. **Al termine, emette automaticamente** `outputs/<scenario>_dora_incident.xml` pronto per upload al portale regolatorio

**Stato implementazione (Sprint 88-90, Maggio 2026):**

- ✅ **Scope decision** documentato (`docs/DORA_EXPORT_SCOPE.md`): MVP target = Major Incident Report (Art. 19) — non COREP/FINREP XBRL (deferred a v1.0 perché richiede 4-6 settimane + domain expert ALM analyst).
- ✅ **Schema Pydantic** (`core/dora/schema.py`): `IncidentReport` + `ClassificationCriteria` + `FinancialEntity` + `AffectedFunction` + `MitigationAction` con i ~40 campi di Annex IV. Le 7 classification criteria mappano agli output del simulator.
- ✅ **XML exporter** (`core/dora/exporter.py`): zero-dependency (solo `xml.etree.ElementTree` stdlib), namespace `urn:eu:europa:dora:incident:report:1.0`, output pretty-printed e human-reviewable prima dell'upload.
- ✅ **Classification helper** (`core/dora/classification.py`): mapping deterministico ed esplicito da metriche simulator quantitative → 7 livelli qualitativi DORA. Audit-friendly: un CRO può vedere esattamente perché un incidente è classificato "high" su clients_affected (es. perché 420.000 clienti impattati cadono nel bucket [10k, 1M)).
- ✅ **15 unit test**: schema construction, classification (è "major" se >= high su qualsiasi axis OR downtime > 2h), XML rendering (well-formed, namespace, classification, mitigation, comms), final-report extras only when ReportType=FINAL, omission opzionali quando ImpactMetrics non ancora quantificate (caso INITIAL report 24h dopo).

**Esempio output (estratto dal golden test):**

```xml
<IncidentReport xmlns="urn:eu:europa:dora:incident:report:1.0"
                reportType="final" schemaVersion="1.0">
  <Header>
    <ReferenceNumber>SELLA-2026-INC-0042</ReferenceNumber>
    <SubmissionTimestamp>2026-05-04T10:00:00Z</SubmissionTimestamp>
    ...
  </Header>
  <Entity>
    <LegalName>Banca Sella Holding S.p.A.</LegalName>
    <LEI>815600B6E5DC0F5BF3D9</LEI>
    <CompetentAuthority>Banca d'Italia</CompetentAuthority>
    <Country>IT</Country>
  </Entity>
  <Classification>
    <ClientsAffected>high</ClientsAffected>
    <DataLosses>medium</DataLosses>
    <ReputationalImpact>high</ReputationalImpact>
    <DurationDowntimeHours>4.5</DurationDowntimeHours>
    <GeographicalSpread>national</GeographicalSpread>
    <EconomicImpactBand>1m-10m</EconomicImpactBand>
    <CriticalityOfServicesAffected>critical</CriticalityOfServicesAffected>
    <IsMajor>true</IsMajor>
  </Classification>
  ...
</IncidentReport>
```

**Sales motion**: il CRO smette di pagare $50K/trimestre a KPMG per compilare a mano i report DORA. Il bot del simulatore li produce in 30 secondi a fine wargame. Il "moat" è doppio:
1. Una volta certificato conforme contro la XSD ufficiale, i clienti restano per inertia regolatoria
2. Il modulo entra nel critical path del compliance team, non solo nel "nice-to-have" della Strategy team

**Effort residuo per certificazione enterprise:**
- ⏳ Validazione contro XSD ufficiale EBA (la XSD richiede download dal portale regolatore con login entità — 1-2 settimane post-pilota Sella)
- ⏳ Audit di conformità su 1-2 cicli reali con un compliance officer Sella (3-6 mesi calendar)
- ⏳ Estensione a COREP/FINREP XBRL templates (4-6 settimane separate, richiede domain expert ALM)
- ⏳ DORA ICT third-party register (Art. 28): aggiungibile in 1 settimana ma richiede un input table separato (contratti firmati con fornitori IT)

**Limiti dichiarati** (in `docs/DORA_EXPORT_SCOPE.md`):
1. XML well-formed e mappato a Annex IV ma NON ancora XSD-validato contro la specifica ufficiale (download da EBA reporting portal richiede registrazione entità).
2. MVP emette solo il "final report" template (più completo); initial e intermediate report sono subset derivabili impostando i campi later-availability a null (out of scope per MVP).
3. Mappatura metric→livello qualitativo è deterministica (`classify_from_simulation`) — un CRO può overridare manualmente prima del submit.

Moat: una volta certificato conforme, i clienti restano per inertia regolatoria — cambiare vendor su un report DORA significa rifare l'audit interno.

### 13.3 BYOD enclave (LLM data isolation) — ✅ IMPLEMENTATO (Maggio 2026, Sprint 84-87)

Architettura compartimenti stagni: il cliente inietta i suoi parametri segreti (LCR esatto, deposit mix, customer cohort breakdown) direttamente nel motore Python on-prem o in un container privato. L'LLM riceve **solo** prompt narrativi sanitizzati ("la stampa è arrabbiata, il regulator ha emesso un comunicato severo"); la matematica gira offline sui dati blindati del cliente. Zero dato finanziario sensibile esce dal perimetro.

**Stato implementazione:**

- ✅ **Audit dei call site LLM** completato (Sprint 84). 17 call site mappati, 9 critici (briefing pipeline) marcati per sanitizer hook. `docs/BYOD_DATA_FLOW_AUDIT.md` documenta cosa attraversa il boundary per ogni call site.

- ✅ **Sanitizer module** (`core/byod/sanitizer.py`, Sprint 85). Regex-based detection per 6 categorie: currency, financial_metric (con label threshold-aware: `LCR healthy` / `LCR breaching` / `CET1 tight`), client_id, IBAN, benchmark_value (Euribor / BTP-Bund), large_amount_in_context. 4 mode: OFF / LOG / STRICT / BLOCK. 20 unit test verificano: ogni categoria triggera, non-financial content preservato (poll %, polarization score, agent position), audit log JSONL well-formed, threshold labels corrette.

- ✅ **Pre-flight hook in `BaseLLMClient.generate()`** (Sprint 86). Sanitizer chiamato a ogni `generate()` con `BYOD_MODE` letto da env. Override per-tenant via `BYOD_TENANT`. Append-only audit log a `outputs/byod_audit.jsonl`.

- ✅ **Architettura defense-in-depth** (già esistente, ora documentata in `docs/BYOD_ARCHITECTURE.md`): il `FinancialTwin` (`core/financial/twin.py`) computa LCR/NIM/CET1/etc. localmente ed emette `FeedbackSignals` **categorici** (`nim_anxiety`, `cet1_alarm`, `runoff_panic`) — non numeri. Il sanitizer è il **secondo livello** di protezione, non il primo: cattura eventuali leak se in futuro una modifica al codice viola l'invariante architetturale.

**Effort residuo per certificazione enterprise:**
- ⏳ Security review formale + threat model (1-2 settimane)
- ⏳ Penetration test su una deployment STRICT (1 settimana)
- ⏳ Certificazione SOC2 Type II (3-6 mesi con auditor esterno, $25-50K)
- ⏳ Eventuale type-level enforcement (`SanitizedStr` newtype) — candidato v1.0

**Esempio funzionante.** Con `BYOD_MODE=STRICT`, una richiesta come:

> "The bank's LCR is 95%, deposit balance reached €12,500,000, with client-12345 issuing a complaint via IBAN IT60 X05428 11101 00..."

diventa, nel prompt che parte verso Gemini/OpenAI:

> "The bank's [LCR breaching: below 100%], deposit balance reached [large-amount], with [client-id] issuing a complaint via [IBAN]"

Il significato narrativo è preservato (l'LLM capisce ancora "LCR sta sforando"), ma l'esatto valore interno **non lascia mai il processo del cliente**.

**Compliance review query** (esempio):

```bash
jq 'select(.patterns | length > 0)' outputs/byod_audit.jsonl
```

Ogni riga con `patterns` non vuoto è prova auditabile che il sanitizer ha catturato un pattern sensibile. Se la query non ritorna righe per un tenant, il contratto BYOD è verificabilmente rispettato.

**Limiti dichiarati** (sezione "honesty box" in `docs/BYOD_ARCHITECTURE.md`):
1. Sanitizer regex-based, NON semantico — non cattura "twelve million euros" scritto a parole.
2. Detector `large_amount_in_context` richiede keyword finanziaria entro ±30 char dal numero.
3. Audit log writes best-effort (silent on IO error) — produzione richiede volume persistente.

Moat: uccide l'obiezione "Privacy/Security" al primo meeting con il CISO. È il prerequisito per ogni vendita a una banca tier-1 che oggi rifiuta ogni soluzione SaaS che tocchi dati clienti. **Adesso DigitalTwinSim può presentarsi al CISO con un'architettura documentata + test automated + audit log dimostrabile.**

**Effort combinato dei tre moat: 8-12 settimane di sviluppo + 3-6 mesi di certificazione/audit per il #2 e #3.** Effort molto sostenibile per un team di 2-3 ingegneri. La domanda strategica non è "è fattibile" (sì), è "in che ordine" (suggerimento: #3 prima, perché sblocca il funnel enterprise immediatamente; #2 secondo, perché monetizza il funnel sbloccato; #1 terzo, perché diventa moat dopo aver visto i clienti reali).

---

## Appendice A — Riferimenti scientifici

- Bonaccorsi di Patti, E. & Magri, S. *Consumer credit: evidence from Italian micro data*, Banca d'Italia working paper
- ECB Economic Bulletin Issues 2-8 / 2025, *Monetary policy transmission*
- EBA Risk Dashboard Q1-Q3 2025, *Banking sector stability indicators*
- EBA 2025 EU-Wide Stress Test methodology
- Anderson, J. L. (2001), *An Ensemble Adjustment Kalman Filter for Data Assimilation*
- Whitaker, J. S. & Hamill, T. M. (2002), *Ensemble Data Assimilation Without Perturbed Observations*
- NumPyro (Phan et al. 2019), *Deep Universal Probabilistic Programming with NumPyro*
- Cox, Ingersoll & Ross (1985), *A Theory of the Term Structure of Interest Rates*

## Appendice B — Componenti open-source utilizzate

Pydantic, FastAPI, asyncpg, urllib (stdlib), JAX, NumPyro, sentence-transformers (opzionale), Next.js, React, D3.js, Recharts.

## Appendice C — Contatti

[da personalizzare per Sella]

---

**Note di redazione**: questa relazione è il documento tecnico-commerciale standard per presentazione enterprise. Per uso esecutivo (board), ridurre a executive summary di 2 pagine + appendix tecnica. Per uso compliance (risk officer), espandere appendix B con security review (SOC2, GDPR data flow, LLM data residency).
