# DigitalTwinSim — Relazione per Accenture

**Versione:** 1.0 · maggio 2026
**Audience:** Accenture Italia, in particolare practice CMT (Communications, Media & Technology) e Strategy & Consulting
**Scopo:** Inquadrare il prodotto, esibire la complessità tecnologica sottostante, posizionarlo rispetto a Palantir e alle alternative di mercato, indicare gli use case account-specifici (es. Open Fiber).

---

## 1 · Executive summary (un minuto di lettura)

DigitalTwinSim è una piattaforma di **simulazione controfattuale calibrata empiricamente** per decisioni strategiche ad alto stakes. Il sistema accetta un *brief* in linguaggio naturale (es. *"Il MEF rinvia il decreto sulla Rete Unica per tensioni MEF/MIMIT, KKR contesta clausole, BCE/DG Comp segnalano rilievi state-aid"*) e produce in 5–10 minuti:

- una simulazione **multi-agente** dove 30–80 stakeholder reali (CEO, ministri, analisti, sindacati, commissari UE, giornalisti) interagiscono per N round su 4 piattaforme social/regulatory, con drift di posizione bayesiano e formazione di coalizioni;
- una proiezione **finanziaria deterministica** per ogni ticker menzionato nel brief (TIT.MI, ENEL.MI, BTP-Bund spread, …) con prezzo reale `t0` da yfinance e impulso di sector-beta per round;
- una **stima di confidence** (Monte Carlo 20–100 run con perturbazione dalla posterior NumPyro);
- un **report XML DORA-compliant** se lo scenario qualifica come Major Incident.

Il sistema non si limita a generare il singolo scenario: ogni notte un *continuous calibration loop* fetcha headlines reali, fa shadow-forecast su 205 ticker, e a T+1/T+3/T+7 confronta le predizioni con i ritorni yfinance realizzati. Questo crea un **moat dei dati**: dopo 12 mesi di operatività la cronologia forecast-vs-realised è di fatto irriproducibile da un competitor che parta oggi.

**Tagline:** *"What AIP is to operational decisions, DigitalTwinSim is to anticipative strategic decisions: same agents, same ontology, but projected onto futures that haven't happened yet, calibrated against real market data."*

---

## 2 · Cosa fa, in concreto

### 2.1 · Input
Un brief testuale (200–2000 caratteri). Esempi reali in produzione:

| # | Brief (estratto) | Domain | Round | Costo LLM tipico |
|---|---|---|---|---|
| 1 | *"L'Italia introduce tassa 5% su transazioni crypto…"* | financial | 9 | $0.4–$0.7 |
| 2 | *"Nike pubblica campagna generata interamente con AI…"* | marketing | 7 | $0.3–$0.5 |
| 3 | *"Il MEF rinvia decreto Rete Unica… KKR contesta clausole…"* | financial | 9 | $0.5–$0.9 |
| 4 | *"L'OMS raccomanda nuovo vaccino obbligatorio…"* | public_health | 9 | $0.4–$0.6 |

### 2.2 · Output
- **Replay temporale** round-by-round: post sociali generati da ogni agente, reaction, hashtag virali, evoluzione coalizioni, indicatore di polarizzazione 0–10.
- **Scenario report** in HTML + Markdown: timeline narrativa, top viral posts, sentiment evolution, agent position strip, network graph delle coalizioni emerse.
- **Financial impact panel**: prezzi ticker reali per round, BTP-Bund spread, sector-beta moves, contagion graph cross-sector.
- **Monte Carlo envelope**: confidence band 95% sulla traiettoria di posizione + outcome probability distribution + cluster di scenari distinti (kmeans su shape della traiettoria).
- **DORA Major Incident Report XML** (per scenari finanziari che superano la soglia RTS Annex I).

---

## 3 · Architettura — i 5 layer

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 5  Operator UX                                            │
│  /sim live · /scenario report · /wargame · /backtest · /admin   │
└──────────────────────────────────────────────────────────────────┘
        ▲                                                ▲
┌──────────────────────────────────────────────────────────────────┐
│  Layer 4  Compliance enclave                                     │
│  BYOD sanitizer (4 modi) · DORA XML exporter · audit log SQLite │
└──────────────────────────────────────────────────────────────────┘
        ▲                                                ▲
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3  Calibration & validation                               │
│  Continuous loop · NumPyro hierarchical · backtest MAE per dom.  │
└──────────────────────────────────────────────────────────────────┘
        ▲                                                ▲
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2  Twin engines (deterministici, calibrati)               │
│  Opinion dynamics · Financial ALM · Ticker prices · Contagion   │
└──────────────────────────────────────────────────────────────────┘
        ▲                                                ▲
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1  Multi-agent LLM core                                   │
│  EliteAgent · InstitutionalAgent · CitizenSwarm · 4 channels    │
└──────────────────────────────────────────────────────────────────┘
        ▲
┌──────────────────────────────────────────────────────────────────┐
│  Layer 0  Knowledge & data substrate                             │
│  745 stakeholders · 193 tickers · 86 sector-β cells · 39 VAR(1) │
└──────────────────────────────────────────────────────────────────┘
```

Ogni layer è autonomo, testato in isolamento, e l'architettura permette di sostituire un layer (es. cambiare provider LLM, swappare il financial twin) senza riscrivere gli altri.

---

## 4 · Layer 0 — Knowledge & Data Substrate

### 4.1 · Stakeholder graph
- **745 stakeholder** distribuiti su 6 paesi (IT, FR, DE, ES, GB, US) + UE
- Categorie: politicians (governo + opposizione + parlamento + sindaci + EU MEPs), CEO industriali/finance/luxury/fashion, magistrati, accademici, sindacalisti, giornalisti, influencer digitali, diplomats, esperti settoriali
- **Schema per nodo**: `id, name, role, country, archetype, tier (1-3), influence (0-1), rigidity (0-1), bio, communication_style, key_traits, platform_primary/secondary, positions[topic_tag, value, source, confidence], relationships[target_id, type, strength], wikidata_qid`
- **Position history**: ogni stakeholder ha una serie temporale di posizioni `[-1, +1]` per topic, aggiornata via EMA dal nightly updater (vedi 6.2)
- **Relazioni**: 6 tipi (`ally, opponent, coalition, supervised_by, shareholder, regulated_by`) con weight `[0, 1]`, usate per propagazione opinione e formazione di coalizioni emergenti
- **Aliasing**: 24'186 alias mappati per 761 stakeholder → entity matching robusto su testi RSS noisy

### 4.2 · Stock universe
- **193 ticker** equity reali (US S&P 500 mega-cap + EU FTSE/CAC/DAX/IBEX large-cap + UK FTSE100 + Asia Nikkei/HSI + EM brasiliani/messicani) + **15 indici globali**
- Tagging GICS-coerente: 16 settori (banking, insurance, telecom, energy_fossil, energy_renewable, utilities, defense, automotive, tech, healthcare, luxury, media, real_estate, infrastructure, food_consumer, sovereign_debt)
- **86 celle di sector-beta** per (settore × geografia) con 4 metriche (`political_beta, spread_beta, crisis_alpha, volatility_multiplier`) — calibrate da OLS su eventi storici noti

### 4.3 · Empirical pieces
- **Cross-correlation matrix** 180×180 ticker da 2015–2025 yfinance (Phase A)
- **Sector betas** per 20 paesi × 16 settori (Phase B)
- **Impulse response coefficients** per 3 bin di intensità × 23 celle (Sprint 79–80)
- **Panic multipliers empirici** per 3 bin di CRI (Crisis Risk Index)
- **VAR(1) cross-sector contagion network** con 39 edges significativi

Tutto in `shared/` come JSON statico, version-controlled, riproducibile da `scripts/calibrate_*.py`.

---

## 5 · Layer 1 — Multi-Agent LLM Core

### 5.1 · Tre tipi di agente
| Tipo | Conteggio tipico | Cosa fa | LLM call/round |
|---|---|---|---|
| `EliteAgent` | 8–12 | Personaggi nominati (CEO, ministri, analisti). Postano contenuti originali, mossa di posizione vincolata `±0.15`/round, memoria riflessiva | 1 |
| `InstitutionalAgent` | 6–10 | Voci collettive (ARERA, AGCom, Confindustria, BCE, DG Comp). Comportamento più rigido, output formale (regulatory_filing) | 0.5 (batched) |
| `CitizenSwarm` | 5–8 cluster | Gruppi sintetici di cittadini. Ciascuno con posizione media + dispersione + popolazione. Drift bayesiano (anchor + social + event + herd weights) | 0 (deterministic) |

Totale: 15–25 LLM call/round + 9 round = **135–225 LLM call/scenario**. A `gemini-3.1-flash-lite-preview` (input $0.25/M, output $1.50/M) il costo è $0.40–$0.90 per scenario completo.

### 5.2 · Channel multi-piattaforma
4 canali per dominio finanziario (esempi): `trading_desk` (jargon, max 280 char), `analyst_report` (3000 char, citazioni multipli), `regulatory_filing` (formale, legale, 5000 char), `fintwit` (punchy, 280 char). Ogni archetipo ha mapping primario+secondario verso i canali.

### 5.3 · Reflective memory + position drift
Ogni elite agent mantiene una memoria episodica delle ultime N round (post fatti, reaction ricevute, viral posts altrui). Il prompt successivo include questa memoria + i top viral posts del round + le tendenze di sentiment, costringendo l'agente a evolvere coerentemente. Cap di drift `±0.15`/round per realismo (cambio repentino di posizione = jailbreak segnalato).

### 5.4 · Realism gate
LLM secondario controlla i post generati con un *Chain-of-Thought reasoning audit* (Sprint 13B): se un post viola il personaggio (es. Giorgetti che attacca Meloni con epiteti), viene rigettato e rigenerato. Strict-mode + always-on (Sprint 10).

---

## 6 · Layer 2 — Twin Engines (deterministici, calibrati)

### 6.1 · Opinion dynamics
Implementazione canonica del modello bounded-confidence + social influence + anchor drift, con 7 pesi calibrati via NumPyro SVI hierarchical fit (Sprint 38, paper v2.8):
```
position[i, t+1] = clamp(
    α_anchor · anchor[i] +
    α_social · Σ neighbours_pos · weight +
    α_event  · shock_t · direction_t +
    α_herd   · sign(majority - position[i]) +
    α_drift  · (original_position[i] - position[i]),
  [-1, +1])
```
Con 4 parametri (`anchor_weight, social_weight, event_weight, herd_weight`) hanno **CI95 dalla posterior** che il Monte Carlo usa direttamente per propagare incertezza.

### 6.2 · Financial twin (ALM + ticker pricing)
Due sotto-componenti, separabili:

**(a) ALM banca commerciale** (`core/financial/twin.py`):
- 9 round → 9 stati `FinancialState` con NIM, CET1, LCR, NSFR, deposit base, loan demand, runoff/round, policy rate
- Parametri country-aware: deposit β (sight + term) calibrato per IT/FR/DE/ES/UK/US/EM dai dati EBA Risk Dashboard 2025 + literature (Bonaccorsi/Magri per credito al consumo)
- Live anchor refresh: ECB SDW (Euribor 3M, BTP-Bund spread, ECB DFR), FRED per US, BoE per UK — 24h cache, fallback graceful
- **Gating intelligente**: si attiva solo se il brief contiene keyword bancarie (`banca, deposito, CET1, LCR, EBA, …` o ticker bancari come UCG.MI). Su uno scenario Open Fiber, l'ALM resta spento — niente "banca commerciale italiana media" su un dossier telco.

**(b) Real ticker pricing** (`core/orchestrator/ticker_prices.py`):
- Estrazione automatica dei ticker dal brief (regex strict + intersezione con stock universe — accetta `TIT.MI_Prezzo_Azione`, sintetizza solo `TIT.MI`)
- Anchor `t0` da yfinance (12h cache)
- Per ogni round: `current_price = anchor · (1 + Σ predict_returns(cri, intensity, sector_β, country))` — totalmente deterministico, nessuna allucinazione LLM
- Prezzi visualizzati in valuta nativa (€, £, $, ¥) con direzione e Δ% cumulativo

### 6.3 · Contagion + escalation
- **Contagion scorer**: combina post count, reaction count, repost rate, top engagement, attivazione sindacati/partiti/istituzioni → CRI ∈ `[0, 1]`
- **Escalation engine**: 5 wave, attiva agenti dinamicamente (es. wave 1 = solo CEO + governo; wave 4 = sindacati + opposizione + commissari UE)
- **Financial impact scorer**: input CRI + engagement + polarization velocity + neg_institutional_pct + neg_ceo_count → output sector-shock vector + market_volatility_warning

---

## 7 · Layer 3 — Calibration & Validation (il moat)

### 7.1 · Hierarchical Bayesian fit (NumPyro v2.3)
- Modello: 4 parametri opinion-dynamics globali (μ_global) + scostamenti per dominio (μ_d) + scostamenti per scenario (θ_s)
- Posterior salvata in `calibration/results/hierarchical_calibration/v2.3_pubop/posteriors_v2.json`
- **Loader esposto via `CalibratedParamLoader`** che restituisce sia point estimates che CI95 per ogni parametro
- Il Monte Carlo usa direttamente la CI95 per sample `Normal(mean, (hi-lo)/3.92)` invece di un ±X% arbitrario

### 7.2 · Continuous calibration loop
Architettura ispirata a Tesla Autopilot ("data flywheel"):
```
Notte 0:  fetch_news(205 tickers) → predict_returns(cri, intensity, β) → SQLite registry
Notte +1: evaluate(forecast_T-1) → fetch realized close → compute MAE → drift log
Notte +3: evaluate(forecast_T-3)
Notte +7: evaluate(forecast_T-7)
```
- Schema SQLite append-only: `(forecast_date, ticker, predicted_pct, realized_pct, abs_error_pp, evaluated_at, horizon_days)`
- **Cron GitHub Actions** ogni notte 04:00 UTC (giusto dopo close EU, prima pre-market US)
- Dashboard `/compliance` panel "Self-calibration" mostra n_forecasts cumulati, n_evaluations, MAE running T+1/T+3/T+7, direction accuracy, breakdown per ticker
- **Dopo 6 mesi di operatività**: ~37k forecast, ~37k evaluation, MAE per ticker statisticamente significativo. Un competitor che parta oggi ha bisogno di 6 mesi di calendario per replicarlo.

### 7.3 · Backtest corpus
- 30+ eventi storici reali (referendum costituzionale 2016, Brexit, downgrade S&P 2014, crisi MPS 2016, ecc.) + 30+ eventi non-IT (Sprint 74)
- Per ogni evento: brief sintetico → forward simulation → confronto con polling reale (per eventi politici) o yfinance (per eventi finanziari)
- **MAE pubblicato nel paper v2.8**: 3.2pp per scenari politici IT, 1.8pp per scenari finanziari IT

### 7.4 · Monte Carlo a 3 fonti di varianza misurate
Non un dial soggettivo, ma 3 incertezze indipendenti:
1. **Posterior CI95** sui pesi opinion dynamics → sample Gaussiano
2. **Shock magnitude** del main sim perturbato con σ_obs = 0.20 (calibrata da residui backtest)
3. **Initial pro_pct** sample da N(50, σ_init) dove σ_init è la cross-sectional std delle posizioni agenti round 1 × 14pp

Risultato: la confidence band riflette tre incertezze **moltiplicative**, ognuna giustificabile davanti a un risk manager.

---

## 8 · Layer 4 — Compliance Enclave

### 8.1 · BYOD enclave
Risponde alla preoccupazione #1 di ogni CISO/CRO bancario europeo: *"i dati dei nostri clienti finiscono al provider LLM?"* — Risposta: **no, mai, by design**.

- **4 modi operativi** configurabili via `BYOD_MODE` env: `OFF` (passthrough, solo single-tenant trust), `LOG` (detect + audit only), `STRICT` (redact + audit, default produzione), `BLOCK` (raise BYODLeakError su qualsiasi pattern)
- **Pattern detector regex-based** su 8 categorie: IBAN, codice fiscale, partita IVA, client_id, email, telefono, conto corrente, importo €
- **Audit log SQLite** append-only: `ts, site, mode, raw_chars, sanitized_chars, patterns[category, count], tenant`
- **Sanitizer playground** UI per testare interattivamente prima di andare in produzione
- Hook diretto in `BaseLLMClient` — qualsiasi nuovo punto di chiamata LLM eredita la sanitizzazione automatica

### 8.2 · DORA Major Incident Report exporter
Risponde a EBA/EIOPA/ESMA JC 2024-43 (DORA Art. 19–20):
- Classificazione automatica per griglia 7-criteri Annex I (clients_affected, data_losses, reputational_impact, duration_downtime_hours, geographical_spread, economic_impact_eur_band, criticality_of_services_affected)
- Output XML conformity-ready, scaricabile da `/compliance/dora/export/{sim_id}`
- Funziona su qualsiasi simulazione completata: il sistema mappa metriche del twin → criteri DORA

### 8.3 · Auth + isolation
- HMAC-SHA256 cookie auth + API key headers (`Tenant` model)
- Per-utente scenario isolation via cookie sub + tenant header (Sprint 68)
- Invite link system con token HMAC-firmati (admin-mints, auto-signin, TTL configurabile)

---

## 9 · Layer 5 — Operator UX

| Route | Cosa fa |
|---|---|
| `/` | Dashboard: brief recenti, scenari attivi, costi mensili |
| `/new` | 4-step wizard (brief + KB + engine + review) |
| `/sim/[id]` | Live monitor: round-by-round con SSE streaming, ticker prices, polarization chart, Monte Carlo panel |
| `/scenario/[id]` | Report finale: viral showcase, network graph D3, ALM table (gated), report markdown |
| `/scenario/[id]/branches` | Scenario tree: what-if branches partendo da un round |
| `/wargame` | Modalità interattiva: pausa nei round critici, operatore inietta interventi (KB doc, regulatory action, public statement, market intervention, …) |
| `/backtest` | Hit-rate per dominio, calibration sidebar, residuals scatter |
| `/compliance` | CISO/CRO console: BYOD status + audit + sanitizer playground · DORA export · self-calibration history |
| `/admin/jobs` | Trigger manuale di calibration-forecast / evaluate / stakeholder-update |
| `/admin/invites` | Generazione invite link |
| `/paper` | Paper v2.8 leggibile in-app (calibration MAE, ablations) |

---

## 10 · Stack tecnologico

### Backend
- **Python 3.11**, FastAPI, asyncio, SSE (sse-starlette)
- **LLM**: Google Gemini `gemini-3.1-flash-lite-preview` via `google-genai`. Astrazione `BaseLLMClient` permette swap a OpenAI/Anthropic in <50 LOC
- **Concorrenza**: semaforo a 5 LLM concorrenti, rate-limit 0.5s, atomic budget reservation con soft-cap, retry con jittered backoff
- **Persistenza**: SQLite (WAL mode, batched commits), Postgres opzionale via `asyncpg`, Redis opzionale per cache distribuita
- **Numerico**: numpy, scikit-learn, scipy, NumPyro (per Bayesian fit), yfinance (live market data)
- **Osservabilità**: structlog, sentry-sdk, prometheus-fastapi-instrumentator
- **Test**: pytest, 100+ unit tests, integration test E2E, concurrency stress test (50 concorrent writers, 200 budget contention scenarios)

### Frontend
- **Next.js 14** App Router, TypeScript, Tailwind v3 (Quiet Intelligence design system)
- **Visualizzazioni**: D3.js (force-directed network, sankey coalitions, replay timeline), Recharts (line/area/bar charts), Framer Motion (transitions)
- **Streaming**: SSE consumer con bounded queue + drop-oldest backpressure
- **Auth**: middleware Edge runtime, HMAC verify, redirect-with-next pattern

### DevOps
- **Railway** per backend (Docker, persistent volume su `/app/outputs`)
- **Vercel** per frontend (Edge middleware + serverless functions per /api/auth)
- **GitHub Actions** per CI (pytest + tsc + nightly admin jobs)

---

## 11 · Posizionamento competitivo

### Mapping ai prodotti Palantir

| Palantir | Cosa fa | Vicinanza a DTS | Differenza chiave |
|---|---|---|---|
| **AIP** (Artificial Intelligence Platform) | Agenti LLM su ontologia per automazione decisioni operative | ★★★★ | Stessa architettura agenti+ontologia. AIP automatizza ops reali, DTS simula controfattuali futuri |
| **Gotham** | Intelligence fusion per analisti governo/difesa | ★★★ | Stesso target (analisti high-stakes), input opposto (intel reale del presente vs agenti sintetici del futuro condizionale) |
| **Foundry** + **Ontology** | Data platform + ontologia per enterprise | ★★★ | Sotto-tecnologia comparabile (stakeholder graph = ontologia), ma DTS è applicazione, non piattaforma |
| **Maven Smart System** | Battlefield AI per DoD | ★★ | Domain-specific (defense), real-time data fusion. Diverso dominio |
| **Apollo** | Deployment + monitoring infrastructure | ★ | Non comparabile |
| **ShipOS / Warp Speed** | Manufacturing OS verticali | ★ | Verticale industriale, DTS è cross-vertical |

**Slot di mercato che Palantir non riempie**: counterfactual scenario simulation con calibrazione empirica continua su dati pubblici. Palantir vende dati-presenti, DTS vende futures-condizionali calibrati.

### Confronto con altri player

| Soluzione | Tipo | Sovrapposizione | Cosa fa che noi non facciamo | Cosa NON fa che noi facciamo |
|---|---|---|---|---|
| **McKinsey QuantumBlack** | Consulting analytics | Bassa | Custom ML pipelines per cliente | Multi-agent LLM, real-time sim, self-calibration loop |
| **BCG X / X-Reactor** | Consulting + GenAI labs | Bassa | Bespoke GenAI products, change mgmt | Ontologia stakeholder pre-popolata IT/EU |
| **Accenture GenAI Studio** | Internal Accenture toolkit | Media | Integrazione SAP/Oracle, change track | Stress-test scenari, calibration empirica, DORA exporter |
| **Aera Technology** (Cognitive OS) | Decision intelligence SaaS | Media | Workflow automation supply chain | Multi-agent LLM, Monte Carlo bayesiano |
| **Beyond Limits** (Cognitive AI) | Symbolic+ML hybrid | Media | Explainability su domini ingegneristici | Sim sociale + finanziaria integrata |
| **DataRobot / Domino** | MLOps platform | Bassa | Train/deploy modelli ML classici | Sim agentica, ontologia, no-train workflow |
| **Salesforce Einstein** | CRM AI | Molto bassa | CRM-bound automation | Tutto il resto |

### Positioning one-liner per ciascun confronto

- **vs Palantir AIP**: "AIP per il futuro condizionale invece che per l'operatività presente"
- **vs Gotham**: "Gotham con agenti sintetici invece che intel reale → permette analisi pre-evento"
- **vs McKinsey QuantumBlack**: "Stessa rigorosità statistica, ma prodotto SaaS riusabile, non engagement custom"
- **vs Aera**: "Loro ottimizzano supply chain ricorrente, noi stress-testiamo decisioni one-shot ad alto stakes"
- **vs Accenture GenAI Studio**: "Complementare — DTS può essere integrato come pillar 'foresight' nello Studio"

---

## 12 · Use case per Accenture CMT

### 12.1 · Open Fiber / TIM-NetCo (account già pianificato)
**Scenario**: rinvio decreto Rete Unica con tensioni MEF/MIMIT e contestazione KKR.

**Cosa DTS produce in 5 minuti**:
- 30 stakeholder reali simulati (Gola, Scannapieco, Labriola, Giorgetti, Urso, Ribera, Virkkunen, analisti Mediobanca/Equita, CGIL/UILcom, Quintarelli, Aresu, Bloomberg Italy, …)
- 9 round di interazione con post realistici per piattaforma (analyst report, fintwit, X, regulatory_filing, parliament)
- Prezzi reali per round su TIT.MI (€0.65 anchor), DTE.DE, ORA.PA, VOD.L, TEF.MC, ENEL.MI, A2A.MI, TRN.MI
- BTP-Bund spread proiezione
- Monte Carlo 95% CI sull'outcome politico (rete unica si sblocca / si arena / KKR esce)
- Network graph delle coalizioni emerse (es. cluster "Pro-merger industriale" vs "Anti foreign-capital sovranista")

**Valore per Accenture**: poter dire al cliente "ecco le 3 traiettorie più probabili nelle prossime 8 settimane secondo il modello, con probabilità calibrate" invece di consegnare un PowerPoint statico.

### 12.2 · Altri scenari CMT-relevant
- **Tariffe roaming UE post-revisione 2026** → ARPU mobile EU + sector-beta telecom
- **DMA designation 2027** → impatto su Vodafone/Orange/Telefonica + gatekeeper sanctions
- **5G coverage obligation miss** → rischio fine ARERA/AGCom + DMA escalation
- **TIM Brasil divestiture** → contagion sovereign + retail telco LatAm
- **Sky / DAZN diritti Serie A 2027** → impact su Vivendi, Mediaset, Netflix

### 12.3 · Vertical-extension (oltre CMT)
- **Banking** (Banca Sella, Intesa, UCG): bank-run scenarios, ALM stress, DORA major incident drilling
- **Energy** (ENI, Snam, Terna): regulatory shocks ARERA, gas price shocks, political pressure rinnovabili
- **Defence** (Leonardo, Fincantieri): export compliance, ITAR fallout, EU strategic autonomy

---

## 13 · Cosa NON facciamo (onestà)

In una relazione tecnica ad Accenture conviene esibire i limiti:

1. **Non sostituiamo il giudizio del consulente**. DTS produce traiettorie probabilistiche, non raccomandazioni. La "next best action" resta umana.
2. **Non simuliamo dinamiche di micro-mercato** (order book, market microstructure). Il financial twin lavora su orizzonti T+1/T+3/T+7, non intraday.
3. **Non sostituiamo Gotham** se servono fonti intel classificate. DTS lavora su dati pubblici (RSS, Google News, yfinance, openparlamento, Wikidata).
4. **Calibration drift è un rischio noto**: se i meccanismi di mercato cambiano radicalmente (es. nuovo regime regolatorio), il modello backtest può degradare. Per questo c'è il drift log.
5. **LLM hallucination residual risk**: nonostante realism gate + reflective memory, in <2% dei post un agente può divergere dal personaggio. Tagged + flaggato nel report.
6. **Domain coverage incompleto**: 8 domini implementati (political, financial, corporate, marketing, public_health, commercial, energy, environmental). Settori esotici (es. agritech, fashion supply chain) richiedono nuovo plugin.

---

## 14 · Roadmap 2026 H2

| Sprint | Cosa | Effort | Valore Accenture |
|---|---|---|---|
| 18 | Multi-LLM orchestration (Claude + Gemini ibrido per scenario) | M | Robustezza vendor |
| 19 | Foreign language support (FR + DE + ES + EN) | M | EU-wide demos |
| 20 | Real-time event injection (push) | L | Wargame live durante crisis room |
| 21 | API tier B2B per integrazione in tool partner | L | Embed in Accenture GenAI Studio |
| 22 | Domain plugin: telecommunications (DAS) | S | Account CMT specifico |
| 23 | Bank-run simulator standalone (sub-product) | M | Vendita separata banking |

---

## 15 · Possibile partnership Accenture-DigitalTwinSim

Tre forme di collaborazione, in ordine di accoppiamento crescente:

### A. **OEM / white-label** (più leggero)
Accenture rivende DigitalTwinSim sotto proprio brand a clienti CMT/banking selezionati. DTS resta black-box, l'integrazione è via API + iframe.
- Time-to-market: 1 mese
- Effort Accenture: minimo (sales + delivery)

### B. **Integrazione tematica nello Studio GenAI**
DTS diventa il "Strategic Foresight pillar" di Accenture GenAI Studio. Single sign-on + ontologia condivisa con altri tool Accenture (Composer, Agent Hub).
- Time-to-market: 3 mesi
- Effort Accenture: medio (integration team)

### C. **Joint product / co-development**
Accenture co-investe sulla roadmap (es. plugin verticali, integrazione SAP/Oracle), DigitalTwinSim diventa Accenture-flavored. IP shared.
- Time-to-market: 6+ mesi
- Effort Accenture: alto (commitment di 2-3 FTE)

---

## 16 · Demo proposta

Per l'incontro con Accenture CMT consiglio di mostrare in 30 minuti:

1. **5 min** — Brief Open Fiber / Rete Unica letto a voce → click "Lancia"
2. **15 min** — Live monitor `/sim/[id]` con narrazione round-by-round mentre la sim gira: vediamo Gola che annuncia, Quintarelli che reagisce su X, Randone (Mediobanca) che pubblica nota, TIT.MI che scende in tempo reale, escalation Wave 3 quando CGIL annuncia sciopero, Ribera (UE) che invia lettera state-aid
3. **5 min** — `/scenario/[id]` report finale: network graph delle coalizioni emerse, viral showcase, prezzi finali ticker, Monte Carlo CI sui 3 cluster di outcome
4. **3 min** — `/compliance` mostriamo BYOD enclave + DORA exporter + self-calibration loop (anche per scenari non-DORA, è la prova del moat)
5. **2 min** — Q&A sulla roadmap + forme di partnership

---

## 17 · Contatti & next steps

- Repository: https://github.com/albertogerli/digital-twin-sim
- Live demo: https://digital-twin-sim.vercel.app
- API base: https://digital-twin-sim-production.up.railway.app
- Author: Alberto Gerli — alberto@albertogerli.it
- Documentazione tecnica completa: `docs/TECHNICAL_DOSSIER.md`, `docs/RELAZIONE_BANCA_SELLA.md` (template equivalente per Banca Sella v0.8)

---

> *"Three things make this defensible: the empirical calibration loop that compounds with operating time, the stakeholder graph that took 18 months to populate accurately, and the BYOD/DORA compliance scaffolding that EU regulators actually accept. None of them are reproducible in 90 days by a competitor with a checkbook."*
