# DigitalTwinSim — Proposta Tecnica per FormatLab

**Piattaforma di Simulazione Multi-Agente per l'Analisi Predittiva di Scenari Complessi**

*Documento tecnico-commerciale — Marzo 2026*

---

## 1. Executive Summary

DigitalTwinSim è una piattaforma di simulazione basata su agenti LLM (Large Language Model) che costruisce **gemelli digitali di ecosistemi di stakeholder** — politici, aziendali, finanziari, sanitari — e ne simula l'evoluzione nel tempo. A differenza dei tradizionali modelli di forecasting statistico, il sistema genera:

- **Contenuti realistici**: post social, comunicati stampa, reazioni di mercato, narrative emergenti
- **Dinamiche di opinione calibrate**: traiettorie validate su 52 scenari storici reali
- **Report esecutivi automatici**: analisi narrativa di coalizioni, polarizzazione, scenari forward
- **Visualizzazioni interattive**: dashboard live con replay temporale, grafi di rete, indicatori

**Il valore per i clienti di FormatLab**: anticipare reazioni a decisioni strategiche — lancio prodotti, ristrutturazioni aziendali, crisi reputazionali, policy pubbliche — prima che accadano nel mondo reale, con un costo di simulazione di **~$3-5 per scenario**.

---

## 2. Architettura del Sistema

### 2.1 Diagramma di Flusso Generale

```
                        ┌─────────────────────────┐
                        │   INPUT: Briefing        │
                        │   Testo libero o YAML    │
                        └────────┬────────────────┘
                                 │
                        ┌────────▼────────────────┐
                        │   BRIEFING ANALYZER      │
                        │   • Web research (opt.)  │
                        │   • Seed Data reali      │
                        │   • LLM: genera config   │
                        └────────┬────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         SCENARIO CONFIG              │
              │  • Elite agents (5-15 stakeholder)   │
              │  • Institutional agents (8-12)       │
              │  • Citizen clusters (3-8 segmenti)   │
              │  • Canali comunicativi               │
              │  • Asse posizionale [-1, +1]          │
              │  • Evento iniziale scatenante        │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         SIMULATION ENGINE            │
              │                                      │
              │  Per ogni round (tipicamente 3-9):   │
              │                                      │
              │  ┌─ Fase 1: Generazione Evento ──┐   │
              │  │  LLM genera evento emergente  │   │
              │  │  dalla situazione corrente     │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 2: Elite Agents ────────┐   │
              │  │  Ogni leader genera post,     │   │
              │  │  posizione, alleanze, targets  │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 3: Institutional Batch ─┐   │
              │  │  Istituzioni reagiscono in     │   │
              │  │  batch (cost-efficient)        │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 4: Citizen Clusters ────┐   │
              │  │  Segmenti demografici:        │   │
              │  │  sentiment, narrative, shift   │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 5: Platform Dynamics ───┐   │
              │  │  Engagement, follow graph,    │   │
              │  │  coalizioni (K-Means)          │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 6: Opinion Dynamics ────┐   │
              │  │  Modello calibrato:           │   │
              │  │  bounded confidence + herding  │   │
              │  │  + anchor + event shock        │   │
              │  └───────────────────────────────┘   │
              │  ┌─ Fase 7: Checkpoint ──────────┐   │
              │  │  Salva stato completo (JSON)   │   │
              │  └───────────────────────────────┘   │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         EVALUATION ENGINE            │
              │  • Realism Score (0-100)             │
              │  • Distribution plausibility         │
              │  • Drift realism                     │
              │  • Event coherence (LLM-as-judge)    │
              └──────────────────┬──────────────────┘
                                 │
         ┌───────────┬───────────┼────────────┬──────────┐
         ▼           ▼           ▼            ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────┐
    │ Report  │ │ SQLite  │ │ State  │ │ Frontend │ │Realism │
    │ .md     │ │ Social  │ │ JSON   │ │ Export   │ │ Score  │
    │ (exec)  │ │ DB      │ │ /round │ │ (13 file)│ │ .json  │
    └─────────┘ └─────────┘ └────────┘ └──────────┘ └────────┘
```

### 2.2 Stack Tecnologico

| Layer | Tecnologia | Ruolo |
|-------|-----------|-------|
| **LLM Engine** | Google Gemini Flash Lite / OpenAI GPT-4 | Generazione agenti, eventi, report |
| **Simulation Core** | Python 3.11+ / asyncio | Orchestrazione multi-agente |
| **Data Models** | Pydantic v2 | Validazione config e output |
| **Platform DB** | SQLite | Social media simulation (post, reactions, follows) |
| **Calibration** | Grid search + DTW + MAE | Ottimizzazione parametri su dati storici |
| **API** | FastAPI + SSE | Interfaccia REST con streaming |
| **Frontend** | Next.js 14 / D3.js / Recharts | Visualizzazione interattiva |
| **Export** | JSON strutturato | 13 file per scenario (replay, agenti, coalizioni, grafi) |

---

## 3. Il Modello a Tre Tier di Agenti

Il sistema simula ecosistemi complessi attraverso tre livelli di agenti, ognuno con un diverso livello di dettaglio e costo computazionale:

### Tier 1: Elite Agents (Leader di Opinione)

**Cosa sono**: Individui nominati con persona completa — CEO, politici, giornalisti, attivisti, analisti.

**Come funzionano**: Ogni elite agent riceve una chiamata LLM individuale per round. Genera:
- Post su piattaforme primarie e secondarie (es. social + TV)
- Nuova posizione [-1, +1] con reasoning
- Stato emotivo (combattivo, preoccupato, trionfante...)
- Alleanze e target (quali agenti supporta/attacca)
- Reazione a post virali del round precedente

**Esempio concreto** (scenario Bank Bitcoin Reserve Shock):

| Agente | Ruolo | Posizione | Stato | Narrativa |
|--------|-------|-----------|-------|-----------|
| Elena Varga | CEO banca | +0.56 | Cauta | "Non è speculazione, è diversificazione controllata" |
| Nadia Petrova | CEO crypto exchange | +0.89 | Trionfante | "Validazione istituzionale del Bitcoin" |
| Dr. Anika Weber | ECB supervisory | -0.87 | Preoccupata | "Rischio sistemico per i depositanti" |
| Marta Silva | Consumer advocate | -0.63 | Worried | "I risparmiatori non devono pagare" |
| Thomas Adler | Macro analyst | +0.11 | Neutro | "È un repricing del rischio di riserva" |

**Vincolo di calibrazione**: posizione max ±0.15 per round (i leader non cambiano idea da un giorno all'altro).

### Tier 2: Institutional Agents (Organizzazioni)

**Cosa sono**: Istituzioni, enti regolatori, sindacati, associazioni di categoria — attori collettivi.

**Come funzionano**: Processati in batch (fino a 10 per chiamata LLM) per efficienza di costo. Ogni istituzione genera un comunicato e aggiorna la propria posizione. Vincolo: ±0.12/round.

### Tier 3: Citizen Clusters (Popolazione)

**Cosa sono**: Segmenti demografici aggregati — "Risparmiatori retail" (8.5M), "Crypto enthusiasts" (1.2M), "Pensionati risk-averse" (3M).

**Come funzionano**: Ogni cluster simula la reazione aggregata di migliaia/milioni di persone. Genera:
- Distribuzione sentiment (positivo/negativo/indifferente/confuso, somma = 100%)
- Narrativa emergente dominante
- Shift di posizione (vincolo: ±0.10/round)
- Livello di engagement (quanto il segmento partecipa al dibattito)

**Perché tre tier**: permette di simulare ecosistemi realistici (10 leader + 10 istituzioni + 5 segmenti = 10M+ cittadini) con solo ~25-30 chiamate LLM per round, mantenendo il costo sotto $5/simulazione.

---

## 4. Sistema di Calibrazione: Ancoraggio alla Realtà

### 4.1 Il Problema

Un sistema multi-agente basato su LLM, senza vincoli empirici, produce output che suonano sofisticati ma sono sostanzialmente **allucinazioni strutturate**. Le dinamiche di opinione, la velocità dei cambiamenti, l'intensità delle reazioni — tutto è inventato.

### 4.2 La Soluzione: 52 Scenari Storici Reali

Abbiamo costruito un corpus di **52 scenari di calibrazione** con dati reali (sondaggi, esiti, eventi chiave) su cui il modello matematico di Opinion Dynamics viene ottimizzato:

```
CALIBRAZIONE (offline, $0, ~55 minuti)
│
├── 52 scenari storici con traiettorie reali
│   ├── Polling a 9 time-step per scenario
│   ├── 3-5 eventi chiave con timing reale
│   └── Esito finale documentato
│
├── Simulazione sintetica (solo matematica, no LLM)
│   └── 972 combinazioni di 7 parametri
│
├── Confronto con ground truth
│   ├── Outcome Accuracy (peso 50%)
│   ├── Mean Absolute Error (peso 30%)
│   └── Dynamic Time Warping (peso 20%)
│
└── Output: parametri ottimali per dominio
    ├── political:  anchor=0.05  social=0.08  event=0.12  ...
    ├── corporate:  anchor=0.12  social=0.15  event=0.06  ...
    ├── financial:  anchor=0.08  social=0.10  event=0.08  ...
    └── [altri 12 domini]
```

### 4.3 Copertura dei Domini

| Dominio | # Scenari | Esempi |
|---------|-----------|--------|
| **Political/Referendum** | 12 | Brexit, Referendum Renzi 2016, Cile 2022, Colombia, Irlanda, Svizzera |
| **Corporate** | 8 | Boeing 737 MAX, VW Dieselgate, Wells Fargo, Amazon HQ2, Tesla/Twitter |
| **Sport** | 5 | Super League, F1 budget cap, Qatar WC, VAR, NFL protests |
| **Technology** | 5 | Apple ATT, GDPR, AI regulation, TikTok ban, Google antitrust |
| **Social/Cultural** | 4 | #MeToo, Depp/Heard, Bud Light/Mulvaney, marijuana legalization |
| **Environmental** | 3 | Parigi clima, plastic straw ban, nucleare post-Fukushima |
| **Labor/Economic** | 4 | Remote work, UBI, salario minimo $15, Prop 22 |
| **Marketing** | 2 | Nike/Kaepernick, Barilla LGBT |
| **Financial** | 2 | GameStop/WSB, crypto regulation post-FTX |
| **Public Health** | 2 | COVID vaccine mandate, UK smoking ban |

### 4.4 Risultati della Calibrazione

Validazione su scenari-test (parametri ottimizzati vs dati storici reali):

| Scenario | Simulato | Reale | Errore |
|----------|----------|-------|--------|
| Referendum Renzi 2016 | 41.2% SÌ | 40.9% SÌ | **0.3%** |
| Costituzione Cile 2022 | 38.3% Apruebo | 38.1% Apruebo | **0.2%** |
| UK Smoking Ban 2006 | 84.8% favore | 78.0% favore | 6.8% |

I parametri calibrati vengono caricati automaticamente nelle simulazioni future dello stesso dominio.

---

## 5. Output Generati dal Sistema

### 5.1 Report Esecutivo Automatico (Markdown, 10-25 pagine)

Per ogni simulazione, il sistema genera un report narrativo completo con:

1. **Executive Summary** — risultato chiave, polarizzazione, dinamica dominante
2. **Timeline Simulata** — evoluzione round-by-round con metriche
3. **Mappa Coalizioni** — chi si allea con chi, perché, come evolve
4. **Sentiment per Segmento** — analisi psicografica per cluster demografico
5. **Polarizzazione e Herding** — echo chamber, cascate FOMO/FUD
6. **Post Virali** — i 5 post che hanno mosso il dibattito (con analisi impatto)
7. **Narrativa Emergente** — quale framing ha vinto e perché
8. **Impatto Istituzionale** — implicazioni regolatorie e sistemiche
9. **Scenari Forward** — 3 percorsi futuri con probabilità e condizioni

**Estratto reale** (scenario Bank Bitcoin Reserve Shock):

> *"The most plausible mind-shift occurred among Young Finance Optimists and Institutional Clients. The former likely moved from curiosity toward guarded acceptance as the bank's cautious messaging framed the event as manageable. Institutional Clients may have moved from initial skepticism toward a neutral 'wait-and-see' stance as the simulation progressed."*

### 5.2 Database Social Media (SQLite)

Ogni simulazione produce un database completo con:
- **200-400+ post** per scenario (testo completo, piattaforma, autore, round)
- **1000-5000 reazioni** (like, repost, reply) con grafo sociale
- **Follow graph** che evolve nel tempo
- **Query SQL** per analisi custom (es. "tutti i post del CEO ordinati per engagement")

### 5.3 State Checkpoint (JSON per round)

Snapshot dello stato di ogni agente per round — posizione, sentiment, engagement, alleanze. Permette:
- Analisi di traiettoria individuale
- Identificazione di turning point
- Replay e branching ("cosa succede se al round 3 cambiamo l'evento?")

### 5.4 Frontend Interattivo (Next.js)

Dashboard web con:

```
┌──────────────────────────────────────────────────────────┐
│  TOP BAR: Scenario selector | Round navigator | Speed   │
├────────────┬──────────────────────────┬──────────────────┤
│            │                          │                  │
│  NETWORK   │    LIVE POST FEED        │   INDICATORS     │
│  GRAPH     │                          │                  │
│  (D3.js)   │  Post appaiono in tempo  │  • Polarization  │
│            │  reale con engagement,   │  • Sentiment     │
│  Nodi =    │  autore, piattaforma     │  • Trending      │
│  agenti    │                          │  • Coalitions    │
│            │  Click = evidenzia       │  • Agent list    │
│  Colore =  │  impatto su grafo        │                  │
│  posizione │                          │                  │
├────────────┴──────────────────────────┴──────────────────┤
│  BOTTOM BAR: Timeline | Coalizione Sankey | Playback     │
└──────────────────────────────────────────────────────────┘
```

- **Replay temporale**: play/pause, velocità 1x-8x, navigazione per round
- **Grafo di rete D3.js**: nodi = agenti, colore = posizione, edge = connessioni
- **Knowledge Graph evolving**: 8 snapshot con slider temporale, sparkline, delta
- **Post-to-Graph causality**: click su un post → evidenzia agenti influenzati nel grafo

### 5.5 Realism Score (0-100)

Valutazione post-simulazione automatica:
- **Distribution plausibility** (25%): le posizioni non sono tutte agli estremi?
- **Drift realism** (25%): nessun agente ha shiftato più di 0.5 in totale?
- **Alliance consistency** (20%): le alleanze riflettono la prossimità di posizione?
- **Event coherence** (30%): gli eventi generati sono plausibili? (LLM-as-judge)

---

## 6. Casi d'Uso per i Clienti FormatLab

### 6.1 Crisis Anticipation & Management

**Cliente tipo**: Grande azienda che deve prendere una decisione controversa.

**Esempio**: "Vogliamo annunciare una ristrutturazione con taglio del 30% del middle management. Come reagiranno stakeholder interni ed esterni?"

**Output**: Simulazione a 4-6 settimane che mostra:
- Coalizioni che si formano (sindacato + media vs management + investitori)
- Narrativa dominante che emerge ("L'AI sostituisce i lavoratori" vs "Innovazione necessaria")
- Segmenti di popolazione più a rischio di radicalizzazione
- Finestra temporale ottimale per comunicazioni correttive
- 3 scenari forward con probabilità

**Valore**: il cliente testa la comunicazione PRIMA dell'annuncio, iterando sulla strategia.

### 6.2 Brand & Reputation Stress Test

**Cliente tipo**: Brand che lancia una campagna controversa o affronta una crisi reputazionale.

**Esempio**: "Nike ha subito un deepfake ad virale. Come evolve il sentiment in 4 settimane?"

**Output**:
- Traiettoria del brand favorability per segmento demografico
- Identificazione degli opinion leader più influenti (pro e contro)
- Post virali simulati che anticipano i frame narrativi reali
- Analisi dell'effetto echo chamber e delle cascate di boycott/support
- Purchase intent simulato per cluster di consumatori

### 6.3 Policy & Regulatory Impact Assessment

**Cliente tipo**: Ente pubblico, associazione di categoria, think tank.

**Esempio**: "Come reagirebbe l'opinione pubblica a un obbligo vaccinale per il personale sanitario?"

**Output**:
- Polarizzazione per segmento (sanitari, genitori, no-vax, istituzioni)
- Effetto delle dichiarazioni dei leader politici sulla traiettoria
- Identificazione dei punti di rottura (quando il consenso crolla o decolla)
- Calibrazione su 2 scenari storici: COVID vaccine mandate USA + UK smoking ban

### 6.4 M&A and Strategic Decision Simulation

**Cliente tipo**: Fondo di investimento, direzione strategica di una corporate.

**Esempio**: "Una banca europea annuncia riserve in Bitcoin. Come reagiscono mercati, regolatori, clienti retail?"

**Output reale prodotto dal sistema** (Bank Bitcoin Reserve Shock):

> *"Polarization remained elevated throughout: 6.2/10 in Week 1, 6.1 in Week 2, 6.0 in Week 3. The market settled into a persistent contested equilibrium. The winning narrative was risk repricing under institutional experimentation — not pure innovation optimism, nor pure regulatory fear."*

Il report identifica:
- **Bull coalition**: Crypto exchange CEO (+0.89) + Crypto Enthusiasts (+1.00)
- **Bear coalition**: ECB official (-0.87) + Consumer advocate (-0.63) + Pensionati (-1.00)
- **Swing audience**: Institutional Clients (-0.11, engagement 0.8) — il segmento decisivo
- **Forward scenarios**: normalizzazione controllata vs regulatory tightening vs adoption wave

### 6.5 Political Campaign Simulation

**Cliente tipo**: Partito politico, comitato referendario, ente istituzionale.

**Esempio**: "Simulare la campagna per il referendum sulla separazione delle carriere dei magistrati."

**Output con seed data reali** (12 stakeholder verificati: Meloni, Nordio, Schlein, Conte...):
- Traiettoria voto SÌ/NO per 9 mesi con eventi chiave
- Effetto delle dichiarazioni di ogni leader sull'opinione
- Segmenti demografici kingmaker
- Scenario what-if: "cosa succede se Berlusconi cambia posizione al round 5?"

---

## 7. Integrazione RAG e Pre-addestramento per Cliente

### 7.1 Il Meccanismo: Seed Data

Ogni simulazione può essere **ancorata a dati reali del cliente** attraverso il sistema di Seed Data:

```
seed_data/corporate/client_xyz/
├── context.md              # Background aziendale, settore, competitor
├── historical.md           # Precedenti storici rilevanti
├── stakeholders.json       # Stakeholder reali con posizioni e citazioni
├── demographics.json       # Segmenti target con profili psicografici
└── known_events.json       # Timeline di eventi reali da ancorare
```

### 7.2 Come Funziona nella Pratica

```
┌───────────────────────────────────────┐
│  DATI DEL CLIENTE (RAG layer)         │
│                                       │
│  • Organigramma con posizioni note    │
│  • Survey interne (engagement, NPS)   │
│  • Media monitoring (rassegna stampa) │
│  • Social listening (sentiment live)  │
│  • Dati di mercato (market share)     │
│  • Report analisti                    │
└──────────────┬────────────────────────┘
               │
               ▼
┌───────────────────────────────────────┐
│  SEED DATA LOADER                     │
│                                       │
│  Converte dati cliente in:            │
│  • VerifiedStakeholder (Pydantic)     │
│  • VerifiedDemographic (Pydantic)     │
│  • Historical context + known events  │
│                                       │
│  Iniettati nei prompt LLM:           │
│  → Briefing usa stakeholder reali    │
│  → Events generati su base storica   │
│  → Cluster demografici realistici    │
└───────────────────────────────────────┘
```

**Esempio**: Un cliente retail fornisce:
- Risultati di una survey interna: 62% dipendenti contrari a policy X
- Nomi e posizioni note di 8 leader interni
- 3 articoli di stampa recenti sulla questione

Il sistema li converte in `stakeholders.json` con posizioni calibrate e `context.md` con il background, producendo una simulazione **ancorata alla realtà specifica del cliente**, non generica.

### 7.3 Vantaggio Competitivo del RAG

| Senza Seed Data | Con Seed Data (RAG) |
|-----------------|---------------------|
| Agenti inventati dal LLM | Agenti basati su persone reali con posizioni verificate |
| Eventi generici plausibili | Eventi ancorati a precedenti storici del settore |
| Cluster demografici standard | Segmenti derivati da survey reali del cliente |
| Output: "interesting fiction" | Output: **scenario planning empiricamente fondato** |

---

## 8. Workflow Consulenziale FormatLab → Cliente

### Fase 1: Discovery (1-2 giorni)

FormatLab raccoglie dal cliente:
- Decisione/scenario da simulare (1 paragrafo di briefing)
- Stakeholder noti e loro posizioni (anche informali)
- Dati esistenti (survey, media monitoring, report interni)
- KPI da monitorare (engagement dipendenti, brand favorability, voting intention...)

### Fase 2: Setup Simulazione (1 giorno)

FormatLab:
- Costruisce seed data dal materiale del cliente
- Configura lo scenario (YAML o generazione da briefing testuale)
- Seleziona il dominio e i parametri calibrati corrispondenti
- Definisce numero di round e timeline (giorni, settimane, mesi)

### Fase 3: Esecuzione (30-90 minuti)

```bash
# Lancio simulazione
python run.py --config configs/client_xyz.yaml \
  --domain corporate --budget 5.0 --rounds 6

# Export per frontend
python export.py --scenario client_xyz --output-dir exports/

# Report stampabile
# → outputs/client_xyz_report.md (auto-generato)
```

Costo LLM: **$3-5** per esecuzione. Possibilità di run multipli per sensitivity analysis.

### Fase 4: Analisi e Deliverable (1-2 giorni)

FormatLab produce per il cliente:
1. **Report esecutivo** (auto-generato + commento consulenziale)
2. **Dashboard interattivo** (link web al frontend con replay)
3. **Scenario comparison** (2-3 varianti: base case, best case, worst case)
4. **Raccomandazioni strategiche** basate sui pattern emergenti

### Fase 5: Iterazione (opzionale)

- "Cosa succede se cambiamo il CEO spokesperson al round 3?"
- "E se il sindacato facesse un'azione il primo giorno?"
- "Simuliamo con 2 strategie comunicative diverse e confrontiamo"

Il sistema supporta **branching**: si parte dallo stato di un round specifico e si esplora un percorso alternativo.

---

## 9. Scenari Già Disponibili (13 simulazioni complete)

| Scenario | Dominio | Round | Agenti | Descrizione |
|----------|---------|-------|--------|-------------|
| Bank Bitcoin Reserve Shock | Financial | 3 | 18 | Banca EU annuncia riserve in Bitcoin |
| Flat Org AI Restructuring | Corporate | 4 | 22 | Fortune 500 elimina tutto il middle management |
| Global iPhone Price Shock | Commercial | 5 | 20 | Apple aumenta prezzi del 40% |
| Nike Deepfake Ad Backlash | Marketing | 4 | 15 | Deepfake virale attribuito a Nike |
| Italy mRNA Booster Mandate | Public Health | 5 | 20 | Obbligo vaccinale per sanitari |
| Luxury Leather Transition | Commercial | 6 | 18 | Luxury brand abbandona la pelle |
| Cree Lighting Restructuring | Corporate | 5 | 16 | Crisi ristrutturazione azienda lighting |
| Arianna Spa Newco Transfer | Corporate | 5 | 14 | Trasferimento newco e percezione mercato |
| AI Generative Expert | Corporate | 5 | 12 | Nomina esperto AI in organizzazione |
| Corsa Civica per Padova | Political | 5 | 16 | Candidatura civica elezioni comunali |
| City Green Light Ownership | Corporate | 5 | 14 | Transizione proprietà utility |
| Leadership Challenge FIGB | Sport | 5 | 14 | Sfida alla leadership federazione sportiva |
| Ronaldo a Padova | Sport | 5 | 14 | Scenario calcistico con colletta tifosi |

Tutti con report, database, export frontend, stato per round.

---

## 10. Metriche di Qualità e Costo

### 10.1 Qualità della Simulazione

| Metrica | Target | Metodo |
|---------|--------|--------|
| Outcome accuracy vs dati storici | < 5% errore | Calibrazione su 52 scenari |
| Trajectory MAE | < 0.05 | Dynamic Time Warping + MAE |
| Realism score | > 70/100 | Post-evaluation automatica |
| Position drift per round | ≤ 0.15 (elite) | Constraint enforcement |
| Sensitivity variance | CV < 15% | Multi-run analysis |
| Event plausibility | Score > 4/10 | LLM-as-judge con rigenerazione |

### 10.2 Costi

| Componente | Costo per Simulazione |
|-----------|----------------------|
| LLM calls (Gemini Flash Lite) | $3-5 |
| Calibrazione (una tantum, offline) | $0 |
| Evaluation (realism scoring) | ~$0.025 |
| Infrastructure (compute locale) | ~$0 |
| **Totale per simulazione** | **~$3-5** |
| **Totale per progetto consulenziale** (5-10 run) | **~$15-50** |

Il costo marginale quasi-zero permette di eseguire **decine di varianti** per ogni progetto consulenziale.

### 10.3 Tempi

| Fase | Durata |
|------|--------|
| Setup scenario (da YAML) | 5 minuti |
| Setup scenario (da briefing testuale) | 15-30 minuti |
| Esecuzione simulazione (3 round) | 15-20 minuti |
| Esecuzione simulazione (9 round) | 45-90 minuti |
| Export + report generation | 2-5 minuti |
| Calibrazione completa (52 scenari) | ~55 minuti (una tantum) |

---

## Appendice A: Schema di un Agente Elite (Pydantic)

```python
class AgentSpec(BaseModel):
    id: str                                    # "ceo_bank"
    name: str                                  # "Elena Varga"
    role: str                                  # "CEO della banca annunciante"
    archetype: str                             # "business_leader"
    position: float = Field(ge=-1.0, le=1.0)   # 0.7 (pro)
    influence: float = Field(ge=0.0, le=1.0)   # 0.95
    rigidity: float = Field(ge=0.0, le=1.0)    # 0.8
    tier: int = 1
    platform_primary: str                      # "press_release"
    platform_secondary: str                    # "market_wire"
    bio: str                                   # Biografia per system prompt
    communication_style: str                   # "Cauta, istituzionale"
```

## Appendice B: Modello Matematico di Opinion Dynamics

```
position(t+1) = position(t) + Δ

Δ = anchor_pull + social_pull + event_shock + herd_pull

anchor_pull = anchor_weight × rigidity × (original_position - position)
social_pull = social_weight × Σ(w_i × (author_i_pos - position)) / Σ(w_i)
              dove w_i = influence_i × engagement_i
              solo se |author_i_pos - position| < tolerance  (bounded confidence)
event_shock = event_weight × shock_magnitude × shock_direction × (1 - rigidity)
herd_pull   = herd_weight × (feed_avg - position) × (1 - rigidity)
              solo se |feed_avg - position| > herd_threshold

Vincoli: position ∈ [-1, +1], Δ ∈ [-delta_cap, +delta_cap]
```

**7 parametri calibrabili** su dati storici: `anchor_weight`, `social_weight`, `event_weight`, `herd_weight`, `herd_threshold`, `direct_shift_weight`, `anchor_drift_rate`.

## Appendice C: Come Lanciare una Simulazione

```bash
# 1. Da file di configurazione YAML
python run.py --config configs/bank_bitcoin.yaml --domain financial

# 2. Da briefing testuale (il sistema genera il config)
python run.py --brief "Un'azienda farmaceutica ritira un farmaco dal mercato
  dopo segnalazioni di effetti collaterali. Simulare la reazione di pazienti,
  medici, regolatori, media e investitori per 6 settimane."

# 3. Via API REST
curl -X POST http://localhost:8000/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"brief": "...", "domain": "public_health", "rounds": 6}'

# 4. Calibrare prima di simulare
python -m calibration.run_calibration --domain corporate
cp calibration/results/calibrated_params_corporate.json outputs/calibrated_params.json
python run.py --config configs/my_scenario.yaml
```

---

*Documento generato il 27 marzo 2026 — DigitalTwinSim v1.0*
*Contatto tecnico: Alberto Giovanni Gerli*
