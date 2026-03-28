# Grounding the Simulation in Reality — Relazione Tecnica

## DigitalTwinSim Calibration & Grounding System

**Versione**: 1.0
**Data**: 27 Marzo 2026
**Autore**: DigitalTwinSim Team

---

## 1. Executive Summary

Il sistema DigitalTwinSim genera comportamenti, eventi e traiettorie di opinione interamente tramite LLM (Gemini Flash Lite). Senza ancoraggi empirici, gli output — per quanto sofisticati — sono allucinazioni strutturate. Il predecessore referendum-sim era grounded perché usava politici reali, dati ISTAT e contesto politico verificato.

Questo documento descrive il **sistema di calibrazione e grounding** implementato per replicare quel grounding in modo domain-agnostic, articolato su quattro livelli:

1. **Seed Data + Constraint Tightening** — Dati verificati e vincoli rigidi
2. **Few-Shot Prompting** — Esempi golden per guidare qualità degli output
3. **Calibrazione Storica** — 16 scenari reali per ottimizzare i parametri del modello
4. **Evaluation Framework** — Valutazione automatica di realismo post-simulazione

---

## 2. Architettura del Sistema di Grounding

```
                         ┌──────────────────────────┐
                         │    SEED DATA BUNDLE       │
                         │  Stakeholder verificati   │
                         │  Demografici reali        │
                         │  Contesto storico         │
                         │  Eventi noti              │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Briefing         │    │ Event Injector   │    │ Agent System    │
    │ Analyzer         │    │                  │    │ Prompts         │
    │                  │    │ + Historical ctx  │    │                 │
    │ Inietta          │    │ + Few-shot        │    │ + Bio verificata│
    │ stakeholder      │    │ + Plausibility    │    │ + Posizioni     │
    │ verificati       │    │   check (LLM)    │    │   reali         │
    │ nel prompt       │    │ + Shock clamping  │    │ + Quote vere    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │      VALIDATORS           │
                         │  Delta cap: ±0.15 (elite) │
                         │  Delta cap: ±0.10 (cluster)│
                         │  Shock: [0.1, 0.6]        │
                         │  Agent ref validation      │
                         │  Sentiment normalization   │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ OpinionDynamics  │    │ Realism Scorer   │    │ Sensitivity     │
    │ (calibrato)      │    │                  │    │ Analysis        │
    │                  │    │ Score 0-100      │    │                 │
    │ 5 parametri      │    │ 4 sotto-metriche │    │ N run, varianza │
    │ da grid search   │    │ + LLM-as-judge   │    │ Target: <15%    │
    │ su dati storici  │    │                  │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 3. Fase 1: Seed Data e Vincoli

### 3.1 Struttura del Seed Data

Ogni scenario può essere ancorato a dati reali tramite un **SeedDataBundle**, caricato da una directory strutturata:

```
seed_data/political/italy_referendum_2026/
├── context.md           # Contesto politico verificato
├── historical.md        # Precedenti storici (referendum 2001, 2006, 2016)
├── stakeholders.json    # 12 attori reali con posizioni, bio, citazioni
├── demographics.json    # 5 segmenti demografici con dati reali
└── known_events.json    # 6 eventi reali per ancorare la timeline
```

**Modelli Pydantic** (`seed_data/schema.py`):

| Modello | Campi chiave | Scopo |
|---------|-------------|-------|
| `VerifiedStakeholder` | name, role, known_position, position_source, bio_verified, key_quotes, archetype, influence, rigidity | Evita che l'LLM inventi nomi e posizioni |
| `VerifiedDemographic` | name, description, population_share, known_position, key_concerns, info_channel | Ancora i cluster a dati demografici reali |
| `SeedDataBundle` | context_text, stakeholders, demographics, historical_text, known_events | Aggregatore con metodi di formattazione per i prompt |

### 3.2 Esempio: Stakeholder verificati (Italia 2026)

| Stakeholder | Ruolo | Posizione | Fonte | Influenza |
|-------------|-------|-----------|-------|-----------|
| Giorgia Meloni | Presidente del Consiglio | +0.85 | public_statement | 0.95 |
| Carlo Nordio | Ministro della Giustizia | +0.95 | voting_record | 0.85 |
| Elly Schlein | Segretaria PD | -0.85 | public_statement | 0.80 |
| Giuseppe Conte | Leader M5S | -0.80 | public_statement | 0.70 |
| Presidente ANM | Assoc. Naz. Magistrati | -0.95 | institutional_statement | 0.70 |
| Presidente UCPI | Camere Penali | +0.90 | institutional_statement | 0.55 |
| Maurizio Landini | Segretario CGIL | -0.75 | public_statement | 0.60 |
| Carlo Calenda | Leader Azione | -0.20 | public_statement | 0.35 |

Ogni stakeholder include **citazioni reali** usate per calibrare il tono. Esempio per Meloni:
> *"La separazione delle carriere non è una riforma contro i magistrati, è una riforma a favore dei cittadini."*

### 3.3 Vincoli di Output (validators.py)

I vincoli sono applicati a runtime su ogni output dell'LLM:

| Vincolo | Tier 1 (Elite) | Tier 2 (Istituzioni) | Tier 3 (Cluster) |
|---------|----------------|---------------------|-------------------|
| **Delta posizione max/round** | ±0.15 | ±0.12 | ±0.10 |
| **Shock magnitude** | [0.1, 0.6] | [0.1, 0.6] | — |
| **Alliance/target validation** | Solo ID esistenti | Solo ID esistenti | — |
| **Sentiment normalization** | — | — | Somma = 100 |

**Plausibility check** (LLM-as-judge): ogni evento generato viene valutato con temp 0.1 e max 200 token. Se il punteggio è < 4/10, l'evento viene rigenerato con temperatura più bassa (0.6) e un feedback esplicito sull'implausibilità.

Costo del plausibility check: **~$0.001/evento**.

---

## 4. Fase 2: Few-Shot Prompting

Ogni domain plugin può fornire **esempi golden** che vengono iniettati nei prompt come riferimento di qualità. Gli esempi non vengono copiati dall'LLM — servono come benchmark di formato, tono e realismo.

### Esempi disponibili (dominio politico)

| File | Scopo | Caratteristiche chiave |
|------|-------|----------------------|
| `elite_round.json` | Risposta round agente elite | Post su social/TV, posizione +0.82, alliances con ID reali, reasoning coerente |
| `cluster_round.json` | Risposta round cluster cittadini | Sentiment distribution sommata a 100, shift ±0.05, 3 post-campione con archetipo |
| `event.json` | Evento generato per un round | Istituzioni reali (CSM), shock 0.45, 4-5 frasi dettagliate, attori coinvolti |

**Costo impatto**: zero (testo statico aggiunto ai prompt esistenti).

**Iniezione nel pipeline**: il `RoundManager` appende gli esempi ai prompt di elite e cluster; l'`EventInjector` include l'esempio evento nel contesto di generazione.

---

## 5. Fase 3: Calibrazione Storica

### 5.1 Il Dataset di Calibrazione

Il cuore del sistema è un corpus di **16 scenari storici reali** con dati verificati, che coprono **7 domini** e **7 pattern di dinamica opinionale** distinti.

#### Scenari politici (6)

| Scenario | Esito | Affluenza | Pattern testato |
|----------|-------|-----------|-----------------|
| **Brexit 2016** | Leave 51.9% | 72.2% | Late surge + shy voters |
| **Referendum Renzi 2016** | NO 59.1% | 65.5% | Personalizzazione = sconfitta |
| **Italia Costituzionale 2006** | NO 61.3% | 52.5% | Momentum unidirezionale |
| **Scozia 2014** | No 55.3% | 84.6% | Narrowing + reversion a status quo |
| **Greferendum 2015** | OXI 61.3% | 62.5% | Compressione estrema (8 giorni), polling error 10+pts |
| **Australia SSM 2017** | Yes 61.6% | 79.5% | Stabilità su temi maturi |

#### Scenari corporate (4)

| Scenario | Sentiment finale | Pattern testato |
|----------|-----------------|-----------------|
| **Boeing 737 MAX** | Pro 12%, Contro 68% | Doppio failure → crollo credibilità |
| **VW Dieselgate** | Pro 22%, Contro 48% | Frode deliberata → drop immediato |
| **Facebook/Cambridge Analytica** | Pro 18%, Contro 55% | Too-big-to-fail + network lock-in |
| **Google Antitrust** | Pro 56%, Contro 28% | Slow-burn istituzionale (4 anni, opinione stabile) |

#### Scenari commercial/marketing/financial/public health (6)

| Scenario | Dominio | Pattern testato |
|----------|---------|-----------------|
| **Super League 2021** | commercial | Cascade unanime in 48h |
| **Barilla 2013-2015** | commercial | Crisi → riabilitazione completa (2 anni) |
| **Nike/Kaepernick 2018** | marketing | Polarizzazione + recovery (check-mark) |
| **GameStop/WSB 2021** | financial | Grassroots vs istituzioni |
| **Vaccini USA 2021** | public_health | Polarizzazione da policy |
| **Smoking Ban UK 2006** | public_health | Curva di accettazione post-implementazione |

### 5.2 Traiettorie di Polling

Ogni scenario contiene **9 data point** di polling reale mappati ai round della simulazione. Esempio per Brexit:

```
Round 1 (Jan 2016):    Leave 41% — Remain 42% — Undecided 17%
Round 2 (Feb 2016):    Leave 40% — Remain 44% — Undecided 16%
...
Round 7 (Early Jun):   Leave 46% — Remain 42% — Undecided 12%  ← Leave peak
Round 8 (Mid Jun):     Leave 44% — Remain 44% — Undecided 12%  ← Post Jo Cox
Round 9 (Jun 23):      Leave 52% — Remain 48%                   ← Risultato
```

Ogni scenario include inoltre **5-6 eventi chiave reali** con timing, permettendo di verificare se la simulazione genera eventi di impatto comparabile nei round giusti.

### 5.3 Metriche di Calibrazione

Il `trajectory_comparator.py` calcola tre metriche:

| Metrica | Formula | Peso nel composito | Cosa misura |
|---------|---------|-------------------|-------------|
| **Outcome Accuracy** | \|sim_pro% - real_pro%\| | 50% | Errore sull'esito finale |
| **Position MAE** | Media degli errori assoluti per round | 30% | Aderenza alla traiettoria |
| **Trajectory DTW** | Dynamic Time Warping distance | 20% | Similarità della *forma* della curva |

Il **composite score** (più basso = migliore) pondera queste tre metriche.

### 5.4 Grid Search dei Parametri

Il modello di opinione (`OpinionDynamics`) ha **5 parametri** che governano le forze che agiscono sulla posizione di ogni agente:

| Parametro | Default | Range grid | Descrizione |
|-----------|---------|-----------|-------------|
| `anchor_weight` | 0.10 | [0.05, 0.10, 0.15, 0.20] | Forza di richiamo alla posizione originale |
| `social_weight` | 0.15 | [0.05, 0.10, 0.15, 0.20, 0.25] | Influenza sociale dal feed |
| `event_weight` | 0.05 | [0.02, 0.05, 0.08, 0.12] | Impatto degli eventi esogeni |
| `herd_weight` | 0.05 | [0.02, 0.05, 0.08] | Effetto gregge |
| `herd_threshold` | 0.20 | [0.15, 0.20, 0.30] | Soglia di distanza per l'effetto gregge |

**Combinazioni totali**: 4 × 5 × 4 × 3 × 3 = **720 set di parametri**.

Per ogni combinazione, la simulazione viene eseguita sullo scenario storico e il composite score viene calcolato contro il ground truth. Il set con il punteggio più basso viene salvato come `calibrated_params.json` e caricato automaticamente nelle simulazioni successive.

**Costo stimato**: ~$15-25 per calibrazione completa (10-20 run × ~$1-1.50/run).

### 5.5 Caricamento dei Parametri Calibrati

```python
# In RoundManager.__init__():
calibrated_path = os.path.join(checkpoint_dir, "calibrated_params.json")
self.opinion_dynamics = OpinionDynamics(calibrated_params_path=calibrated_path)
```

Se il file esiste, i parametri vengono caricati; altrimenti si usano i default. Questo permette calibrazione incrementale per dominio.

---

## 6. Fase 4: Evaluation Framework

### 6.1 Realism Score (0-100)

Dopo ogni simulazione, il sistema calcola automaticamente un punteggio di realismo composito:

| Sotto-metrica | Peso | Cosa controlla |
|---------------|------|---------------|
| **Distribution Plausibility** | 25% | Agenti non tutti agli estremi? Varianza ragionevole? Moderati presenti? |
| **Drift Realism** | 25% | Nessun agente ha fatto shift totale > 0.5? |
| **Alliance Consistency** | 20% | Le alleanze sono tra agenti con posizioni compatibili (distanza < 0.8)? |
| **Event Coherence** | 30% | LLM-as-judge valuta coerenza narrativa dell'intera sequenza di eventi (0-100) |

**Output esempio**:
```
┌─ REALISM EVALUATION ─────────────────────
│  Score: 74/100
│  Distribution: 85 | Drift: 80 | Events: 62
│  ⚠ [drift] agent_meloni: drifted 0.52 (>0.5)
│  ⚠ [events] Weak causal connection between Round 4 and 5
└────────────────────────────────────────────
```

### 6.2 Sensitivity Analysis

Per scenari ad alta posta, si possono eseguire N run dello stesso scenario e misurare la varianza:

| Metrica | Target | Significato |
|---------|--------|------------|
| **Polarization CV%** | < 15% | Coefficiente di variazione della polarizzazione finale |
| **Avg Agent Position Stdev** | < 0.15 | Deviazione standard media delle posizioni finali |
| **is_stable** | true | CV < 15% AND avg_stdev < 0.15 |

---

## 7. Copertura dei Pattern di Dinamica Opinionale

I 16 scenari coprono un catalogo completo di pattern, utilizzabili per validare qualsiasi dominio:

### Pattern e scenari di riferimento

| # | Pattern | Scenari | Dinamica chiave |
|---|---------|---------|-----------------|
| 1 | **Late surge** | Brexit | Undecided break for change; shy voters |
| 2 | **Personalizzazione** | Renzi 2016 | Leader lega destino al voto → plebiscito anti-governo |
| 3 | **Momentum unidirezionale** | Italia 2006 | Riforma orfana, NO cresce costantemente |
| 4 | **Narrowing + reversion** | Scozia 2014 | Challenger si avvicina ma status quo bias prevale |
| 5 | **Compressione estrema** | Grecia 2015 | 8 giorni, polling error 10+pts, rally-around-flag |
| 6 | **Stabilità su temi maturi** | Australia SSM | Campagne mobilizzano ma non convertono |
| 7 | **Doppio failure** | Boeing MAX | Secondo crash distrugge credibilità residua |
| 8 | **Frode deliberata** | VW Dieselgate | Drop immediato, recovery lenta via remediation |
| 9 | **Too-big-to-fail** | Facebook/CA | Trust crolla ma utenti restano (network lock-in) |
| 10 | **Slow-burn istituzionale** | Google Antitrust | Opinione stabile per anni (~55-59%) |
| 11 | **Cascade unanime** | Super League | Opposizione universale in 48h, zero constituency pro |
| 12 | **Crisi → riabilitazione** | Barilla | Position reversal completo in 2 anni |
| 13 | **Polarizzazione + recovery** | Nike/Kaepernick | Boycott vocale ma vendite +31%, stock all-time high |
| 14 | **Grassroots vs istituzioni** | GameStop/WSB | Retail vs hedge funds, regole cambiate mid-game |
| 15 | **Polarizzazione da policy** | Vaccini USA | Executive action collassa undecideds in opposizione |
| 16 | **Curva di accettazione** | Smoking Ban UK | Da 48% a 78%: proof-of-concept + esperienza diretta |

---

## 8. Fonti dei Dati

### Fonti per scenario

| Scenario | Fonti principali |
|----------|-----------------|
| Brexit | What UK Thinks EU, British Polling Council, Electoral Commission UK |
| Renzi 2016 | Ministero dell'Interno, sondaggi SWG/EMG/Tecnè |
| Italia 2006 | Ministero dell'Interno, Ipsos Italia |
| Scozia 2014 | What Scotland Thinks, ScotCen, Electoral Commission |
| Grecia 2015 | Hellenic Ministry of Interior, VPRC/GPO/Prorata polls |
| Australia SSM 2017 | Australian Bureau of Statistics, Newspoll, Essential Research |
| Boeing 737 MAX | NTSB reports, Boeing SEC filings, Congressional hearing records |
| VW Dieselgate | EPA Notice of Violation, DOJ filings, VW annual reports |
| Facebook/CA | NYT/Guardian investigations, FTC, Pew Research trust surveys |
| Google Antitrust | DOJ filings, Pew tech regulation surveys, Morning Consult |
| Super League | UEFA, club withdrawal statements, media coverage |
| Barilla | HRC Corporate Equality Index, media coverage, David Mixner public statements |
| Nike/Kaepernick | Edison Trends, Nike SEC filings (Q1 FY2019), Morning Consult |
| GameStop/WSB | FINRA short interest data, SEC GameStop report (Oct 2021), Robinhood filings |
| Vaccini USA | KFF COVID-19 Vaccine Monitor, Gallup, AP-NORC, Pew Research |
| Smoking Ban UK | YouGov, Ipsos MORI, ASH, British Social Attitudes Survey |

---

## 9. Impatto sul Budget della Simulazione

| Componente | Costo aggiuntivo per run | Note |
|------------|-------------------------|------|
| **Plausibility check** | ~$0.008 (8 round × $0.001) | LLM-as-judge, temp 0.1, max 200 token |
| **Event regeneration** | ~$0.005 (se fallisce ~60% dei check) | Un retry con temp 0.6 |
| **Few-shot nel prompt** | $0.00 | Testo statico, nessun costo aggiuntivo |
| **Seed data nel prompt** | ~$0.01 | Contesto aggiuntivo nel prompt dell'analyzer |
| **Realism evaluation** | ~$0.002 | Un singolo LLM-as-judge call post-simulazione |
| **Totale overhead** | **~$0.025/run** | Su budget $5/run = +0.5% |

Il sistema di calibrazione completo (grid search) richiede ~$15-25, ma è un costo una-tantum per dominio.

---

## 10. File e Codice Prodotto

### Nuovi file (23)

| Directory | File | Linee |
|-----------|------|-------|
| `seed_data/` | `__init__.py` (SeedDataLoader) | 55 |
| `seed_data/` | `schema.py` (Pydantic models) | 81 |
| `seed_data/political/italy_referendum_2026/` | `context.md` | 30 |
| | `historical.md` | 28 |
| | `stakeholders.json` | 150 |
| | `demographics.json` | 72 |
| | `known_events.json` | 30 |
| `core/simulation/` | `validators.py` | 108 |
| `domains/political/examples/` | `elite_round.json` | 25 |
| | `cluster_round.json` | 35 |
| | `event.json` | 15 |
| `calibration/` | `__init__.py` | 1 |
| | `historical_scenario.py` | 20 |
| | `trajectory_comparator.py` | 75 |
| | `parameter_tuner.py` | 82 |
| | 16 × scenario JSON | ~1600 |
| `evaluation/` | `__init__.py` | 1 |
| | `realism_scorer.py` | 155 |
| | `sensitivity.py` | 65 |

### File modificati (10)

| File | Modifica |
|------|----------|
| `core/config/schema.py` | +1 campo (`seed_data_path`) |
| `core/simulation/engine.py` | +seed data loading, +realism evaluation |
| `core/simulation/event_injector.py` | +historical context, +few-shot, +plausibility check, +shock clamping |
| `core/simulation/round_manager.py` | +calibrated params, +few-shot injection |
| `core/simulation/opinion_dynamics.py` | +calibrated params loading |
| `core/agents/elite_agent.py` | +delta cap, +agent ref validation |
| `core/agents/citizen_swarm.py` | +delta cap, +sentiment normalization |
| `briefing/brief_analyzer.py` | +seed_data_bundle parameter |
| `domains/base_domain.py` | +few-shot methods (_load_example, get_*_few_shot) |
| `domains/political/prompts.py` | +constraint instructions in prompts |

### Totale codice

| Categoria | Linee approssimative |
|-----------|---------------------|
| Nuovi moduli Python | ~640 |
| Scenari JSON di calibrazione | ~1,600 |
| Seed data (JSON + markdown) | ~310 |
| Few-shot examples | ~75 |
| Modifiche a file esistenti | ~200 |
| **Totale** | **~2,825 linee** |

---

## 11. Prossimi Passi

### Immediati
1. **Eseguire la prima calibrazione** sul referendum Renzi 2016 (ground truth più completo)
2. **Validare** che gli stakeholder seed producano agenti con nomi e posizioni corrette
3. **Misurare** il realism score prima/dopo il grounding sullo scenario Italia 2026

### A medio termine
4. **Calibrare per dominio**: corporate, marketing, public_health (un grid search ciascuno)
5. **Aggiungere seed data** per altri scenari (Brexit, Dieselgate, Nike)
6. **Sensitivity analysis**: 5 run dello stesso scenario, verificare varianza <15%

### A lungo termine
7. **Calibrazione continua**: ogni nuova simulazione con ground truth disponibile migliora i parametri
8. **Cross-domain transfer**: verificare se i parametri calibrati su un dominio funzionano su un altro
9. **Evaluation dashboard**: visualizzare realism score nel frontend
