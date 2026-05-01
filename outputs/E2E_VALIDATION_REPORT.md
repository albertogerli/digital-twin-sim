# E2E Validation Report — Sprint 1-13 + paper scenarios

**Data**: 2026-05-01 (in corso)
**Sim totali eseguite**: 13 (3 custom + 10 paper)
**Costo totale**: ~$2.34 LLM
**Fallimenti pipeline**: 0/13

## Setup

3 sim custom mirate sui bug noti + 10 scenari del paper (mix per coverage dominio):
- **FIN**: GameStop 2021, FTX 2022, SVB 2023
- **POL**: Brexit 2016, Referendum costituzionale IT 2016, USA Presidential 2020
- **CORP**: Dieselgate 2015, Boeing 737 MAX 2019
- **PH**: COVID vax IT 2021
- **TECH**: GDPR 2018

3 sim custom: Banca Sella Q2 (banking IT), JPMorgan auto loans (banking US), Foreverland cacao (food IT).

## Risultati di validazione

### ✅ Bug fix che TENGONO (regression-free)

| Bug originale | Sim test | Esito |
|---|---|---|
| Biden/Trump in sim Banca Sella | 89bafead (28 agenti) | **0 US politicians** ✓ |
| Mattarella/Nordio in sim cacao | 712946a4 (22 agenti) | **0 IT top political** ✓ |
| 3-tier in agents.json | 89bafead, 712946a4 | T1=13/8, T2=9/9, T3=6/5 ✓ |
| Pipeline crash 'bool no predict' | 13/13 sim | **0 crash** ✓ |
| Empty `agents.json` | Tutti gli scenari | populated ✓ |
| Financial twin attivo per banking | GameStop/FTX/SVB/Sella/JPM | **ALM=True** ✓ |
| Financial twin spento per non-banking | Brexit/Cacao/Dieselgate ecc. | **ALM=False** ✓ |

### ❌ Bug critico TROVATO

**False negative su brief politici espliciti**

Scenari affetti:
- POL-2020-USA Presidential (sim 55873633): **0 candidati USA** (Trump, Biden, Pence, Harris tutti droppati)
- POL-2016-Brexit (sim 7790237c): **0 leader UK** (Cameron, May, Johnson, Farage tutti droppati)
- POL-2016-Referendum IT (sim d2adc511): da verificare (ha 34 agenti)

**Causa**: lo Sprint 11b in-country head-of-state penalty si attiva quando `_is_political_brief()` ritorna False. La detection si basava SOLO sui prefix di `scope.sector` (`politics`, `policy`, ecc.). Ma il LLM scope detector spesso ritorna sector specifici (es. `us_politics_2020`, `uk_referendum`) NON in lista → penalty erroneamente applicata → top political figures droppati anche su brief politici espliciti.

**Fix applicato (commit `ba2322d`)**:
1. `_is_political_brief()` ora accetta brief_lower e fa keyword fallback (`presidential election`, `presidenziali`, `referendum`, `primary 20`, ecc.). Il brief stesso decide.
2. `_SECTOR_TO_TOPIC_TAGS` esteso con `us_politics`, `us_politics_2020/2024/2028`, `uk_politics`, `us_elections`, `elections`.
3. Nuovo regression test: `test_election_brief_no_explicit_names_still_keeps_top_candidates`.

**Validazione fix**: in corso (re-launch sim USA Pres 2020 + Brexit dopo deploy `ba2322d`).

## Costi e performance

| Metrica | Valore | Note |
|---|---|---|
| Tempo totale 10 paper sim | ~16 min | 3 in parallelo |
| Tempo medio per sim | ~3-4 min | (paper: 6 round; custom: 9 round) |
| Costo medio per sim | $0.18 | budget cap $1.5 mai raggiunto |
| Throughput Railway | 3 sim concurrent stabile | DTS_MAX_CONCURRENT=4 |
| Failure rate | 0% | 13/13 completed |

## Open issues

1. **Validazione fix election brief**: in corso, re-launch attivo
2. **Reasoning trace presence**: not surfaced nel test (solo verifica di presenza nel JSON, non rendering)
3. **Restanti 32 scenari paper**: non eseguiti per ora (~$6 + 5h runtime)
4. **Stakeholder DB coverage**: alcuni scenari paper (es. Greek bailout 2015, Catalan independence) potrebbero non avere stakeholder nel DB
5. **Duplicate IDs warning** (3 stakeholders): cosmetico, non blocca pipeline

## Prossimi passi

- ✅ Re-launch 2 critical political sims dopo deploy `ba2322d`
- ⏳ Validate fix tiene
- ⏳ Considera batch completo 44 scenari (4-6 ore in background)
- ⏳ Sprint 14: HTML rendering reasoning trace

---
*Report generato durante E2E validation pipeline post-Sprint 1-13.*
