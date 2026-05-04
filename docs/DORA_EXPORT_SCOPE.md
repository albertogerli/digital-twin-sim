# DORA / EBA export — scope decision for MVP

**Status:** Sprint 88, May 2026.
**Decision target:** which regulatory schema to export *first*, given the
business goal in Sella relazione §13.2 (sell to CRO/CISO with budget
compliance illimitato instead of CFO/Strategy with budget tagliato).

## Three candidate schemas

The Digital Operational Resilience Act (Regulation (EU) 2022/2554, in
force from 17 Jan 2025) defines several mandatory reporting obligations
for financial entities. Each maps differently to what DigitalTwinSim
already produces.

### A. DORA Major ICT-related Incident Report (Art. 19, RTS/ITS JC 2024-43)

When a financial entity classifies an ICT-related incident as "major",
it must submit three reports to the competent authority:
- **Initial notification** — within 24h of classification
- **Intermediate report** — within 72h
- **Final report** — within 1 month

The XML schema includes ~40 fields covering:
- Incident metadata (reference number, dates, types)
- Classification criteria scores (clients affected, data losses, reputational
  impact, duration, geographical spread, economic impact, criticality)
- Root cause analysis
- Impact metrics (financial loss, customers affected, data records lost)
- Mitigation actions taken / planned
- Affected business functions and ICT services
- Communications to clients / public

**Maps cleanly to:** the wargame mode (Sella CRO simulates a cyber crisis,
fills in the role of "incident commander", and at the end the system
generates the regulatory report). The simulator already tracks crisis
metrics, customer impact estimates, financial loss projections, and
narrative cause descriptions — all of which are inputs the DORA
template requires.

**Effort:** 2-3 days for the MVP (XML emitter + Pydantic schema +
golden-file tests). Validation against the official XSD requires
downloading the spec from the EBA portal and is a follow-up.

### B. EBA COREP / FINREP-like quarterly supervisory reporting (XBRL)

EBA banking supervisory reporting (FINREP for financial info, COREP
for capital adequacy) uses XBRL with the EBA Data Point Model (DPM).
This is NOT a DORA-specific output but is the existing quarterly
reporting that every Sella ALM analyst already does.

**Maps cleanly to:** the FinancialTwin's CET1/LCR/NIM/RWA evolution
under stress scenarios. The simulator already emits these per round.

**Effort:** 4-6 weeks. The XBRL DPM is *very* large (1000+ data points)
and requires a domain expert to map correctly. For an MVP, exporting
even a single template (e.g. C 03.00 Capital Adequacy) is non-trivial
because XBRL requires a full taxonomy reference, instance documents,
and compliance with the EBA filing rules.

### C. ICT third-party arrangements register (Art. 28, RTS)

Annual XML register of all the bank's contracts with ICT third-party
providers. **Does not map** to DigitalTwinSim's outputs — this is a
contract-management problem, not a simulation problem.

## Decision: ship A (Major Incident Report) first

Rationale:

1. **Maps directly to the wargame use case** Sella is most likely to
   demo. A CRO playing through a cyber-incident wargame with the
   simulator naturally produces the inputs the DORA report needs.
2. **Effort/value ratio**: 2-3 days to MVP vs 4-6 weeks for COREP/FINREP.
   The MVP unlocks the same sales motion (sell to CRO/CISO) at a
   fraction of the cost.
3. **In-force date**: DORA reporting is mandatory NOW (Jan 2025). Banks
   have a real budget line for it this year.
4. **EBA/EIOPA/ESMA joint RTS published**: the schema is fixed (JC
   2024-43, July 2024), so we're not chasing a moving target.

COREP/FINREP (option B) is deferred to v1.0+ as a separate sprint when
a Sella ALM analyst can co-design the field mapping. ICT third-party
register (option C) is out of scope.

## MVP scope (what we ship in Phase 2-3)

- `core/dora/schema.py` — Pydantic model approximating the Annex IV
  template fields. NOT XSD-validated yet; uses the published RTS
  field definitions as ground truth.
- `core/dora/exporter.py` — `build_incident_report(scenario, role,
  classification) -> str` returns well-formed XML.
- `core/dora/classification.py` — DORA classification helper (the bank
  must score the incident on 7 criteria to determine if it's "major").
- `tests/test_dora_exporter.py` — golden-file tests against
  `tests/fixtures/dora_incident_golden.xml`.
- Hook into `core/simulation/reporting.py`: when `domain == "banking"`
  AND `DORA_EXPORT=1` env, emit `outputs/<scenario>_dora_incident.xml`
  alongside the HTML report.
- Update relazione §13.2 from roadmap → implemented.

## Honest scope limits (carried into the documentation)

- **Not yet XSD-validated.** The XML is well-formed and matches the
  published RTS field definitions, but full conformance to the
  official EBA XSD requires downloading the spec from the EBA reporting
  portal (typically gated behind a registered-entity login) and adding
  an `lxml`-based validation step. Marked as next-sprint follow-up.
- **English-only field labels.** Italian bank submits to Banca d'Italia,
  but the wire format is English XML per the RTS — this is correct.
- **Single submission flow.** We emit the "final report" template
  format (most-complete schema). Initial / intermediate reports are
  subsets and can be derived by setting later-availability fields to
  null. Out of scope for MVP.
