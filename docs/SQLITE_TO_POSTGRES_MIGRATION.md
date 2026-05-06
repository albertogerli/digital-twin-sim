# SQLite → Postgres Migration Plan

**Status**: planned, not yet executed.
**Trigger**: any enterprise pilot (Banca Sella, Sunrise, Accenture client engagement) where the customer's CISO requires production-grade RDBMS.
**Owner**: Alberto Gerli
**Last updated**: 2026-05-06

---

## Why this matters

SQLite is the current default storage for several pieces of state:

| Path | Purpose | Today | Cloud risk |
|---|---|---|---|
| `outputs/social_*.db` | per-sim social-media artefacts (posts, engagement) | SQLite (WAL mode, file-per-sim) | Single-writer; lost on container restart unless persistent volume; no backup |
| `data/calibration_registry.db` | continuous-calibration shadow predictions + drift log | SQLite via `core/calibration/continuous.py` | Same |
| `data/sessions.db` | invite-token + auth sessions | SQLite | Same; auth state lost on restart is a hard outage |
| `data/audit_byod.db` | BYOD redaction audit trail | SQLite | **Compliance-critical** — losing audit log on container restart is a DORA/GDPR violation |
| `data/admin_jobs.db` | admin-jobs registry (DORA refit, stakeholder updates) | SQLite | Cross-instance state inconsistency on Vercel/Railway scale-out |

For pilot/POC on Railway with a single instance + persistent volume, this is **acceptable with explicit disclosure** in the SOW. For a production deployment at a regulated bank (Sella) or telco (Sunrise) it is **not acceptable** — the procurement security questionnaire will flag it on day one.

## Target architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Customer cloud / our managed cloud                              │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │ FastAPI app │──▶│ pgBouncer   │──▶│ Postgres 16          │    │
│  │ (Railway)   │   │ (connection │   │ • multi-AZ replica   │    │
│  └─────────────┘   │  pooling)   │   │ • PITR (7 days)      │    │
│                    └─────────────┘   │ • TDE at rest (KMS)  │    │
│                                      │ • TLS 1.3 in transit │    │
│                                      │ • pgaudit extension  │    │
│                                      └─────────────────────┘    │
│                                              │                   │
│                                              ▼                   │
│                                      ┌─────────────────────┐    │
│                                      │ Encrypted backups   │    │
│                                      │ (S3 + lifecycle)    │    │
│                                      └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Hosting options ranked by enterprise-readiness

| Provider | Pricing (SOC2 tier) | Multi-AZ | PITR | pgaudit | Notes |
|---|---|---|---|---|---|
| **AWS RDS Aurora Postgres** | from €50/mo, prod tier ~€500-1500/mo | ✓ | 35d | ✓ | The default for any EU bank; HSM via CloudHSM |
| **Neon serverless Postgres** | from €30/mo, scale-to-zero | ✓ | 7d (paid) | manual | Best DX; check SOC2 tier matches customer policy |
| **Supabase Pro** | from €25/mo | ✓ | 7d | ✓ | Good for B2B SaaS; not common in IT banks |
| **Railway Postgres** | from €5/mo | manual | manual | manual | Pilot-grade only; not enterprise |
| **On-prem customer Postgres** | n/a | depends | depends | depends | Required for some banks; need ops runbook |

For Banca Sella + Sunrise specifically: **Aurora Postgres in eu-west-1 / eu-central-1**. Both have AWS already approved internally.

## Migration steps (per database)

The migration is incremental: one SQLite db → one Postgres schema/role pair, gated on a feature flag. We never migrate all at once.

### 1. Prep work (one-time, ~1 day)

- [ ] Add `psycopg[binary]` and `sqlalchemy>=2.0` to `requirements.txt`.
- [ ] Add `DATABASE_URL` env var (postgres://user:pass@host:5432/dbname) read by a new `core/storage/db.py` factory.
- [ ] Wrap each existing `sqlite3.connect(...)` site in a thin `get_connection(db_name)` that returns either a SQLite connection (DEV) or a Postgres connection (when `DATABASE_URL` is set).
- [ ] Add `core/storage/migrations/` directory and adopt **Alembic** for schema versioning.

### 2. Per-database migration (each ~½ day)

For each of the 5 databases above, in order of compliance criticality (audit_byod first, then sessions, then admin_jobs, then calibration_registry, then social_*):

- [ ] **Schema port**: write the equivalent CREATE TABLE in Postgres dialect (mostly identical except `INTEGER PRIMARY KEY AUTOINCREMENT` → `BIGSERIAL`/`IDENTITY`, `DATETIME` → `TIMESTAMPTZ`, `BLOB` → `BYTEA`).
- [ ] **Data port**: one-shot Python script using `pandas.read_sql` from SQLite → `df.to_sql` to Postgres for any rows worth preserving.
- [ ] **Code port**: replace SQLite-specific SQL (e.g. `INSERT OR REPLACE`) with Postgres equivalent (`INSERT ... ON CONFLICT (key) DO UPDATE`).
- [ ] **Tests**: re-run the relevant test suite against Postgres (use `pytest-postgresql` for an in-process Postgres in CI).
- [ ] **Audit log**: enable pgaudit extension and configure to log all DML on the `audit_byod` schema.

### 3. Production cutover (per pilot deployment)

- [ ] Set `DATABASE_URL` in Railway/customer env.
- [ ] Run Alembic migrations: `alembic upgrade head`.
- [ ] Smoke-test all `/api/admin/*` endpoints + `/api/compliance/dora/*` endpoints.
- [ ] Verify pgaudit log captures redaction events (BYOD STRICT mode).
- [ ] Verify backup ran (snapshot in S3).
- [ ] Verify PITR works by restoring a test backup to a staging DB.
- [ ] Update SOW: customer confirms acceptance of Postgres-backed architecture.

### 4. Post-migration cleanup

- [ ] Remove SQLite fallback from `core/storage/db.py` for any database that has been migrated.
- [ ] Update `docs/SECURITY_AND_COMPLIANCE_BYOD.md` to remove "SQLite caveat" disclosure.
- [ ] Update vendor security questionnaire responses.

## Estimated effort

| Phase | Effort | Calendar time |
|---|---|---|
| Prep (Alembic, factory, env wiring) | 1 day | 1 day |
| Per-db migration × 5 | 0.5 day each | 1 week |
| Production cutover (Railway pilot) | 0.5 day | 0.5 day |
| Production cutover (Sella/Sunrise) | 1 day each | depends on customer ops cadence |
| **Total dev** | **~5 dev-days** | **2-3 weeks elapsed** |

Plus ~€15-25K one-time for SOC2 Type 1 readiness audit if the customer requires it (Drata or Vanta — €1-1.5K/mo subscription + audit fee). This is an investment that unlocks enterprise sales generally, not just this migration.

## What's explicitly NOT in scope here

- **Multi-region replication** beyond AWS multi-AZ (out of scope unless a customer demands EU-only data residency).
- **HSM-backed key management** (CloudHSM) — we currently rely on AWS-managed KMS keys; HSM is a year-2 hardening if requested.
- **Read replicas for analytics** — we're not at the volume where this matters.
- **Sharding** — N=40 incidents, 50 sims/day pilot scale fits comfortably on a single Postgres instance.

## Why this is in `docs/` and not implemented yet

Because **none of the current pilot conversations require it before signing**. Doing this work pre-emptively against a deal that hasn't closed would burn ~5 dev-days that are better spent on the actual product. The trigger is unambiguous: when the first procurement questionnaire from Sella or Sunrise lands, **page 1, question 1** will be "what RDBMS does the system use, where is it hosted, and what's the backup/PITR strategy?". The honest answer at that point becomes "Postgres on Aurora, multi-AZ, 7-day PITR — here's the runbook" (this file), not "SQLite for now, we'll migrate after pilot".

The document exists so that the answer is ready before the question.
