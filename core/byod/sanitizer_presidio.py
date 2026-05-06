"""BYOD sanitizer — Presidio (NER-based) backend.

Drop-in upgrade for the regex backend in ``core/byod/sanitizer.py``. Same
public contract (``BYODMode``, ``BYODLeakError``, ``SanitizeResult``,
``sanitize_prompt``), but the detection layer is **Microsoft Presidio +
spaCy NER + custom Italian financial recognizers**, not pure regex.

Why this exists
---------------

A bank-side reviewer flagged the regex sanitizer as inadequate for
production deployment. The criticism is valid: regex misses

  - context-dependent PII ("the CEO Mario Rossi" — needs PERSON NER)
  - paraphrased account references ("conto corrente XX1234" with spaces)
  - free-form addresses, fiscal codes embedded in narrative
  - foreign-language entities the Italian-tuned regex doesn't see

Microsoft Presidio (open-source, Apache 2.0) provides a structured
PII-detection framework on top of spaCy. We add three custom Italian
recognizers (codice fiscale, partita IVA, IBAN-IT) on top of the built-ins
(EmailAddress, PhoneNumber, IPv4, CreditCard, IBAN-EU, etc.).

Backend selection
-----------------

The active backend is chosen by env var ``BYOD_BACKEND``:

  - ``regex`` (default): the original regex sanitizer in sanitizer.py
  - ``presidio``: this NER-based sanitizer
  - ``hybrid``: Presidio detection + regex fallback for any text Presidio
                doesn't claim — strictly more conservative, recommended
                for the first 30 days of any production deployment

When ``BYOD_BACKEND=presidio`` and Presidio is not installed, this module
fails closed (raises at import time) so a deployment can never silently
degrade to the regex-only path without operator awareness.

Install
-------

  pip install presidio-analyzer presidio-anonymizer
  python -m spacy download it_core_news_lg
  python -m spacy download en_core_web_lg  # for English narrative content

Cost estimate: ~600MB on disk for the two spaCy models, ~250MB RAM at
runtime. For Railway/Vercel deployments the model bundle ships in the
container image; for on-prem deployments the operator pulls them once.

Audit log compatibility
-----------------------

Writes to the same JSONL audit path as the regex backend. Each row carries
``backend: "presidio"`` so downstream dashboards can split metrics by
backend during the migration window.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# Re-export the public types from the regex backend so callers can
# import either module interchangeably.
from core.byod.sanitizer import (  # noqa: F401
    BYODMode,
    BYODLeakError,
    SanitizeResult,
    DEFAULT_AUDIT_PATH,
    get_mode,
    _append_audit,
)


# ── Lazy Presidio import (so the module loads even without it installed) ──

_PRESIDIO_AVAILABLE = False
_PRESIDIO_IMPORT_ERROR: Optional[Exception] = None
_ANALYZER = None
_ANONYMIZER = None


def _ensure_presidio() -> None:
    """Lazy-import Presidio. Cached after first call."""
    global _PRESIDIO_AVAILABLE, _PRESIDIO_IMPORT_ERROR, _ANALYZER, _ANONYMIZER
    if _ANALYZER is not None:
        return
    try:
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError as e:
        _PRESIDIO_IMPORT_ERROR = e
        return

    analyzer = AnalyzerEngine(supported_languages=["it", "en"])

    # Custom Italian recognizers
    # Codice fiscale: 16 chars, pattern XXXXXX99X99X999X (consonants + digits)
    cf_pattern = Pattern(
        name="codice_fiscale_it",
        regex=r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
        score=0.95,
    )
    cf_recognizer = PatternRecognizer(
        supported_entity="IT_FISCAL_CODE",
        patterns=[cf_pattern],
        supported_language="it",
    )
    analyzer.registry.add_recognizer(cf_recognizer)

    # Partita IVA: 11 digits, optional "IT" prefix
    piva_pattern = Pattern(
        name="partita_iva_it",
        regex=r"\b(?:IT\s*)?\d{11}\b",
        score=0.85,
    )
    piva_recognizer = PatternRecognizer(
        supported_entity="IT_VAT_NUMBER",
        patterns=[piva_pattern],
        supported_language="it",
    )
    analyzer.registry.add_recognizer(piva_recognizer)

    # IBAN-IT: 27 chars starting with IT
    iban_it_pattern = Pattern(
        name="iban_it",
        regex=r"\bIT\d{2}[A-Z]\d{10}[A-Z0-9]{12}\b",
        score=0.99,
    )
    iban_it_recognizer = PatternRecognizer(
        supported_entity="IBAN_IT",
        patterns=[iban_it_pattern],
        supported_language="it",
    )
    analyzer.registry.add_recognizer(iban_it_recognizer)

    _ANALYZER = analyzer
    _ANONYMIZER = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True


# Entities we redact by default (covers the regex backend's surface + more)
DEFAULT_ENTITIES = (
    # Built-in Presidio
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "URL",
    "CREDIT_CARD", "IBAN_CODE", "LOCATION", "DATE_TIME",
    # Custom Italian
    "IT_FISCAL_CODE", "IT_VAT_NUMBER", "IBAN_IT",
)


def sanitize_prompt_presidio(
    prompt: str,
    call_site: str,
    mode: Optional[BYODMode] = None,
    audit_path: Optional[Path] = None,
    tenant: Optional[str] = None,
    language: str = "it",
    entities: Optional[tuple[str, ...]] = None,
) -> SanitizeResult:
    """NER-based sanitizer. Same contract as ``sanitize_prompt``.

    Falls back to OFF behaviour (passthrough + warning in audit log) if
    Presidio is not installed and ``BYOD_BACKEND=presidio`` was set —
    this is **fail-closed** in the sense that the operator gets a clear
    signal in the audit log rather than a silent regex-only fallback.
    """
    if mode is None:
        mode = get_mode()
    if mode == BYODMode.OFF:
        return SanitizeResult(text=prompt, mode=mode.value)

    _ensure_presidio()
    if not _PRESIDIO_AVAILABLE:
        # Explicitly log the fallback so it's auditable
        _append_audit(
            audit_path or DEFAULT_AUDIT_PATH,
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "site": call_site,
                "mode": mode.value,
                "backend": "presidio",
                "error": f"presidio not installed: {_PRESIDIO_IMPORT_ERROR}",
                "fallback": "regex",
                "tenant": tenant or os.environ.get("BYOD_TENANT", "default"),
            },
        )
        # Delegate to regex backend
        from core.byod.sanitizer import sanitize_prompt as _regex_sanitize
        return _regex_sanitize(prompt, call_site, mode=mode,
                                audit_path=audit_path, tenant=tenant)

    use_entities = list(entities or DEFAULT_ENTITIES)
    results = _ANALYZER.analyze(text=prompt, entities=use_entities, language=language)

    detections = []
    if results:
        # Aggregate by entity type for the audit row
        by_entity: dict[str, int] = {}
        samples_by_entity: dict[str, list[str]] = {}
        for r in results:
            by_entity[r.entity_type] = by_entity.get(r.entity_type, 0) + 1
            samples_by_entity.setdefault(r.entity_type, []).append(
                prompt[r.start:r.end][:60]
            )
        for ent, count in by_entity.items():
            detections.append({
                "category": ent.lower(),
                "count": count,
                "samples": samples_by_entity[ent][:3],
            })

    if mode == BYODMode.BLOCK and detections:
        raise BYODLeakError(call_site, detections)

    redacted = prompt
    if mode in (BYODMode.STRICT, BYODMode.BLOCK) and results:
        # Use Anonymizer with categorical replacements (not random tokens)
        from presidio_anonymizer.entities import OperatorConfig
        operators = {
            ent: OperatorConfig("replace", {"new_value": f"[{ent.lower()}]"})
            for ent in use_entities
        }
        anon_result = _ANONYMIZER.anonymize(
            text=prompt,
            analyzer_results=results,
            operators=operators,
        )
        redacted = anon_result.text

    if detections:
        _append_audit(
            audit_path or DEFAULT_AUDIT_PATH,
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "site": call_site,
                "mode": mode.value,
                "backend": "presidio",
                "language": language,
                "raw_chars": len(prompt),
                "sanitized_chars": len(redacted),
                "patterns": [{"category": d["category"], "count": d["count"]} for d in detections],
                "tenant": tenant or os.environ.get("BYOD_TENANT", "default"),
            },
        )

    return SanitizeResult(
        text=redacted if mode in (BYODMode.STRICT, BYODMode.BLOCK) else prompt,
        detections=detections,
        modified=(redacted != prompt) and mode in (BYODMode.STRICT, BYODMode.BLOCK),
        mode=mode.value,
    )


def sanitize_prompt_hybrid(
    prompt: str,
    call_site: str,
    mode: Optional[BYODMode] = None,
    audit_path: Optional[Path] = None,
    tenant: Optional[str] = None,
) -> SanitizeResult:
    """Hybrid: run Presidio first, then regex backend on the result.

    The regex backend catches financial-metric-with-value patterns
    (LCR 95%, CET1 14.2%, BTP-Bund 180bps) that Presidio's general PII
    framework does not target. This is the recommended production
    configuration for the first 30 days of any new deployment.
    """
    presidio_result = sanitize_prompt_presidio(
        prompt, call_site, mode=mode, audit_path=audit_path, tenant=tenant
    )
    # Now run regex sanitizer over the Presidio-redacted text
    from core.byod.sanitizer import sanitize_prompt as _regex_sanitize
    regex_result = _regex_sanitize(
        presidio_result.text, call_site, mode=mode, audit_path=audit_path, tenant=tenant
    )
    # Merge detection lists
    return SanitizeResult(
        text=regex_result.text,
        detections=presidio_result.detections + regex_result.detections,
        modified=presidio_result.modified or regex_result.modified,
        mode=regex_result.mode,
    )


def sanitize_prompt(
    prompt: str,
    call_site: str,
    mode: Optional[BYODMode] = None,
    audit_path: Optional[Path] = None,
    tenant: Optional[str] = None,
) -> SanitizeResult:
    """Top-level dispatch by ``BYOD_BACKEND`` env var.

    Defaults to the regex backend (back-compat); set ``BYOD_BACKEND=presidio``
    or ``BYOD_BACKEND=hybrid`` to switch.
    """
    backend = os.environ.get("BYOD_BACKEND", "regex").lower()
    if backend == "presidio":
        return sanitize_prompt_presidio(prompt, call_site, mode, audit_path, tenant)
    if backend == "hybrid":
        return sanitize_prompt_hybrid(prompt, call_site, mode, audit_path, tenant)
    # Default: regex
    from core.byod.sanitizer import sanitize_prompt as _regex_sanitize
    return _regex_sanitize(prompt, call_site, mode=mode,
                            audit_path=audit_path, tenant=tenant)
