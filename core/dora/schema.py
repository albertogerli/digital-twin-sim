"""DORA Major ICT-related Incident Report — Pydantic schema.

Approximates the field definitions published in EBA/EIOPA/ESMA Joint
Committee Final Report JC 2024-43 (July 2024) on the technical
standards for incident reporting under Regulation (EU) 2022/2554
(DORA), Art. 19 and 20.

The XML emitted by ``core.dora.exporter.build_incident_report`` follows
this structure. The schema is NOT yet XSD-validated against the
official spec — that requires the EBA reporting portal download — but
the field names, types, and enumerations match the RTS Annex IV
templates as published.

Usage::

    from core.dora.schema import IncidentReport, ClassificationCriteria

    report = IncidentReport(
        reference_number="SELLA-2026-INC-0042",
        report_type="final",
        ...
    )
    xml = report.to_xml()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ReportType(str, Enum):
    """DORA Art. 19 mandates three sequential reports per major incident."""
    INITIAL = "initial"          # within 24h of classification
    INTERMEDIATE = "intermediate"  # within 72h
    FINAL = "final"              # within 1 month


class IncidentType(str, Enum):
    """Categorisation per RTS Annex II (high-level)."""
    AVAILABILITY = "availability_loss"      # service outage
    INTEGRITY = "data_integrity_loss"        # data corruption
    CONFIDENTIALITY = "data_confidentiality_breach"  # leak / unauthorised access
    AUTHENTICITY = "authenticity_failure"    # impersonation / fraud
    OTHER = "other"


class RootCauseCategory(str, Enum):
    """RTS Annex III root-cause taxonomy."""
    MALICIOUS_EXTERNAL = "malicious_external_actor"
    MALICIOUS_INTERNAL = "malicious_internal_actor"
    SYSTEM_FAILURE = "system_failure"
    PROCESS_FAILURE = "process_failure"
    HUMAN_ERROR = "human_error"
    EXTERNAL_EVENT = "external_event"  # e.g. natural disaster
    THIRD_PARTY_FAILURE = "third_party_provider_failure"


@dataclass
class ClassificationCriteria:
    """The 7 criteria DORA uses to determine if an incident is 'major'.

    A score >= threshold on any single criterion makes the incident
    reportable. See RTS Annex I for full thresholds; we keep the
    qualitative levels (low / medium / high / critical) here for
    auditability without leaking precise internal values.
    """
    clients_affected: str  # "low" / "medium" / "high" / "critical"
    data_losses: str
    reputational_impact: str
    duration_downtime_hours: float
    geographical_spread: str  # "local" / "national" / "cross_border" / "eu_wide"
    economic_impact_eur_band: str  # "<100k" / "100k-1m" / "1m-10m" / ">10m"
    criticality_of_services_affected: str  # "non_critical" / "critical" / "vital"

    def is_major(self) -> bool:
        """Coarse 'is major' check — any criterion at high or critical OR
        downtime > 2h triggers reporting under DORA classification rules."""
        major_levels = {"high", "critical"}
        if self.duration_downtime_hours > 2.0:
            return True
        return any(
            getattr(self, f) in major_levels
            for f in ("clients_affected", "data_losses", "reputational_impact",
                      "criticality_of_services_affected")
        )


@dataclass
class FinancialEntity:
    """Reporting entity identification (LEI is the global standard)."""
    legal_name: str
    lei_code: str  # 20-char Legal Entity Identifier
    competent_authority: str  # e.g. "Banca d'Italia"
    country: str  # ISO-2


@dataclass
class AffectedFunction:
    """RTS Annex IV: business functions / ICT services impacted."""
    function_name: str
    function_type: str  # "core_business" / "support" / "ict_service"
    downtime_minutes: int
    workaround_in_place: bool


@dataclass
class MitigationAction:
    """Action taken or planned in response to the incident."""
    action: str
    status: str  # "completed" / "in_progress" / "planned"
    responsible_team: str


@dataclass
class IncidentReport:
    """Top-level DORA Major Incident Report — Annex IV final-report shape."""
    # ── Header (all reports) ──
    reference_number: str
    report_type: ReportType
    submission_timestamp: datetime
    incident_detection_timestamp: datetime
    incident_classification_timestamp: datetime

    entity: FinancialEntity
    classification: ClassificationCriteria

    # ── Incident details ──
    incident_type: IncidentType
    root_cause_category: RootCauseCategory
    root_cause_description: str  # narrative, max ~2000 chars

    affected_functions: list[AffectedFunction] = field(default_factory=list)

    # ── Impact metrics (filled progressively) ──
    customers_affected_estimate: Optional[int] = None
    data_records_affected_estimate: Optional[int] = None
    economic_impact_eur_estimate: Optional[float] = None  # for INTERMEDIATE+

    # ── Mitigation ──
    mitigation_actions: list[MitigationAction] = field(default_factory=list)

    # ── Communications ──
    notified_clients: bool = False
    public_communication_issued: bool = False
    third_party_providers_notified: list[str] = field(default_factory=list)

    # ── Final report only ──
    incident_resolution_timestamp: Optional[datetime] = None
    permanent_remediation_summary: Optional[str] = None
    lessons_learned: Optional[str] = None

    def to_xml(self, indent: int = 2) -> str:
        """Emit the XML wire format. See exporter.py for the actual
        rendering — this method delegates."""
        from .exporter import render_xml
        return render_xml(self, indent=indent)
