"""DORA Major Incident Report XML exporter.

Renders an :class:`IncidentReport` (see ``core.dora.schema``) into the
XML wire format prescribed by EBA/EIOPA/ESMA Joint Committee Final
Report JC 2024-43 (July 2024) on the technical standards for ICT-related
incident reporting under Regulation (EU) 2022/2554 (DORA), Art. 19-20.

The XML uses simple element/attribute nesting with the namespace
``urn:eu:europa:dora:incident:report:1.0``. We deliberately avoid any
external XML library beyond ``xml.etree.ElementTree`` (stdlib) so the
exporter has zero runtime dependencies and integrates trivially with
the on-prem deployment requirement of the BYOD enclave.

Reverse ingestion (parsing an XML back into an ``IncidentReport``) is
out of scope — the Banca d'Italia / EBA submission flow is one-way.

Usage::

    from core.dora.exporter import build_incident_report

    xml_text = build_incident_report(
        scenario_state=...,   # output of a wargame run
        entity=...,           # FinancialEntity dataclass
        classification=...,   # ClassificationCriteria dataclass
    )
    # xml_text is ready for upload to the bank's regulatory portal
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from xml.etree import ElementTree as ET

from .schema import (
    AffectedFunction,
    ClassificationCriteria,
    FinancialEntity,
    IncidentReport,
    IncidentType,
    MitigationAction,
    ReportType,
    RootCauseCategory,
)

# Namespace for the DORA wire format. Banca d'Italia / EBA submission
# portals expect a namespace; we use a stable URN derived from the RTS.
_NS = "urn:eu:europa:dora:incident:report:1.0"
_NSMAP = {None: _NS}


def _iso(ts: Optional[datetime]) -> str:
    if ts is None:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def _el(parent: ET.Element, tag: str, text: Any = None, **attrs) -> ET.Element:
    """Create a child element on ``parent``; coerce text to str when not None."""
    e = ET.SubElement(parent, f"{{{_NS}}}{tag}", attrib={k: str(v) for k, v in attrs.items()})
    if text is not None:
        e.text = str(text)
    return e


def render_xml(report: IncidentReport, indent: int = 2) -> str:
    """Render an :class:`IncidentReport` to a UTF-8 XML string."""
    root = ET.Element(f"{{{_NS}}}IncidentReport")
    root.set("reportType", report.report_type.value)
    root.set("schemaVersion", "1.0")

    # ── Header ──
    hdr = _el(root, "Header")
    _el(hdr, "ReferenceNumber", report.reference_number)
    _el(hdr, "SubmissionTimestamp", _iso(report.submission_timestamp))
    _el(hdr, "IncidentDetectionTimestamp", _iso(report.incident_detection_timestamp))
    _el(hdr, "IncidentClassificationTimestamp", _iso(report.incident_classification_timestamp))
    if report.incident_resolution_timestamp:
        _el(hdr, "IncidentResolutionTimestamp", _iso(report.incident_resolution_timestamp))

    # ── Entity ──
    ent = _el(root, "Entity")
    _el(ent, "LegalName", report.entity.legal_name)
    _el(ent, "LEI", report.entity.lei_code)
    _el(ent, "CompetentAuthority", report.entity.competent_authority)
    _el(ent, "Country", report.entity.country)

    # ── Classification (the 7 DORA criteria) ──
    cls = _el(root, "Classification")
    cri = report.classification
    _el(cls, "ClientsAffected", cri.clients_affected)
    _el(cls, "DataLosses", cri.data_losses)
    _el(cls, "ReputationalImpact", cri.reputational_impact)
    _el(cls, "DurationDowntimeHours", cri.duration_downtime_hours)
    _el(cls, "GeographicalSpread", cri.geographical_spread)
    _el(cls, "EconomicImpactBand", cri.economic_impact_eur_band)
    _el(cls, "CriticalityOfServicesAffected", cri.criticality_of_services_affected)
    _el(cls, "IsMajor", "true" if cri.is_major() else "false")

    # ── Incident details ──
    det = _el(root, "IncidentDetails")
    _el(det, "IncidentType", report.incident_type.value)
    _el(det, "RootCauseCategory", report.root_cause_category.value)
    _el(det, "RootCauseDescription", report.root_cause_description)

    # ── Affected functions ──
    if report.affected_functions:
        afs = _el(root, "AffectedFunctions")
        for f in report.affected_functions:
            af = _el(afs, "Function")
            _el(af, "Name", f.function_name)
            _el(af, "Type", f.function_type)
            _el(af, "DowntimeMinutes", f.downtime_minutes)
            _el(af, "WorkaroundInPlace", "true" if f.workaround_in_place else "false")

    # ── Impact metrics (intermediate + final reports) ──
    if any(v is not None for v in (
        report.customers_affected_estimate,
        report.data_records_affected_estimate,
        report.economic_impact_eur_estimate,
    )):
        imp = _el(root, "ImpactMetrics")
        if report.customers_affected_estimate is not None:
            _el(imp, "CustomersAffected", int(report.customers_affected_estimate))
        if report.data_records_affected_estimate is not None:
            _el(imp, "DataRecordsAffected", int(report.data_records_affected_estimate))
        if report.economic_impact_eur_estimate is not None:
            _el(imp, "EconomicImpactEUR", round(float(report.economic_impact_eur_estimate), 2))

    # ── Mitigation actions ──
    if report.mitigation_actions:
        mit = _el(root, "Mitigations")
        for a in report.mitigation_actions:
            ma = _el(mit, "Action")
            _el(ma, "Description", a.action)
            _el(ma, "Status", a.status)
            _el(ma, "ResponsibleTeam", a.responsible_team)

    # ── Communications ──
    com = _el(root, "Communications")
    _el(com, "NotifiedClients", "true" if report.notified_clients else "false")
    _el(com, "PublicCommunicationIssued",
        "true" if report.public_communication_issued else "false")
    if report.third_party_providers_notified:
        notif = _el(com, "ThirdPartyProvidersNotified")
        for p in report.third_party_providers_notified:
            _el(notif, "Provider", p)

    # ── Final report extras ──
    if report.report_type == ReportType.FINAL:
        fin = _el(root, "FinalReportSection")
        if report.permanent_remediation_summary:
            _el(fin, "PermanentRemediationSummary", report.permanent_remediation_summary)
        if report.lessons_learned:
            _el(fin, "LessonsLearned", report.lessons_learned)

    # Pretty-print (3.9+ has ET.indent)
    try:
        ET.indent(root, space=" " * indent)
    except AttributeError:
        pass
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def build_incident_report(
    *,
    reference_number: str,
    report_type: ReportType,
    entity: FinancialEntity,
    classification: ClassificationCriteria,
    incident_type: IncidentType,
    root_cause_category: RootCauseCategory,
    root_cause_description: str,
    detected_at: datetime,
    classified_at: datetime,
    affected_functions: Optional[list[AffectedFunction]] = None,
    mitigation_actions: Optional[list[MitigationAction]] = None,
    customers_affected: Optional[int] = None,
    data_records_affected: Optional[int] = None,
    economic_impact_eur: Optional[float] = None,
    notified_clients: bool = False,
    public_communication_issued: bool = False,
    third_party_providers_notified: Optional[list[str]] = None,
    resolved_at: Optional[datetime] = None,
    permanent_remediation_summary: Optional[str] = None,
    lessons_learned: Optional[str] = None,
    submitted_at: Optional[datetime] = None,
) -> str:
    """High-level convenience builder. Returns the XML wire format string.

    For the simulation integration path see
    ``core/dora/from_scenario.py:scenario_to_incident_report`` which
    pulls the inputs from a wargame scenario state automatically.
    """
    report = IncidentReport(
        reference_number=reference_number,
        report_type=report_type,
        submission_timestamp=submitted_at or datetime.now(timezone.utc),
        incident_detection_timestamp=detected_at,
        incident_classification_timestamp=classified_at,
        entity=entity,
        classification=classification,
        incident_type=incident_type,
        root_cause_category=root_cause_category,
        root_cause_description=root_cause_description,
        affected_functions=affected_functions or [],
        mitigation_actions=mitigation_actions or [],
        customers_affected_estimate=customers_affected,
        data_records_affected_estimate=data_records_affected,
        economic_impact_eur_estimate=economic_impact_eur,
        notified_clients=notified_clients,
        public_communication_issued=public_communication_issued,
        third_party_providers_notified=third_party_providers_notified or [],
        incident_resolution_timestamp=resolved_at,
        permanent_remediation_summary=permanent_remediation_summary,
        lessons_learned=lessons_learned,
    )
    return render_xml(report)
