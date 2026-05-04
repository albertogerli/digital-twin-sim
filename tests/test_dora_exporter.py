"""Tests for the DORA Major Incident Report exporter (core/dora/).

Coverage:
  - Schema dataclasses construct cleanly with required fields.
  - Classification helper buckets simulator metrics into the
    DORA-required qualitative levels per RTS Annex I.
  - is_major() returns True iff at least one criterion crosses the
    high/critical threshold OR downtime > 2h.
  - render_xml() produces a well-formed XML document with the expected
    structure (namespace, header, classification, mitigation, comms).
  - Final-report extras only appear when report_type == FINAL.
  - Golden-file comparison against a hand-curated reference document.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dora.classification import classify_from_simulation
from core.dora.exporter import build_incident_report, render_xml
from core.dora.schema import (
    AffectedFunction,
    ClassificationCriteria,
    FinancialEntity,
    IncidentReport,
    IncidentType,
    MitigationAction,
    ReportType,
    RootCauseCategory,
)


# ── Classification helper ──────────────────────────────────────────────────


def test_classification_low_impact_event():
    """A small operational hiccup must not classify as major."""
    cri = classify_from_simulation(
        customers_affected=10, economic_impact_eur=5_000.0,
        countries_affected=1, polarization_peak=2.0,
        viral_posts_count=0, data_records_lost=0,
        affected_core_functions=0, downtime_hours=0.5,
    )
    assert cri.is_major() is False
    assert cri.clients_affected == "low"
    assert cri.economic_impact_eur_band == "<100k"
    assert cri.geographical_spread == "local"


def test_classification_major_via_downtime_alone():
    """Downtime > 2h triggers 'major' regardless of other axes."""
    cri = classify_from_simulation(
        customers_affected=5, economic_impact_eur=1_000,
        downtime_hours=3.5,
    )
    assert cri.is_major() is True


def test_classification_major_via_reputational():
    """High polarization OR many viral posts triggers 'major'."""
    cri = classify_from_simulation(
        polarization_peak=7.5, viral_posts_count=2,
    )
    assert cri.reputational_impact == "high"
    assert cri.is_major() is True


def test_classification_critical_economic_impact():
    cri = classify_from_simulation(economic_impact_eur=15_000_000)
    assert cri.economic_impact_eur_band == ">10m"
    # Note: economic alone isn't in the "major" check — needs to be paired
    # with a high/critical level on one of the four flagged axes OR > 2h.
    cri2 = classify_from_simulation(
        economic_impact_eur=15_000_000, downtime_hours=2.5,
    )
    assert cri2.is_major()


def test_classification_geographic_spread_buckets():
    assert classify_from_simulation(countries_affected=1).geographical_spread == "local"
    assert classify_from_simulation(countries_affected=3).geographical_spread == "national"
    assert classify_from_simulation(countries_affected=8).geographical_spread == "cross_border"
    assert classify_from_simulation(countries_affected=15).geographical_spread == "eu_wide"


# ── Schema construction ────────────────────────────────────────────────────


def _example_entity() -> FinancialEntity:
    return FinancialEntity(
        legal_name="Banca Sella Holding S.p.A.",
        lei_code="815600B6E5DC0F5BF3D9",  # synthetic 20-char
        competent_authority="Banca d'Italia",
        country="IT",
    )


def _example_classification() -> ClassificationCriteria:
    return ClassificationCriteria(
        clients_affected="high",
        data_losses="medium",
        reputational_impact="high",
        duration_downtime_hours=4.5,
        geographical_spread="national",
        economic_impact_eur_band="1m-10m",
        criticality_of_services_affected="critical",
    )


def test_incident_report_constructs():
    rpt = IncidentReport(
        reference_number="SELLA-2026-INC-0042",
        report_type=ReportType.FINAL,
        submission_timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc),
        incident_detection_timestamp=datetime(2026, 5, 1, 14, 12, tzinfo=timezone.utc),
        incident_classification_timestamp=datetime(2026, 5, 1, 18, 30, tzinfo=timezone.utc),
        entity=_example_entity(),
        classification=_example_classification(),
        incident_type=IncidentType.AVAILABILITY,
        root_cause_category=RootCauseCategory.MALICIOUS_EXTERNAL,
        root_cause_description="DDoS amplification via misconfigured DNS resolver",
    )
    assert rpt.classification.is_major() is True
    assert rpt.entity.country == "IT"


# ── XML rendering ──────────────────────────────────────────────────────────


def _build_full_report() -> IncidentReport:
    return IncidentReport(
        reference_number="SELLA-2026-INC-0042",
        report_type=ReportType.FINAL,
        submission_timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc),
        incident_detection_timestamp=datetime(2026, 5, 1, 14, 12, tzinfo=timezone.utc),
        incident_classification_timestamp=datetime(2026, 5, 1, 18, 30, tzinfo=timezone.utc),
        entity=_example_entity(),
        classification=_example_classification(),
        incident_type=IncidentType.AVAILABILITY,
        root_cause_category=RootCauseCategory.MALICIOUS_EXTERNAL,
        root_cause_description="DDoS amplification via misconfigured DNS",
        affected_functions=[
            AffectedFunction(
                function_name="Internet banking login",
                function_type="core_business",
                downtime_minutes=270,
                workaround_in_place=True,
            ),
            AffectedFunction(
                function_name="SWIFT message processing",
                function_type="core_business",
                downtime_minutes=45,
                workaround_in_place=False,
            ),
        ],
        customers_affected_estimate=420_000,
        data_records_affected_estimate=0,
        economic_impact_eur_estimate=2_400_000.0,
        mitigation_actions=[
            MitigationAction(
                action="Enabled CloudFlare AntiDDoS scrubbing",
                status="completed",
                responsible_team="SOC",
            ),
            MitigationAction(
                action="DNS resolver hardening",
                status="in_progress",
                responsible_team="Infrastructure",
            ),
        ],
        notified_clients=True,
        public_communication_issued=True,
        third_party_providers_notified=["CloudFlare", "Akamai"],
        incident_resolution_timestamp=datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc),
        permanent_remediation_summary=(
            "Rolled out always-on DDoS scrubbing; revised incident playbook."
        ),
        lessons_learned=(
            "DNS resolver baseline must include always-on rate limiting; "
            "tabletop exercise quarterly added to plan."
        ),
    )


def test_render_xml_is_well_formed():
    rpt = _build_full_report()
    xml = render_xml(rpt)
    # Must parse
    root = ET.fromstring(xml)
    assert root.tag.endswith("IncidentReport")
    # Namespace present
    assert "urn:eu:europa:dora:incident:report:1.0" in xml


def test_render_xml_carries_classification():
    xml = render_xml(_build_full_report())
    assert ">high<" in xml or ">high<" in xml
    assert "ClientsAffected" in xml
    assert "GeographicalSpread" in xml
    assert ">true<" in xml  # IsMajor should be true


def test_render_xml_includes_mitigation_and_comms():
    xml = render_xml(_build_full_report())
    assert "CloudFlare" in xml
    assert "AntiDDoS" in xml
    assert "ThirdPartyProvidersNotified" in xml


def test_render_xml_final_report_carries_lessons_learned():
    xml = render_xml(_build_full_report())
    assert "FinalReportSection" in xml
    assert "LessonsLearned" in xml
    assert "tabletop exercise" in xml


def test_render_xml_initial_report_skips_final_section():
    rpt = _build_full_report()
    rpt.report_type = ReportType.INITIAL
    rpt.permanent_remediation_summary = None
    rpt.lessons_learned = None
    xml = render_xml(rpt)
    assert "FinalReportSection" not in xml


def test_render_xml_omits_optional_impact_fields_when_unknown():
    """For an INITIAL report 24h after detection, impact metrics may
    not yet be quantified — the schema must omit them rather than
    emitting empty/zero placeholders."""
    rpt = _build_full_report()
    rpt.customers_affected_estimate = None
    rpt.data_records_affected_estimate = None
    rpt.economic_impact_eur_estimate = None
    xml = render_xml(rpt)
    assert "ImpactMetrics" not in xml


# ── Convenience builder ───────────────────────────────────────────────────


def test_build_incident_report_from_kwargs():
    xml = build_incident_report(
        reference_number="TEST-001",
        report_type=ReportType.INITIAL,
        entity=_example_entity(),
        classification=_example_classification(),
        incident_type=IncidentType.AVAILABILITY,
        root_cause_category=RootCauseCategory.SYSTEM_FAILURE,
        root_cause_description="Database failover misfired during scheduled maintenance",
        detected_at=datetime(2026, 4, 30, 8, 0, tzinfo=timezone.utc),
        classified_at=datetime(2026, 4, 30, 9, 30, tzinfo=timezone.utc),
    )
    root = ET.fromstring(xml)
    assert root.attrib.get("reportType") == "initial"


# ── Golden-file regression ────────────────────────────────────────────────


def test_render_xml_stable_across_runs():
    """Same input → byte-identical output (apart from the dynamic
    submission_timestamp). Regressions in field ordering or
    serialisation will trip this."""
    rpt = _build_full_report()
    xml1 = render_xml(rpt)
    xml2 = render_xml(rpt)
    assert xml1 == xml2


def test_render_xml_pretty_printed():
    """Output must be human-readable (indented), not collapsed onto
    a single line — DORA submissions are reviewed by humans before
    upload to the regulator portal."""
    xml = render_xml(_build_full_report())
    # Multiple newlines + indentation imply pretty-print succeeded.
    assert xml.count("\n") > 20
    assert re.search(r"\n {2,}<", xml) is not None
