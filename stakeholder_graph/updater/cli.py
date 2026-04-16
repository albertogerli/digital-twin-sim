"""CLI entry point for the Graph Updater.

Usage:
    python -m stakeholder_graph.updater                          # Full nightly run
    python -m stakeholder_graph.updater --dry-run                # Analyze without writing
    python -m stakeholder_graph.updater --stakeholder giorgia_meloni  # Single profile
    python -m stakeholder_graph.updater --since 2026-04-10       # Custom window
    python -m stakeholder_graph.updater --report                 # Show last run report
    python -m stakeholder_graph.updater --changelog giorgia_meloni   # Audit trail
    python -m stakeholder_graph.updater --stats                  # DB statistics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("stakeholder_graph.updater")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="stakeholder-graph-updater",
        description="Continuous Integration for the Stakeholder Graph",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze articles and compute updates without writing to disk",
    )
    parser.add_argument(
        "--stakeholder", type=str, default=None,
        help="Only update a specific stakeholder (by ID)",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only process articles since this date (ISO format: 2026-04-10)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Show the last run report and exit",
    )
    parser.add_argument(
        "--changelog", type=str, nargs="?", const="__all__", default=None,
        help="Show changelog for a stakeholder (or all if no ID given)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show database statistics and exit",
    )
    parser.add_argument(
        "--no-google-news", action="store_true",
        help="Disable Google News source (faster, RSS only)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def cmd_report(data_dir: Path):
    """Show the last run report."""
    report_path = data_dir / ".last_report.json"
    if not report_path.exists():
        print("No previous run found.")
        return

    report = json.loads(report_path.read_text())
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         GRAPH UPDATER — LAST RUN REPORT             ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print(f"  Run ID:        {report.get('run_id', 'N/A')}")
    print(f"  Started:       {report.get('started_at', 'N/A')}")
    print(f"  Duration:      {report.get('duration_seconds', 0):.1f}s")
    print()
    print("  ── Fetch ──")
    print(f"  Articles fetched:     {report.get('articles_fetched', 0)}")
    print(f"  After dedup:          {report.get('articles_deduplicated', 0)}")
    print()
    print("  ── Match ──")
    print(f"  Stakeholders matched: {report.get('stakeholders_matched', 0)}")
    print(f"  Total matches:        {report.get('total_matches', 0)}")
    print()
    print("  ── Analyze ──")
    print(f"  LLM calls:            {report.get('llm_calls', 0)}")
    print(f"  Signals extracted:    {report.get('signals_extracted', 0)}")
    print()
    print("  ── Update ──")
    print(f"  Position changes:     {report.get('position_changes', 0)}")
    print(f"  Influence changes:    {report.get('influence_changes', 0)}")
    print(f"  Quotes added:         {report.get('quotes_added', 0)}")
    print(f"  Flagged for review:   {report.get('updates_flagged', 0)}")
    print(f"  Rejected:             {report.get('updates_rejected', 0)}")
    print()
    print(f"  ── Cost ──")
    print(f"  Estimated:            ${report.get('estimated_cost_usd', 0):.4f}")

    if report.get("errors"):
        print("\n  ── Errors ──")
        for e in report["errors"]:
            print(f"  ⚠ {e}")
    print()


def cmd_changelog(data_dir: Path, stakeholder_id: str | None, limit: int = 50):
    """Show changelog entries."""
    from stakeholder_graph.updater.persistence.changelog import Changelog

    cl = Changelog(data_dir / ".changelog.jsonl")
    sid = stakeholder_id if stakeholder_id != "__all__" else None
    entries = cl.query(stakeholder_id=sid, limit=limit)

    if not entries:
        print(f"No changelog entries found{f' for {stakeholder_id}' if sid else ''}.")
        return

    print(f"\n{'─' * 70}")
    print(f" Changelog{f' for {stakeholder_id}' if sid else ''} ({len(entries)} entries)")
    print(f"{'─' * 70}\n")

    for entry in entries:
        etype = entry.get("type", "unknown")
        ts = entry.get("timestamp", "")[:19]
        sid = entry.get("stakeholder_id", "")

        if etype == "position_change":
            old = entry.get("old_value", "?")
            new = entry.get("new_value", "?")
            tag = entry.get("topic_tag", "?")
            delta = entry.get("delta", 0)
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  [{ts}] {sid}.{tag}: {old} {arrow} {new} (Δ{delta:+.4f}, {entry.get('n_signals', 0)} signals)")
        elif etype == "influence_change":
            old = entry.get("old_value", "?")
            new = entry.get("new_value", "?")
            print(f"  [{ts}] {sid} influence: {old} → {new} ({entry.get('reason', '')})")
        elif etype == "quotes_added":
            tag = entry.get("topic_tag", "?")
            n = len(entry.get("quotes", []))
            print(f"  [{ts}] {sid}.{tag}: +{n} quotes")
        elif etype == "run_summary":
            print(f"  [{ts}] === Run {entry.get('run_id', '?')}: {entry.get('position_changes', 0)} changes ===")

    print()


def cmd_stats(data_dir: Path):
    """Show DB statistics."""
    from stakeholder_graph.db import StakeholderDB

    db = StakeholderDB(data_dir)
    stats = db.stats()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║           STAKEHOLDER GRAPH — STATISTICS             ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print(f"  Total stakeholders:  {stats['total']}")
    print(f"  Total positions:     {stats['total_positions']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print()
    print("  ── By Country ──")
    for country, n in sorted(stats["by_country"].items(), key=lambda x: -x[1]):
        print(f"    {country}: {n}")
    print()
    print("  ── By Category ──")
    for cat, n in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")
    print()
    print("  ── By Tier ──")
    for tier, n in sorted(stats["by_tier"].items()):
        label = {1: "Elite", 2: "Institutional", 3: "Minor"}.get(int(tier), f"Tier {tier}")
        print(f"    {label}: {n}")
    print()


async def cmd_run(args, data_dir: Path):
    """Run the update pipeline."""
    from stakeholder_graph.db import StakeholderDB
    from stakeholder_graph.updater.config import UpdaterConfig
    from stakeholder_graph.updater.pipeline import UpdatePipeline

    config = UpdaterConfig(
        dry_run=args.dry_run,
        google_news_enabled=not args.no_google_news,
    )

    db = StakeholderDB(data_dir)

    # Try to initialize LLM client
    llm_client = None
    try:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            from core.llm.gemini_client import GeminiClient
            llm_client = GeminiClient(
                api_key=api_key,
                model=config.llm_model,
            )
            logger.info(f"LLM client initialized: {config.llm_model}")
        else:
            logger.warning("GOOGLE_API_KEY not set — will fetch and match only (no LLM analysis)")
    except ImportError:
        logger.warning("GeminiClient not available — will fetch and match only")

    pipeline = UpdatePipeline(config, db, llm_client)

    # Parse since date
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    # Stakeholder filter
    stakeholder_ids = None
    if args.stakeholder:
        stakeholder_ids = [args.stakeholder]

    report = await pipeline.run(since=since, stakeholder_ids=stakeholder_ids)

    # Print summary
    print(f"\n{'═' * 50}")
    print(f" Run {report.run_id} complete")
    print(f" {report.articles_fetched} articles → {report.stakeholders_matched} stakeholders → {report.position_changes} changes")
    if report.estimated_cost_usd > 0:
        print(f" Cost: ~${report.estimated_cost_usd:.4f}")
    if args.dry_run:
        print(" (DRY RUN — no changes written)")
    print(f"{'═' * 50}\n")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Resolve data dir
    data_dir = Path(__file__).parent.parent / "data"

    if args.report:
        cmd_report(data_dir)
    elif args.changelog is not None:
        sid = args.changelog if args.changelog != "__all__" else None
        cmd_changelog(data_dir, sid)
    elif args.stats:
        cmd_stats(data_dir)
    else:
        asyncio.run(cmd_run(args, data_dir))


if __name__ == "__main__":
    main()
