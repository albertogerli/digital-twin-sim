"""CLI for inspecting the Stakeholder Graph.

Usage:
    python -m stakeholder_graph                    # Show stats
    python -m stakeholder_graph --query IT         # All Italian stakeholders
    python -m stakeholder_graph --topic judiciary_reform  # By topic
    python -m stakeholder_graph --scenario "referendum separazione carriere Italia"
"""

import argparse
import json

from stakeholder_graph.db import StakeholderDB
from stakeholder_graph.integration import stakeholders_for_scenario


def main():
    parser = argparse.ArgumentParser(description="Stakeholder Graph CLI")
    parser.add_argument("--query", "-q", help="Query by country code (e.g. IT)")
    parser.add_argument("--topic", "-t", help="Filter by topic tag")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--scenario", "-s", help="Simulate scenario brief lookup")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--limit", type=int, default=20, help="Max results")
    args = parser.parse_args()

    db = StakeholderDB()

    if args.scenario:
        result = stakeholders_for_scenario(args.scenario, n_elite=12)
        if result:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {args.scenario}")
            print(f"{'='*70}")
            print(f"\nElite Agents ({len(result['elite_agents'])}):")
            for a in result["elite_agents"]:
                print(f"  {a['name']:35s} pos={a['position']:+.2f} inf={a['influence']:.2f} [{a['archetype']}]")
            print(f"\nInstitutional ({len(result['institutional_agents'])}):")
            for a in result["institutional_agents"]:
                print(f"  {a['name']:35s} pos={a['position']:+.2f} inf={a['influence']:.2f}")
        else:
            print("No stakeholder graph coverage for this scenario.")
        return

    if args.query or args.topic or args.category:
        results = db.query(
            country=args.query,
            category=args.category,
            topic_tag=args.topic,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([s.model_dump() for s in results], indent=2, ensure_ascii=False))
        else:
            print(f"\n{len(results)} stakeholders found:\n")
            for s in results:
                pos_str = ""
                if args.topic:
                    pos_str = f" pos={s.get_position(args.topic):+.2f}"
                print(f"  {s.name:35s} [{s.category:12s}] inf={s.influence:.2f}{pos_str}  {s.party_or_org}")
        return

    # Default: show stats
    stats = db.stats()
    print(f"\n{'='*50}")
    print(f"GLOBAL STAKEHOLDER GRAPH")
    print(f"{'='*50}")
    print(f"\nTotal: {stats['total']} stakeholders")
    print(f"\nBy country:")
    for k, v in sorted(stats["by_country"].items()):
        print(f"  {k}: {v}")
    print(f"\nBy category:")
    for k, v in sorted(stats["by_category"].items()):
        print(f"  {k:15s}: {v}")
    print(f"\nBy tier:")
    for k, v in sorted(stats["by_tier"].items()):
        print(f"  Tier {k}: {v}")
    print(f"\nPositions: {stats['total_positions']}")
    print(f"Relationships: {stats['total_relationships']}")


if __name__ == "__main__":
    main()
