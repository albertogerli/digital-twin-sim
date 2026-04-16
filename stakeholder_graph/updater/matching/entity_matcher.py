"""Entity Matcher — maps articles to stakeholders via name/alias matching."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stakeholder_graph.schema import Stakeholder
    from stakeholder_graph.updater.sources.rss_source import RawArticle

logger = logging.getLogger(__name__)


class AliasIndex:
    """Pre-built inverted index: alias → set of stakeholder IDs.

    For each stakeholder, generates aliases:
    - Full name lowercased: "giorgia meloni"
    - Surname only (if unique): "meloni"
    - Common title + surname: "presidente meloni", "ministro nordio"
    - Party + surname: "fdi meloni" (only for politicians)
    """

    # Common Italian titles that precede surnames
    TITLE_PREFIXES = [
        "presidente", "ministro", "ministra", "senatore", "senatrice",
        "deputato", "deputata", "sindaco", "sindaca", "governatore",
        "premier", "segretario", "segretaria", "leader", "onorevole",
        "ex presidente", "ex ministro", "ex ministra",
        "prof", "professor", "professoressa", "dott", "dottor", "dottoressa",
        "cardinale", "monsignor", "don", "padre",
        "generale", "ammiraglio",
    ]

    def __init__(self, stakeholders: list[Stakeholder]):
        self._index: dict[str, set[str]] = defaultdict(set)
        self._full_names: dict[str, str] = {}  # id -> full name
        self._surnames: dict[str, list[str]] = defaultdict(list)  # surname -> [ids]
        self._build(stakeholders)

    def _build(self, stakeholders: list[Stakeholder]):
        """Build the alias index from stakeholder list."""
        # First pass: collect all surnames to detect collisions
        for s in stakeholders:
            parts = s.name.lower().split()
            if len(parts) >= 2:
                surname = parts[-1]
                self._surnames[surname].append(s.id)

        # Second pass: build index
        for s in stakeholders:
            name_lower = s.name.lower()
            parts = name_lower.split()
            self._full_names[s.id] = s.name

            # Always index full name
            self._index[name_lower].add(s.id)

            if len(parts) < 2:
                continue

            surname = parts[-1]
            # Handle multi-word surnames like "de luca", "della vedova"
            if len(parts) >= 3 and parts[-2] in ("de", "di", "del", "della", "dello", "van", "von", "el", "al", "la", "le", "lo"):
                surname = f"{parts[-2]} {parts[-1]}"

            # Surname only — if unique (no other stakeholder has same surname)
            if len(self._surnames.get(parts[-1], [])) == 1:
                self._index[surname].add(s.id)

            # Title + surname combinations
            for title in self.TITLE_PREFIXES:
                self._index[f"{title} {surname}"].add(s.id)

            # Party + surname (for politicians)
            if s.party_or_org:
                party_short = s.party_or_org.lower().split("/")[0].strip()
                if party_short and len(party_short) <= 30:
                    self._index[f"{party_short} {surname}"].add(s.id)

        logger.info(f"AliasIndex: {len(self._index)} aliases for {len(self._full_names)} stakeholders")

    def lookup(self, text: str) -> set[str]:
        """Find all stakeholder IDs mentioned in text."""
        text_lower = text.lower()
        found = set()

        # Check full names first (most specific)
        for alias, ids in self._index.items():
            if len(alias) >= 4 and alias in text_lower:  # skip very short aliases
                found.update(ids)

        return found


class EntityMatcher:
    """Matches articles to stakeholders using the AliasIndex."""

    def __init__(self, stakeholders: list[Stakeholder], min_body_mentions: int = 2):
        self.alias_index = AliasIndex(stakeholders)
        self.min_body_mentions = min_body_mentions
        self._stakeholder_names = {s.id: s.name.lower() for s in stakeholders}

    def match(self, articles: list[RawArticle]) -> dict[str, list[RawArticle]]:
        """Match articles to stakeholders.

        Returns dict mapping stakeholder_id -> list of matched articles.

        Matching rules:
        - Name in title → automatic match (high relevance)
        - Name in body ≥ min_body_mentions → match
        """
        matches: dict[str, list[RawArticle]] = defaultdict(list)

        for article in articles:
            # Check title (high weight)
            title_matches = self.alias_index.lookup(article.title)

            # Check body
            body_matches = self.alias_index.lookup(article.body[:1500])

            # Combine: title match is automatic, body needs multiple mentions
            for sid in title_matches:
                if sid not in [a_sid for a_sid in matches if article in matches[a_sid]]:
                    matches[sid].append(article)

            for sid in body_matches - title_matches:
                # Count actual occurrences in body for non-title matches
                name = self._stakeholder_names.get(sid, "")
                if name and article.body.lower().count(name) >= self.min_body_mentions:
                    matches[sid].append(article)
                else:
                    # Also check surname count
                    surname = name.split()[-1] if name and " " in name else ""
                    if surname and len(surname) >= 4 and article.body.lower().count(surname) >= self.min_body_mentions:
                        matches[sid].append(article)

        total_matches = sum(len(v) for v in matches.values())
        logger.info(f"EntityMatcher: {total_matches} matches across {len(matches)} stakeholders from {len(articles)} articles")
        return dict(matches)
