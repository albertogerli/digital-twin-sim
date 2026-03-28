"""Engagement metrics, virality scoring, and narrative extraction."""

import math
import re
from collections import Counter
from typing import Optional


class EngagementMetrics:
    """Compute engagement and virality metrics from platform data."""

    def __init__(self, platform_engine):
        self.platform = platform_engine

    def polarization_index(self, positions: list[float]) -> float:
        """Standard deviation of positions, normalized to 0-10 scale."""
        if not positions:
            return 0.0
        n = len(positions)
        mean = sum(positions) / n
        variance = sum((p - mean) ** 2 for p in positions) / n
        std = math.sqrt(variance)
        return min(10.0, std * 10)

    def engagement_curve(self) -> list[dict]:
        """Total interactions per round."""
        rows = self.platform.conn.execute(
            """SELECT round, COUNT(*) as total_reactions
               FROM reactions GROUP BY round ORDER BY round"""
        ).fetchall()
        return [{"round": r["round"], "reactions": r["total_reactions"]} for r in rows]

    def extract_narratives(self, round_num: Optional[int] = None,
                           top_n: int = 5) -> list[str]:
        """Extract top narrative themes using keyword extraction."""
        query = "SELECT content FROM posts WHERE parent_id IS NULL"
        params: list = []
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
        rows = self.platform.conn.execute(query, params).fetchall()

        if not rows:
            return ["no dominant themes"]

        # Simple keyword extraction (language-agnostic stopwords kept minimal)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "we", "our", "you", "your", "he", "she", "his", "her",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "all", "each", "every", "any", "few", "more", "most", "other",
            "some", "such", "than", "too", "very", "just", "also",
        }

        word_counts: Counter = Counter()
        bigram_counts: Counter = Counter()

        for row in rows:
            text = row["content"].lower()
            words = re.findall(r'\b[a-zàèéìòùáéíóúñüö]{3,}\b', text)
            filtered = [w for w in words if w not in stopwords]

            word_counts.update(filtered)

            for i in range(len(filtered) - 1):
                bigram = f"{filtered[i]} {filtered[i+1]}"
                bigram_counts.update([bigram])

        themes = []
        for bigram, _ in bigram_counts.most_common(top_n):
            themes.append(bigram)

        if len(themes) < top_n:
            for word, _ in word_counts.most_common(top_n - len(themes)):
                if word not in " ".join(themes):
                    themes.append(word)

        return themes[:top_n] if themes else ["general discussion"]

    def round_summary(self, round_num: int, all_positions: list[float]) -> dict:
        """Compute summary metrics for a round."""
        stats = self.platform.get_round_stats(round_num)
        top_posts = self.platform.get_top_posts(round_num, top_n=5)
        narratives = self.extract_narratives(round_num, top_n=3)
        pol = self.polarization_index(all_positions)

        return {
            "round": round_num,
            "posts": stats["posts"],
            "reactions": stats["reactions"],
            "polarization": round(pol, 1),
            "top_narratives": narratives,
            "top_posts": [
                {
                    "author": p["author_id"],
                    "content": p["content"][:200],
                    "engagement": p.get("likes", 0) + p.get("reposts", 0) * 2,
                }
                for p in top_posts
            ],
        }
