"""Agent memory — tracks round summaries, posts, alliances across rounds."""

from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    """Per-round memory for an agent."""
    round_summaries: list[str] = field(default_factory=list)
    posts_history: list[dict] = field(default_factory=list)
    engagement_history: list[dict] = field(default_factory=list)
    alliances: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def add_round(self, round_num: int, summary: str, posts: list[dict],
                  engagement: dict, alliances: list[str] = None,
                  targets: list[str] = None):
        self.round_summaries.append(f"[Round {round_num}] {summary}")
        self.posts_history.extend(posts)
        self.engagement_history.append(engagement)
        if alliances:
            self.alliances = alliances
        if targets:
            self.targets = targets

    def get_context(self, last_n: int = 3,
                    memory_strings: dict | None = None) -> str:
        """Get recent memory context for prompt injection."""
        s = memory_strings or {
            "recent_rounds": "RECENT ROUNDS SUMMARY:\n",
            "recent_posts": "YOUR RECENT POSTS:\n",
            "current_alliances": "CURRENT ALLIANCES: ",
            "current_targets": "CURRENT TARGETS: ",
            "no_memory": "No previous memory.",
            "no_history": "No history.",
        }
        recent = self.round_summaries[-last_n:] if self.round_summaries else []
        recent_posts = self.posts_history[-6:] if self.posts_history else []

        parts = []
        if recent:
            parts.append(s["recent_rounds"] + "\n".join(recent))
        if recent_posts:
            post_texts = []
            for p in recent_posts:
                platform = p.get("platform", "?")
                text = p.get("text", "")[:150]
                likes = p.get("likes", 0)
                post_texts.append(f"  [{platform}] {text}... ({likes} likes)")
            parts.append(s["recent_posts"] + "\n".join(post_texts))
        if self.alliances:
            parts.append(f"{s['current_alliances']}{', '.join(self.alliances)}")
        if self.targets:
            parts.append(f"{s['current_targets']}{', '.join(self.targets)}")
        return "\n\n".join(parts) if parts else s["no_memory"]

    def get_full_history(self) -> str:
        return "\n\n".join(self.round_summaries) if self.round_summaries else "No history."
