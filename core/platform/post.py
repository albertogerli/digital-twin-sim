"""Data models for posts, comments, and reactions."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Post:
    """A post on any platform/channel."""
    id: Optional[int] = None
    author_id: str = ""
    author_tier: int = 1
    platform: str = "social"
    content: str = ""
    parent_id: Optional[int] = None
    channel: Optional[str] = None
    round: int = 0
    timestamp_sim: str = ""
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    upvotes: int = 0
    downvotes: int = 0
    # RAG citations: chunks the agent's reasoning was grounded in for this post
    citations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_reply(self) -> bool:
        return self.parent_id is not None

    @property
    def engagement_total(self) -> int:
        return self.likes + self.reposts * 2 + self.replies * 3
