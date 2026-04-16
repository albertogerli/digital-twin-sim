"""Core social platform engine with SQLite backend — fully domain-agnostic.

Concurrency notes:
- SQLite is opened in WAL mode with `synchronous = NORMAL` so concurrent readers
  don't block writers and fsync cost is bounded.
- `check_same_thread=False` lets us call the sync methods from executor threads
  (via `asyncio.to_thread`) when running multiple simulations in one process.
- A per-instance threading lock serializes writes against the single connection —
  cheaper than opening a new connection per call and avoids "database is locked".
"""

import asyncio
import logging
import sqlite3
import threading
import os
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id TEXT NOT NULL,
    author_tier INTEGER NOT NULL,
    platform TEXT NOT NULL,
    content TEXT NOT NULL,
    parent_id INTEGER,
    channel TEXT,
    round INTEGER NOT NULL,
    timestamp_sim TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    reaction_type TEXT NOT NULL,
    round INTEGER NOT NULL,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

CREATE TABLE IF NOT EXISTS follows (
    follower_id TEXT NOT NULL,
    followed_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    since_round INTEGER NOT NULL,
    PRIMARY KEY (follower_id, followed_id, platform)
);

CREATE TABLE IF NOT EXISTS agent_states (
    agent_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    position REAL NOT NULL,
    sentiment TEXT,
    engagement REAL,
    memory_summary TEXT,
    PRIMARY KEY (agent_id, round)
);

CREATE INDEX IF NOT EXISTS idx_posts_round ON posts(round);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_id);
CREATE INDEX IF NOT EXISTS idx_reactions_post ON reactions(post_id);
CREATE INDEX IF NOT EXISTS idx_reactions_round ON reactions(round);
"""


class PlatformEngine:
    """Manages social platforms with SQLite."""

    def __init__(self, db_path: str = "outputs/social.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".",
                    exist_ok=True)
        self.db_path = db_path
        # check_same_thread=False allows safe use from asyncio.to_thread / executors
        # provided all writes go through the shared lock below.
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._write_lock = threading.Lock()
        self._pending_writes = 0
        self._commit_every = 50  # batch commits under load; flush() forces a commit
        self._apply_pragmas()
        self._init_db()

    def _apply_pragmas(self):
        """Tune SQLite for concurrent simulation workloads."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL")
        cur.execute("PRAGMA synchronous = NORMAL")
        cur.execute("PRAGMA cache_size = -65536")  # 64 MiB
        cur.execute("PRAGMA temp_store = MEMORY")
        cur.execute("PRAGMA busy_timeout = 5000")
        cur.execute("PRAGMA wal_autocheckpoint = 1000")
        cur.close()

    def _init_db(self):
        with self._write_lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _maybe_commit(self, force: bool = False):
        """Batch commits to reduce fsync pressure under concurrent writes."""
        self._pending_writes += 1
        if force or self._pending_writes >= self._commit_every:
            self.conn.commit()
            self._pending_writes = 0

    def flush(self):
        """Force-commit any pending writes. Call at round boundaries."""
        with self._write_lock:
            if self._pending_writes:
                self.conn.commit()
                self._pending_writes = 0

    async def aflush(self):
        await asyncio.to_thread(self.flush)

    def close(self):
        with self._write_lock:
            if self._pending_writes:
                self.conn.commit()
            self.conn.close()

    def add_post(self, post_data: dict, round_num: int) -> int:
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO posts (author_id, author_tier, platform, content,
                   parent_id, channel, round, timestamp_sim)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    post_data["author_id"],
                    post_data.get("author_tier", 1),
                    post_data.get("platform") or "social",
                    post_data["text"],
                    post_data.get("parent_id"),
                    post_data.get("channel"),
                    round_num,
                    post_data.get("timestamp_sim", ""),
                ),
            )
            self._maybe_commit()
            return cursor.lastrowid

    async def aadd_post(self, post_data: dict, round_num: int) -> int:
        return await asyncio.to_thread(self.add_post, post_data, round_num)

    def add_reaction(self, post_id: int, agent_id: str, reaction_type: str,
                     round_num: int):
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO reactions (post_id, agent_id, reaction_type, round) "
                "VALUES (?, ?, ?, ?)",
                (post_id, agent_id, reaction_type, round_num),
            )
            self._maybe_commit()

    async def aadd_reaction(self, post_id: int, agent_id: str,
                            reaction_type: str, round_num: int):
        await asyncio.to_thread(self.add_reaction, post_id, agent_id,
                                reaction_type, round_num)

    def add_follow(self, follower_id: str, followed_id: str, platform: str,
                   round_num: int):
        with self._write_lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO follows (follower_id, followed_id, platform, "
                "since_round) VALUES (?, ?, ?, ?)",
                (follower_id, followed_id, platform, round_num),
            )
            self._maybe_commit()

    def save_agent_state(self, agent_id: str, round_num: int, position: float,
                         sentiment: str = "", engagement: float = 0.0,
                         memory_summary: str = ""):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO agent_states
                   (agent_id, round, position, sentiment, engagement, memory_summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent_id, round_num, position, sentiment, engagement, memory_summary),
            )
            self._maybe_commit()

    def get_posts_by_round(self, round_num: int, platform: Optional[str] = None,
                           limit: int = 100) -> list[dict]:
        query = "SELECT * FROM posts WHERE round = ?"
        params: list = [round_num]
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_posts_by_author(self, author_id: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE author_id = ? ORDER BY round DESC, id DESC "
            "LIMIT ?",
            (author_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_posts(self, round_num: int, platform: Optional[str] = None,
                      top_n: int = 10) -> list[dict]:
        query = """
            SELECT p.*,
                   COALESCE(SUM(CASE WHEN r.reaction_type = 'like' THEN 1 ELSE 0 END), 0) as likes,
                   COALESCE(SUM(CASE WHEN r.reaction_type = 'repost' THEN 1 ELSE 0 END), 0) as reposts,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.parent_id = p.id) as reply_count
            FROM posts p
            LEFT JOIN reactions r ON r.post_id = p.id
            WHERE p.round = ? AND p.parent_id IS NULL
        """
        params: list = [round_num]
        if platform:
            query += " AND p.platform = ?"
            params.append(platform)
        query += " GROUP BY p.id ORDER BY (likes + reposts * 2 + reply_count * 3) DESC LIMIT ?"
        params.append(top_n)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_following_ids(self, agent_id: str, platform: str = "social") -> list[str]:
        rows = self.conn.execute(
            "SELECT followed_id FROM follows WHERE follower_id = ? AND platform = ?",
            (agent_id, platform),
        ).fetchall()
        return [r["followed_id"] for r in rows]

    def get_round_stats(self, round_num: int) -> dict:
        post_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM posts WHERE round = ?", (round_num,)
        ).fetchone()["cnt"]
        reaction_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM reactions WHERE round = ?", (round_num,)
        ).fetchone()["cnt"]
        return {"posts": post_count, "reactions": reaction_count}

    def get_total_stats(self) -> dict:
        posts = self.conn.execute("SELECT COUNT(*) as cnt FROM posts").fetchone()["cnt"]
        reactions = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM reactions"
        ).fetchone()["cnt"]
        follows = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM follows"
        ).fetchone()["cnt"]
        return {"total_posts": posts, "total_reactions": reactions,
                "total_follows": follows}

    def format_viral_posts(self, round_num: int, top_n: int = 5) -> str:
        posts = self.get_top_posts(round_num, top_n=top_n)
        if not posts:
            return "No viral posts available."
        lines = []
        for i, p in enumerate(posts, 1):
            likes = p.get("likes", 0)
            reposts = p.get("reposts", 0)
            replies = p.get("reply_count", 0)
            lines.append(
                f"{i}. [@{p['author_id']}] {p['content'][:200]}\n"
                f"   ({likes} likes, {reposts} reposts, {replies} replies)"
            )
        return "\n".join(lines)
