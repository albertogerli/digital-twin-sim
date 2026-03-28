"""Feed algorithms: hot score for short-form, Wilson score for long-form channels."""

import math


def hot_score(likes: int, reposts: int, replies: int,
              hours_since_post: float) -> float:
    """Hot score: engagement weighted by recency."""
    engagement = likes + 2 * reposts + 3 * replies
    time_decay = (hours_since_post + 2) ** 1.5
    return engagement / max(time_decay, 0.001)


def wilson_score(upvotes: int, downvotes: int, confidence: float = 0.95) -> float:
    """Wilson score interval lower bound for sorting."""
    n = upvotes + downvotes
    if n == 0:
        return 0.0
    z = 1.96 if confidence == 0.95 else 1.645
    p_hat = upvotes / n
    denominator = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    return (center - spread) / denominator


class FeedAlgorithm:
    """Generates personalized feeds for agents."""

    def __init__(self, platform_engine):
        self.platform = platform_engine

    def get_feed(self, agent_id: str, round_num: int,
                 feed_size: int = 13) -> list[dict]:
        """Get feed: top from followed + top trending."""
        following = self.platform.get_following_ids(agent_id)

        followed_posts = []
        if following:
            placeholders = ",".join(["?"] * len(following))
            rows = self.platform.conn.execute(
                f"""SELECT p.*,
                       COALESCE(SUM(CASE WHEN r.reaction_type='like' THEN 1 ELSE 0 END), 0) as likes,
                       COALESCE(SUM(CASE WHEN r.reaction_type='repost' THEN 1 ELSE 0 END), 0) as reposts,
                       (SELECT COUNT(*) FROM posts p2 WHERE p2.parent_id = p.id) as reply_count
                FROM posts p
                LEFT JOIN reactions r ON r.post_id = p.id
                WHERE p.author_id IN ({placeholders})
                  AND p.round = ? AND p.parent_id IS NULL
                GROUP BY p.id""",
                (*following, round_num),
            ).fetchall()
            followed_posts = [dict(r) for r in rows]

        for p in followed_posts:
            hours = max(1, (p.get("id", 0) % 168))
            p["hot_score"] = hot_score(
                p.get("likes", 0), p.get("reposts", 0),
                p.get("reply_count", 0), hours
            )

        followed_posts.sort(key=lambda x: x.get("hot_score", 0), reverse=True)
        top_followed = followed_posts[:10]

        trending = self.platform.get_top_posts(round_num, top_n=3)

        seen_ids = {p["id"] for p in top_followed}
        for t in trending:
            if t["id"] not in seen_ids:
                top_followed.append(t)
                seen_ids.add(t["id"])

        return top_followed[:feed_size]
