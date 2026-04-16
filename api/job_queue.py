"""Redis-backed distributed job queue for DigitalTwinSim.

Falls back to asyncio.Semaphore when REDIS_URL is not set.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
MAX_CONCURRENT = int(os.getenv("DTS_MAX_CONCURRENT", "4"))
LOCK_TTL = 3600  # 1 hour max sim duration before lock expires
QUEUE_KEY = "dts:job_queue"
RUNNING_KEY = "dts:running_jobs"

_redis = None
_fallback_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def get_redis():
    """Get or create Redis connection."""
    global _redis
    if _redis is None and REDIS_URL:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await _redis.ping()
            logger.info("Redis connected for job queue")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local semaphore: {e}")
            _redis = None
    return _redis


async def check_health() -> bool:
    """Check Redis connectivity."""
    r = await get_redis()
    if not r:
        return False
    try:
        return await r.ping()
    except Exception:
        return False


async def running_count() -> int:
    """Count currently running jobs."""
    r = await get_redis()
    if not r:
        return 0
    try:
        return await r.scard(RUNNING_KEY)
    except Exception:
        return 0


@asynccontextmanager
async def acquire(sim_id: str):
    """Acquire a job slot. Blocks until a slot is available.

    Uses Redis SETNX for distributed locking when available,
    falls back to asyncio.Semaphore for single-instance mode.
    """
    r = await get_redis()

    if not r:
        # Fallback: local semaphore
        async with _fallback_semaphore:
            yield
        return

    # Redis-based distributed semaphore
    lock_key = f"dts:job:{sim_id}"

    try:
        # Wait for slot
        while True:
            count = await r.scard(RUNNING_KEY)
            if count < MAX_CONCURRENT:
                # Try to claim slot
                added = await r.sadd(RUNNING_KEY, sim_id)
                if added:
                    await r.set(lock_key, str(time.time()), ex=LOCK_TTL)
                    break
            # Wait and retry
            await asyncio.sleep(1.0)

        yield
    finally:
        # Release slot
        await r.srem(RUNNING_KEY, sim_id)
        await r.delete(lock_key)


async def cleanup_stale_jobs():
    """Remove stale job locks (e.g. after crash)."""
    r = await get_redis()
    if not r:
        return
    members = await r.smembers(RUNNING_KEY)
    for sim_id in members:
        lock_key = f"dts:job:{sim_id}"
        if not await r.exists(lock_key):
            await r.srem(RUNNING_KEY, sim_id)
            logger.info(f"Cleaned stale job lock: {sim_id}")
