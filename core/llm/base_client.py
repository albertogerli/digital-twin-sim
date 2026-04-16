"""Abstract LLM client interface — all providers implement this.

Concurrency model (for ≥50 concurrent simulation streams):
- `_semaphore` caps in-flight provider calls to `max_concurrent`.
- `_rate_lock` serializes access to `_last_call_time` so many coroutines
  cannot collapse into the same rate-limit slot (the previous implementation
  let all waiters read the same stale timestamp and issue a burst).
- `_budget_lock` guards the reservation/commit pattern that prevents N
  concurrent calls from each seeing `total_cost < budget` and collectively
  blowing the cap by up to N× `_expected_cost_per_call`.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from .usage_stats import UsageStats
from .json_parser import parse_json_response, LLMError, BudgetExceededError

logger = logging.getLogger(__name__)

# Heuristic reservation for an in-flight call; the real cost is committed
# when `_call_api` records tokens via UsageStats.record().
_DEFAULT_RESERVATION_USD = 0.005


class BaseLLMClient(ABC):
    """Abstract base for all LLM providers (Gemini, Claude, OpenAI)."""

    def __init__(self, budget: float = 5.0, max_concurrent: int = 10,
                 rate_limit_delay: float = 0.3,
                 reservation_usd: float = _DEFAULT_RESERVATION_USD):
        self.stats = UsageStats()
        self.budget = budget
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0
        self._rate_lock = asyncio.Lock()
        self._budget_lock = asyncio.Lock()
        self._reserved_cost = 0.0
        self._reservation_usd = reservation_usd

    async def _reserve_budget(self) -> float:
        """Reserve headroom against the budget before issuing a provider call.

        Returns the reserved amount, which MUST be released via
        `_release_reservation` once the call completes (success or failure).
        """
        async with self._budget_lock:
            projected = self.stats.total_cost + self._reserved_cost + self._reservation_usd
            if projected > self.budget:
                raise BudgetExceededError(
                    f"Budget would be exceeded: ${projected:.4f} > ${self.budget:.2f} "
                    f"(spent=${self.stats.total_cost:.4f}, in-flight=${self._reserved_cost:.4f})"
                )
            self._reserved_cost += self._reservation_usd
            return self._reservation_usd

    async def _release_reservation(self, amount: float):
        async with self._budget_lock:
            self._reserved_cost = max(0.0, self._reserved_cost - amount)

    async def _enforce_rate_limit(self):
        """Serialize rate-limit gating across concurrent callers."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._rate_limit_delay:
                await asyncio.sleep(self._rate_limit_delay - elapsed)
            self._last_call_time = time.time()

    @abstractmethod
    async def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: int,
        response_json: bool,
        component: str,
    ) -> str:
        """Provider-specific API call. Must track usage via self.stats."""
        ...

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_output_tokens: int = 2000,
        component: str = "unknown",
        response_json: bool = True,
        retries: int = 4,
    ) -> str:
        """Generate content with retry, rate limiting, and cost tracking."""
        async with self._semaphore:
            reserved = await self._reserve_budget()
            try:
                await self._enforce_rate_limit()

                last_error = None
                for attempt in range(retries):
                    try:
                        return await self._call_api(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            response_json=response_json,
                            component=component,
                        )
                    except BudgetExceededError:
                        raise
                    except Exception as e:
                        last_error = e
                        self.stats.errors += 1
                        # Treat 429 / rate-limit as needing a longer pause with
                        # full jitter; other errors get standard exponential.
                        msg = str(e).lower()
                        is_rate_limit = ("429" in msg or "rate" in msg
                                         or "quota" in msg or "resource_exhausted" in msg)
                        base = 4.0 if is_rate_limit else 1.0
                        wait = min(30.0, base * (2 ** attempt))
                        wait = random.uniform(0.5 * wait, wait)  # full jitter
                        logger.warning(
                            f"[{component}] Attempt {attempt+1}/{retries} failed"
                            f"{' (rate-limit)' if is_rate_limit else ''}: {e}. "
                            f"Retrying in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)

                raise LLMError(f"All retries failed for {component}: {last_error}")
            finally:
                await self._release_reservation(reserved)

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4000,
        component: str = "unknown",
        unwrap_list: bool = True,
        parse_retries: int = 2,
    ) -> dict | list:
        """Generate and parse JSON response.

        If unwrap_list=True (default), a single-element list is unwrapped to
        its contained dict.  This prevents 'list' object has no attribute 'get'
        errors when the LLM wraps its JSON object in an array.

        Retries on JSON parse failure (e.g. truncated output) up to
        parse_retries times with increased max_output_tokens.
        """
        tokens = max_output_tokens
        last_error = None
        for attempt in range(1 + parse_retries):
            text = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=tokens,
                component=component,
                response_json=True,
            )
            try:
                parsed = parse_json_response(text)
                if unwrap_list and isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    logger.debug(f"[{component}] Unwrapped single-element list to dict")
                    return parsed[0]
                return parsed
            except Exception as e:
                last_error = e
                tokens = int(tokens * 1.5)  # increase token budget for retry
                logger.warning(
                    f"[{component}] JSON parse failed (attempt {attempt+1}), "
                    f"retrying with max_output_tokens={tokens}: {e}"
                )
        raise last_error

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_output_tokens: int = 2000,
        component: str = "unknown",
    ) -> str:
        """Generate plain text response (no JSON)."""
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            component=component,
            response_json=False,
        )
