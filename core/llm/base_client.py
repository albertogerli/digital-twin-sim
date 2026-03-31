"""Abstract LLM client interface — all providers implement this."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from .usage_stats import UsageStats
from .json_parser import parse_json_response, LLMError, BudgetExceededError

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base for all LLM providers (Gemini, Claude, OpenAI)."""

    def __init__(self, budget: float = 5.0, max_concurrent: int = 10,
                 rate_limit_delay: float = 0.3):
        self.stats = UsageStats()
        self.budget = budget
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0

    async def _enforce_limits(self):
        """Check budget and apply rate limiting."""
        if self.stats.total_cost >= self.budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.stats.total_cost:.2f} >= ${self.budget:.2f}"
            )
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
            await self._enforce_limits()

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
                except Exception as e:
                    last_error = e
                    self.stats.errors += 1
                    wait = (2 ** attempt) + 0.5
                    logger.warning(
                        f"[{component}] Attempt {attempt+1}/{retries} failed: {e}. "
                        f"Retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)

            raise LLMError(f"All retries failed for {component}: {last_error}")

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
