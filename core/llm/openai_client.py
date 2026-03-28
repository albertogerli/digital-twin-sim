"""OpenAI LLM adapter (GPT-5.4-mini and compatible models)."""

import asyncio
import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Pricing per 1M tokens
OPENAI_PRICING = {
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIClient(BaseLLMClient):
    """Async wrapper around OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 budget: float = 5.0):
        super().__init__(budget=budget, max_concurrent=10, rate_limit_delay=0.1)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: int,
        response_json: bool,
        component: str,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
        }

        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""

        # Track usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else len(prompt) // 4
        output_tokens = usage.completion_tokens if usage else len(text) // 4

        pricing = OPENAI_PRICING.get(self.model, OPENAI_PRICING[DEFAULT_MODEL])
        cost = self.stats.record(
            self.model, input_tokens, output_tokens,
            pricing["input"], pricing["output"], component
        )

        logger.debug(
            f"[{component}] {self.model} | {input_tokens}in/{output_tokens}out | ${cost:.4f}"
        )
        return text
