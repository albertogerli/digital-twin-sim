"""Google Gemini LLM adapter."""

import asyncio
import logging
import os
from typing import Optional

from google import genai
from google.genai import types

from .base_client import BaseLLMClient
from .json_parser import LLMError

logger = logging.getLogger(__name__)

# Pricing per 1M tokens
GEMINI_PRICING = {
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


class GeminiClient(BaseLLMClient):
    """Async wrapper around Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 budget: float = 5.0):
        super().__init__(budget=budget, max_concurrent=10, rate_limit_delay=0.3)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=self.api_key)
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
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            # Relax safety filters for simulation content (political, corporate scenarios)
            "safety_settings": [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if response_json:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=config,
        )

        # Extract text
        text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        text += part.text
            # Check for blocked content
            finish_reason = getattr(candidate, 'finish_reason', None)
            if not text and finish_reason:
                logger.warning(f"[{component}] Empty response, finish_reason={finish_reason}")
        else:
            # No candidates at all — likely safety filter
            block_reason = getattr(response, 'prompt_feedback', None)
            logger.warning(f"[{component}] No candidates returned. prompt_feedback={block_reason}")

        if not text:
            logger.warning(f"[{component}] Empty text from API (prompt length={len(prompt)})")
            raise LLMError(f"Empty response from Gemini for {component}")

        # Track usage
        input_tokens = getattr(
            response.usage_metadata, "prompt_token_count", len(prompt) // 4
        )
        output_tokens = getattr(
            response.usage_metadata, "candidates_token_count", len(text) // 4
        )
        pricing = GEMINI_PRICING.get(self.model, GEMINI_PRICING[DEFAULT_MODEL])
        cost = self.stats.record(
            self.model, input_tokens, output_tokens,
            pricing["input"], pricing["output"], component
        )

        logger.debug(
            f"[{component}] {self.model} | {input_tokens}in/{output_tokens}out | ${cost:.4f}"
        )
        return text
