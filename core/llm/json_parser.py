"""JSON extraction from LLM responses with fallback strategies."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> dict | list:
    """Parse JSON from LLM response with fallback extraction."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    logger.error(f"Failed to parse JSON from response: {text[:200]}...")
    raise JSONParseError(f"Could not parse JSON from response: {text[:100]}...")


class LLMError(Exception):
    pass


class BudgetExceededError(LLMError):
    pass


class JSONParseError(LLMError):
    pass
