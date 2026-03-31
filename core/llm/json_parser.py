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

    # Gemini sometimes returns JSON without outer braces: '  "key": value, ...'
    # Wrap in {} and try again
    if text and not text.startswith(("{", "[", "```")):
        wrapped = "{" + text + "}"
        try:
            return json.loads(wrapped)
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

    # Try to repair truncated JSON (missing closing braces/brackets)
    brace_start = text.find("{")
    if brace_start >= 0:
        candidate = text[brace_start:]
        # Count unmatched braces/brackets and close them
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces > 0 or open_brackets > 0:
            # Strip trailing incomplete values (after last comma or colon)
            repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"{}[\]]*$', '', candidate, flags=re.DOTALL)
            repaired = repaired.rstrip(", \n\t")
            repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            try:
                result = json.loads(repaired)
                logger.warning(f"Repaired truncated JSON (closed {open_braces} braces, {open_brackets} brackets)")
                return result
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
