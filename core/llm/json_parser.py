"""JSON extraction from LLM responses with fallback strategies."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _try_parse(text: str):
    """Try json.loads, return result or None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _repair_truncated(candidate: str):
    """Try to close unmatched braces/brackets in truncated JSON."""
    open_braces = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    if open_braces <= 0 and open_brackets <= 0:
        return None
    # Strip trailing incomplete values (after last comma or colon)
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"{}[\]]*$', '', candidate, flags=re.DOTALL)
    repaired = repaired.rstrip(", \n\t")
    repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    result = _try_parse(repaired)
    if result is not None:
        logger.warning(f"Repaired truncated JSON (closed {open_braces} braces, {open_brackets} brackets)")
    return result


def parse_json_response(text: str) -> dict | list:
    """Parse JSON from LLM response with fallback extraction."""
    text = text.strip()

    # 1. Direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Gemini sometimes returns JSON without outer braces: '  "key": value, ...'
    #    Wrap in {} and try again
    if text and not text.startswith(("{", "[", "```")):
        wrapped = "{" + text + "}"
        result = _try_parse(wrapped)
        if result is not None:
            return result
        # Also try repair on the wrapped version (truncated without outer braces)
        result = _repair_truncated(wrapped)
        if result is not None:
            return result

    # 3. Markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        result = _try_parse(match.group(1).strip())
        if result is not None:
            return result

    # 4. Find JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = _try_parse(match.group(0))
            if result is not None:
                return result

    # 5. Repair truncated JSON (with outer brace present)
    brace_start = text.find("{")
    if brace_start >= 0:
        result = _repair_truncated(text[brace_start:])
        if result is not None:
            return result

    # 6. Last resort: try bracket-start repair
    bracket_start = text.find("[")
    if bracket_start >= 0:
        result = _repair_truncated(text[bracket_start:])
        if result is not None:
            return result

    logger.error(f"Failed to parse JSON from response: {text[:200]}...")
    raise JSONParseError(f"Could not parse JSON from response: {text[:100]}...")


class LLMError(Exception):
    pass


class BudgetExceededError(LLMError):
    pass


class JSONParseError(LLMError):
    pass
