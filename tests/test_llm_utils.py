"""Tests for LLM utilities — JSON parser, UsageStats, budget errors."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm.json_parser import (
    BudgetExceededError,
    JSONParseError,
    parse_json_response,
    _repair_truncated,
    _try_parse,
)
from core.llm.usage_stats import UsageStats


# ── Tests: _try_parse ────────────────────────────────────────────────────────


class TestTryParse:
    def test_valid_json(self):
        assert _try_parse('{"a": 1}') == {"a": 1}

    def test_invalid_json(self):
        assert _try_parse('not json') is None

    def test_valid_array(self):
        assert _try_parse('[1, 2, 3]') == [1, 2, 3]

    def test_empty_string(self):
        assert _try_parse('') is None


# ── Tests: parse_json_response ───────────────────────────────────────────────


class TestParseJsonResponse:
    def test_direct_json_object(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_direct_json_array(self):
        result = parse_json_response('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"name": "test", "value": 42}\n```'
        result = parse_json_response(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_block(self):
        text = '```\n{"name": "test"}\n```'
        result = parse_json_response(text)
        assert result == {"name": "test"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the JSON:\n{"data": true}\nEnd of response.'
        result = parse_json_response(text)
        assert result == {"data": True}

    def test_gemini_no_outer_braces(self):
        """Gemini sometimes returns JSON without outer braces."""
        text = '"key": "value", "num": 42'
        result = parse_json_response(text)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_malformed_raises_json_parse_error(self):
        with pytest.raises(JSONParseError):
            parse_json_response("completely invalid not json at all xyz")

    def test_empty_string_raises(self):
        with pytest.raises(JSONParseError):
            parse_json_response("")

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = parse_json_response(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_whitespace_padded(self):
        result = parse_json_response('   \n  {"a": 1}  \n  ')
        assert result == {"a": 1}


# ── Tests: truncated JSON repair ─────────────────────────────────────────────


class TestRepairTruncated:
    def test_unclosed_brace(self):
        """JSON with missing closing brace should be repaired."""
        result = _repair_truncated('{"a": 1, "b": 2')
        assert result is not None
        assert result["a"] == 1

    def test_unclosed_bracket(self):
        result = _repair_truncated('[1, 2, 3')
        assert result is not None
        assert result == [1, 2, 3]

    def test_balanced_json_returns_none(self):
        """Already balanced JSON should return None (no repair needed)."""
        result = _repair_truncated('{"a": 1}')
        assert result is None

    def test_parse_json_response_handles_truncated(self):
        """parse_json_response should recover truncated JSON."""
        text = '{"name": "test", "items": [1, 2'
        result = parse_json_response(text)
        assert result["name"] == "test"


# ── Tests: UsageStats ────────────────────────────────────────────────────────


class TestUsageStats:
    def test_initial_state(self):
        stats = UsageStats()
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.call_count == 0

    def test_record_single_call(self):
        stats = UsageStats()
        cost = stats.record(
            model="gemini-flash",
            input_tokens=1000,
            output_tokens=500,
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.40,
            component="elite_agent",
        )
        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 500
        assert stats.call_count == 1
        expected_cost = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
        assert abs(cost - expected_cost) < 1e-10
        assert abs(stats.total_cost - expected_cost) < 1e-10

    def test_record_multiple_calls(self):
        stats = UsageStats()
        stats.record("m1", 1000, 500, 0.10, 0.40, "comp_a")
        stats.record("m1", 2000, 1000, 0.10, 0.40, "comp_b")
        assert stats.total_input_tokens == 3000
        assert stats.total_output_tokens == 1500
        assert stats.call_count == 2

    def test_component_tracking(self):
        stats = UsageStats()
        stats.record("m1", 1000, 500, 0.10, 0.40, "elite")
        stats.record("m1", 1000, 500, 0.10, 0.40, "elite")
        stats.record("m1", 1000, 500, 0.10, 0.40, "citizen")

        assert stats.calls_by_component["elite"]["calls"] == 2
        assert stats.calls_by_component["citizen"]["calls"] == 1

    def test_cost_computation_accuracy(self):
        """Cost = (input_tokens / 1M) * input_rate + (output_tokens / 1M) * output_rate."""
        stats = UsageStats()
        cost = stats.record(
            model="test",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cost_per_1m_input=1.0,
            cost_per_1m_output=2.0,
            component="test",
        )
        # 1M tokens * $1/1M + 1M tokens * $2/1M = $3
        assert abs(cost - 3.0) < 1e-10

    def test_zero_tokens_zero_cost(self):
        stats = UsageStats()
        cost = stats.record("m", 0, 0, 1.0, 1.0, "x")
        assert cost == 0.0

    def test_summary_output(self):
        stats = UsageStats()
        stats.record("m", 100, 50, 0.10, 0.40, "test")
        summary = stats.summary()
        assert "Total calls: 1" in summary
        assert "Tokens:" in summary
        assert "Cost:" in summary
        assert "test:" in summary

    def test_errors_field(self):
        stats = UsageStats()
        stats.errors = 3
        assert stats.errors == 3
        assert "Errors: 3" in stats.summary()


# ── Tests: BudgetExceededError ───────────────────────────────────────────────


class TestBudgetExceededError:
    def test_is_llm_error_subclass(self):
        from core.llm.json_parser import LLMError
        assert issubclass(BudgetExceededError, LLMError)

    def test_can_raise_and_catch(self):
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("Budget of $5.00 exceeded")


# ── Tests: JSONParseError ────────────────────────────────────────────────────


class TestJSONParseError:
    def test_is_llm_error_subclass(self):
        from core.llm.json_parser import LLMError
        assert issubclass(JSONParseError, LLMError)

    def test_message_preserved(self):
        try:
            raise JSONParseError("bad json")
        except JSONParseError as e:
            assert "bad json" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
