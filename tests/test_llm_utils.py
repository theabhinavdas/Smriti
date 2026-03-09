"""Tests for smriti.llm_utils."""

import pytest

from smriti.llm_utils import parse_llm_json


class TestParseLlmJson:
    def test_plain_json_array(self):
        assert parse_llm_json('[1, 2, 3]') == [1, 2, 3]

    def test_plain_json_object(self):
        assert parse_llm_json('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self):
        raw = '```json\n[{"summary": "test"}]\n```'
        assert parse_llm_json(raw) == [{"summary": "test"}]

    def test_fenced_no_lang_tag(self):
        raw = '```\n[1, 2]\n```'
        assert parse_llm_json(raw) == [1, 2]

    def test_leading_trailing_whitespace(self):
        assert parse_llm_json('  \n[1]\n  ') == [1]

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n[1, 2, 3]\nHope that helps!'
        assert parse_llm_json(raw) == [1, 2, 3]

    def test_json_object_with_surrounding_text(self):
        raw = 'Sure!\n{"a": 1}\nDone.'
        assert parse_llm_json(raw) == {"a": 1}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json("   \n  ")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            parse_llm_json("no json here at all")

    def test_nested_json(self):
        raw = '```json\n[{"facts": ["a", "b"], "nested": {"x": 1}}]\n```'
        result = parse_llm_json(raw)
        assert result == [{"facts": ["a", "b"], "nested": {"x": 1}}]

    def test_extra_data_after_json(self):
        """The 'Extra data' error case from the bug report."""
        raw = '[{"summary": "test"}]\n\nSome trailing commentary.'
        assert parse_llm_json(raw) == [{"summary": "test"}]
