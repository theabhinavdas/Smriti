"""Utilities for parsing LLM responses that should contain JSON."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)```", re.DOTALL
)


def parse_llm_json(raw: str | None) -> Any:
    """Parse JSON from an LLM response, handling common quirks.

    Handles:
    - None or empty responses (raises ValueError)
    - Markdown code fences wrapping the JSON
    - Leading/trailing whitespace or commentary
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response")

    text = raw.strip()

    # Strip markdown code fences if present
    match = _FENCED_JSON_RE.search(text)
    if match:
        text = match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [ ... ] or { ... } block
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")
