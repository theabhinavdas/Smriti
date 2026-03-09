"""ChatGPT JSON export parser.

Handles the official OpenAI data export format (conversations.json).
Each file is a JSON array of conversation objects. Each conversation has
a `mapping` dict that forms a tree of message nodes.

The parser extracts each conversation as a separate SourceEvent with the
full user/assistant dialogue as the raw_content.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smriti.models.events import ActivityContext, SourceEvent

logger = logging.getLogger(__name__)


class ChatGPTParser:
    """Parses ChatGPT conversations.json exports."""

    @property
    def name(self) -> str:
        return "chatgpt"

    def can_parse(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_chars = f.read(200)
            if not first_chars.strip().startswith("["):
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) == 0:
                return False
            return "mapping" in data[0]
        except Exception:
            return False

    def parse(self, path: Path) -> list[SourceEvent]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []

        events: list[SourceEvent] = []
        for conversation in data:
            event = self._parse_conversation(conversation, path)
            if event is not None:
                events.append(event)

        return events

    def _parse_conversation(
        self, conv: dict[str, Any], path: Path
    ) -> SourceEvent | None:
        title = conv.get("title", "Untitled")
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            return None

        messages = self._extract_messages(mapping)
        if not messages:
            return None

        lines: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Tool: {content}")

        raw_content = f"# {title}\n\n" + "\n\n".join(lines)

        create_time = conv.get("create_time")
        timestamp = (
            datetime.fromtimestamp(create_time, tz=timezone.utc)
            if isinstance(create_time, (int, float)) and create_time > 0
            else datetime.now(timezone.utc)
        )

        model_slug = self._detect_model(mapping)

        return SourceEvent(
            source="chatgpt",
            event_type="conversation",
            timestamp=timestamp,
            raw_content=raw_content,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "format": "chatgpt",
                "title": title,
                "conversation_id": conv.get("id", ""),
                "message_count": len(messages),
                "model": model_slug,
            },
            activity_context=ActivityContext(project=None),
        )

    @staticmethod
    def _extract_messages(mapping: dict[str, Any]) -> list[dict[str, str]]:
        """Walk the mapping tree in order and extract user/assistant messages."""
        nodes: dict[str, dict[str, Any]] = {}
        children_map: dict[str, list[str]] = {}
        root_id: str | None = None

        for node_id, node in mapping.items():
            nodes[node_id] = node
            parent_id = node.get("parent")
            if parent_id is None:
                root_id = node_id
            else:
                children_map.setdefault(parent_id, []).append(node_id)

        if root_id is None:
            return []

        messages: list[dict[str, str]] = []
        stack = [root_id]
        while stack:
            nid = stack.pop(0)
            node = nodes.get(nid, {})
            msg = node.get("message")
            if msg and isinstance(msg, dict):
                role = msg.get("author", {}).get("role", "")
                content = _extract_content(msg)
                if role in ("user", "assistant", "tool") and content.strip():
                    messages.append({"role": role, "content": content})
            for child_id in children_map.get(nid, []):
                stack.append(child_id)

        return messages

    @staticmethod
    def _detect_model(mapping: dict[str, Any]) -> str:
        """Try to find the model slug from assistant messages."""
        for node in mapping.values():
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            meta = msg.get("metadata", {})
            if isinstance(meta, dict) and "model_slug" in meta:
                return meta["model_slug"]
        return "unknown"


def _extract_content(msg: dict[str, Any]) -> str:
    """Extract text content from a ChatGPT message node."""
    content_obj = msg.get("content", {})
    if not isinstance(content_obj, dict):
        return str(content_obj) if content_obj else ""

    content_type = content_obj.get("content_type", "")
    parts = content_obj.get("parts", [])

    if content_type == "text" and isinstance(parts, list):
        text_parts = [str(p) for p in parts if isinstance(p, str) and p.strip()]
        return "\n".join(text_parts)

    if content_type == "code" and isinstance(parts, list):
        return "\n".join(str(p) for p in parts)

    if isinstance(parts, list) and parts:
        return "\n".join(str(p) for p in parts if isinstance(p, str))

    text = content_obj.get("text", "")
    return str(text) if text else ""
