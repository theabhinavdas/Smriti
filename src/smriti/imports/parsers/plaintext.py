"""Plain text parser: imports .txt files as single document events."""

from __future__ import annotations

from pathlib import Path

from smriti.models.events import ActivityContext, SourceEvent

MAX_CONTENT_LENGTH = 50_000


class PlainTextParser:
    """Parses plain text files into SourceEvents."""

    @property
    def name(self) -> str:
        return "plaintext"

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in (".txt", ".text", ".log")

    def parse(self, path: Path) -> list[SourceEvent]:
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []

        if len(content) > MAX_CONTENT_LENGTH:
            return self._chunk(path, content)

        return [self._make_event(path, content)]

    def _chunk(self, path: Path, content: str) -> list[SourceEvent]:
        """Split large files into multiple events by paragraphs."""
        paragraphs = content.split("\n\n")
        events: list[SourceEvent] = []
        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > MAX_CONTENT_LENGTH and current_chunk:
                events.append(self._make_event(path, "\n\n".join(current_chunk)))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            events.append(self._make_event(path, "\n\n".join(current_chunk)))

        return events

    @staticmethod
    def _make_event(path: Path, content: str) -> SourceEvent:
        return SourceEvent(
            source="import",
            event_type="document",
            raw_content=content,
            metadata={"file_name": path.name, "file_path": str(path), "format": "plaintext"},
            activity_context=ActivityContext(project=None),
        )
