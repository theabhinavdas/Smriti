"""Markdown parser: imports .md files (including Obsidian notes).

Splits documents by top-level headings (# or ##) so each section becomes
its own SourceEvent. Strips YAML frontmatter and uses it as metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

from smriti.models.events import ActivityContext, SourceEvent

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)

MAX_SECTION_LENGTH = 50_000


class MarkdownParser:
    """Parses markdown files into SourceEvents, one per top-level section."""

    @property
    def name(self) -> str:
        return "markdown"

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in (".md", ".markdown")

    def parse(self, path: Path) -> list[SourceEvent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            return []

        frontmatter = self._extract_frontmatter(raw)
        body = _FRONTMATTER_RE.sub("", raw).strip()

        if not body:
            return []

        title = frontmatter.get("title") or path.stem
        sections = self._split_sections(body)

        if not sections:
            return [self._make_event(path, title, body, frontmatter)]

        events: list[SourceEvent] = []
        for heading, content in sections:
            section_title = f"{title} > {heading}" if heading else title
            text = f"# {heading}\n\n{content}" if heading else content
            if text.strip():
                events.append(self._make_event(path, section_title, text, frontmatter))

        return events

    @staticmethod
    def _extract_frontmatter(raw: str) -> dict[str, str]:
        """Parse simple key: value pairs from YAML frontmatter."""
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return {}
        result: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    @staticmethod
    def _split_sections(body: str) -> list[tuple[str, str]]:
        """Split markdown into (heading, content) pairs on # and ## headings."""
        headings = list(_HEADING_RE.finditer(body))
        if not headings:
            return []

        sections: list[tuple[str, str]] = []

        if headings[0].start() > 0:
            preamble = body[: headings[0].start()].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(headings):
            heading_text = match.group(2).strip()
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            content = body[start:end].strip()
            sections.append((heading_text, content))

        return sections

    @staticmethod
    def _make_event(
        path: Path, title: str, content: str, frontmatter: dict[str, str]
    ) -> SourceEvent:
        metadata: dict[str, object] = {
            "file_name": path.name,
            "file_path": str(path),
            "format": "markdown",
            "title": title,
        }
        if frontmatter:
            metadata["frontmatter"] = frontmatter

        return SourceEvent(
            source="obsidian",
            event_type="note",
            raw_content=content[:MAX_SECTION_LENGTH],
            metadata=metadata,
            activity_context=ActivityContext(project=None),
        )
