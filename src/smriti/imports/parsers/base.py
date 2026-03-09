"""Base parser protocol and shared utilities for import file parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from smriti.models.events import SourceEvent


@runtime_checkable
class ImportParser(Protocol):
    """Protocol that all import parsers must implement."""

    @property
    def name(self) -> str:
        """Short identifier for this parser (e.g. 'chatgpt', 'markdown')."""
        ...

    def can_parse(self, path: Path) -> bool:
        """Return True if this parser can handle the given file."""
        ...

    def parse(self, path: Path) -> list[SourceEvent]:
        """Parse a file and return a list of SourceEvents for ingestion."""
        ...
