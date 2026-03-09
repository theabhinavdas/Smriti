"""Tests for import file parsers."""

from __future__ import annotations

import json
from pathlib import Path

from smriti.imports.parsers.base import ImportParser
from smriti.imports.parsers.chatgpt import ChatGPTParser
from smriti.imports.parsers.markdown import MarkdownParser
from smriti.imports.parsers.plaintext import PlainTextParser


def _chatgpt_conversation(
    title: str = "Test Chat",
    conv_id: str = "conv-1",
    create_time: float = 1709913600.0,
    user_msg: str = "Hello",
    assistant_msg: str = "Hi there!",
    model_slug: str = "gpt-4",
) -> dict:
    """Build a minimal ChatGPT conversation matching the official export format."""
    return {
        "id": conv_id,
        "title": title,
        "create_time": create_time,
        "mapping": {
            "root": {"id": "root", "parent": None, "message": None},
            "msg-1": {
                "id": "msg-1",
                "parent": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [user_msg]},
                    "metadata": {},
                },
            },
            "msg-2": {
                "id": "msg-2",
                "parent": "msg-1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [assistant_msg]},
                    "metadata": {"model_slug": model_slug},
                },
            },
        },
    }


class TestPlainTextParser:
    def test_implements_protocol(self) -> None:
        parser = PlainTextParser()
        assert isinstance(parser, ImportParser)
        assert parser.name == "plaintext"

    def test_can_parse_txt(self, tmp_path: Path) -> None:
        parser = PlainTextParser()
        assert parser.can_parse(tmp_path / "notes.txt")
        assert parser.can_parse(tmp_path / "readme.text")
        assert parser.can_parse(tmp_path / "server.log")
        assert not parser.can_parse(tmp_path / "data.json")
        assert not parser.can_parse(tmp_path / "notes.md")

    def test_parse_simple_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!\nSecond line.")
        parser = PlainTextParser()
        events = parser.parse(f)
        assert len(events) == 1
        assert events[0].source == "import"
        assert events[0].event_type == "document"
        assert "Hello, world!" in events[0].raw_content
        assert events[0].metadata["file_name"] == "hello.txt"
        assert events[0].metadata["format"] == "plaintext"

    def test_parse_empty_file_returns_nothing(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("   \n  \n  ")
        parser = PlainTextParser()
        events = parser.parse(f)
        assert events == []

    def test_large_file_is_chunked(self, tmp_path: Path) -> None:
        paragraph = "A" * 10_000
        content = ("\n\n").join([paragraph] * 10)  # 10 paragraphs, ~100K chars
        f = tmp_path / "big.txt"
        f.write_text(content)
        parser = PlainTextParser()
        events = parser.parse(f)
        assert len(events) > 1
        total_content = "".join(e.raw_content for e in events)
        assert len(total_content) >= len(content) - 100  # allow minor whitespace diff

    def test_content_hash_is_set(self, tmp_path: Path) -> None:
        f = tmp_path / "hashed.txt"
        f.write_text("unique content here")
        parser = PlainTextParser()
        events = parser.parse(f)
        assert len(events) == 1
        assert events[0].content_hash != ""
        assert len(events[0].content_hash) == 16  # SHA-256 truncated to 16 hex chars


class TestMarkdownParser:
    def test_implements_protocol(self) -> None:
        parser = MarkdownParser()
        assert isinstance(parser, ImportParser)
        assert parser.name == "markdown"

    def test_can_parse_md(self, tmp_path: Path) -> None:
        parser = MarkdownParser()
        assert parser.can_parse(tmp_path / "notes.md")
        assert parser.can_parse(tmp_path / "readme.markdown")
        assert not parser.can_parse(tmp_path / "notes.txt")
        assert not parser.can_parse(tmp_path / "data.json")

    def test_simple_note_without_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "simple.md"
        f.write_text("Just a paragraph of text.\n\nAnother paragraph.")
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 1
        assert events[0].source == "obsidian"
        assert events[0].event_type == "note"
        assert "Just a paragraph" in events[0].raw_content
        assert events[0].metadata["title"] == "simple"

    def test_splits_on_headings(self, tmp_path: Path) -> None:
        content = (
            "# Introduction\n\nSome intro text.\n\n"
            "# Methods\n\nMethodology here.\n\n"
            "# Results\n\nThe results.\n"
        )
        f = tmp_path / "paper.md"
        f.write_text(content)
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 3
        assert "Introduction" in events[0].metadata["title"]
        assert "Methods" in events[1].metadata["title"]
        assert "Results" in events[2].metadata["title"]
        assert "Some intro text" in events[0].raw_content
        assert "Methodology here" in events[1].raw_content

    def test_preamble_before_first_heading(self, tmp_path: Path) -> None:
        content = "Preamble text.\n\n# First Section\n\nSection content."
        f = tmp_path / "preamble.md"
        f.write_text(content)
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 2
        assert "Preamble text" in events[0].raw_content
        assert events[0].metadata["title"] == "preamble"
        assert "First Section" in events[1].metadata["title"]

    def test_frontmatter_extraction(self, tmp_path: Path) -> None:
        content = (
            "---\ntitle: My Note\ntags: python, memory\n---\n\n"
            "Body content here."
        )
        f = tmp_path / "noted.md"
        f.write_text(content)
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 1
        assert events[0].metadata["title"] == "My Note"
        assert events[0].metadata["frontmatter"]["tags"] == "python, memory"
        assert "---" not in events[0].raw_content

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("  \n\n  ")
        parser = MarkdownParser()
        assert parser.parse(f) == []

    def test_h2_headings_also_split(self, tmp_path: Path) -> None:
        content = "## Part A\n\nAlpha.\n\n## Part B\n\nBeta."
        f = tmp_path / "parts.md"
        f.write_text(content)
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 2

    def test_h3_headings_do_not_split(self, tmp_path: Path) -> None:
        content = "# Top\n\nIntro.\n\n### Sub-sub\n\nDetail."
        f = tmp_path / "deep.md"
        f.write_text(content)
        parser = MarkdownParser()
        events = parser.parse(f)
        assert len(events) == 1
        assert "### Sub-sub" in events[0].raw_content


class TestChatGPTParser:
    def test_implements_protocol(self) -> None:
        parser = ChatGPTParser()
        assert isinstance(parser, ImportParser)
        assert parser.name == "chatgpt"

    def test_can_parse_chatgpt_json(self, tmp_path: Path) -> None:
        parser = ChatGPTParser()
        good = tmp_path / "conversations.json"
        good.write_text(json.dumps([_chatgpt_conversation()]))
        assert parser.can_parse(good)

    def test_rejects_non_chatgpt_json(self, tmp_path: Path) -> None:
        parser = ChatGPTParser()
        plain_json = tmp_path / "config.json"
        plain_json.write_text(json.dumps({"key": "value"}))
        assert not parser.can_parse(plain_json)

        txt = tmp_path / "notes.txt"
        txt.write_text("not json")
        assert not parser.can_parse(txt)

    def test_rejects_empty_array(self, tmp_path: Path) -> None:
        parser = ChatGPTParser()
        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert not parser.can_parse(f)

    def test_parse_single_conversation(self, tmp_path: Path) -> None:
        conv = _chatgpt_conversation(
            title="Debug CORS",
            user_msg="How do I fix CORS?",
            assistant_msg="Add the Access-Control-Allow-Origin header.",
        )
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps([conv]))
        parser = ChatGPTParser()
        events = parser.parse(f)
        assert len(events) == 1
        e = events[0]
        assert e.source == "chatgpt"
        assert e.event_type == "conversation"
        assert "Debug CORS" in e.raw_content
        assert "How do I fix CORS?" in e.raw_content
        assert "Access-Control-Allow-Origin" in e.raw_content
        assert e.metadata["title"] == "Debug CORS"
        assert e.metadata["message_count"] == 2
        assert e.metadata["model"] == "gpt-4"

    def test_parse_multiple_conversations(self, tmp_path: Path) -> None:
        convs = [
            _chatgpt_conversation(title="Chat 1", conv_id="c1"),
            _chatgpt_conversation(title="Chat 2", conv_id="c2"),
            _chatgpt_conversation(title="Chat 3", conv_id="c3"),
        ]
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps(convs))
        parser = ChatGPTParser()
        events = parser.parse(f)
        assert len(events) == 3
        titles = {e.metadata["title"] for e in events}
        assert titles == {"Chat 1", "Chat 2", "Chat 3"}

    def test_skips_conversations_without_messages(self, tmp_path: Path) -> None:
        conv = {
            "id": "empty",
            "title": "Empty",
            "create_time": 0,
            "mapping": {
                "root": {"id": "root", "parent": None, "message": None},
            },
        }
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps([conv]))
        parser = ChatGPTParser()
        events = parser.parse(f)
        assert events == []

    def test_timestamp_extracted(self, tmp_path: Path) -> None:
        conv = _chatgpt_conversation(create_time=1709913600.0)
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps([conv]))
        parser = ChatGPTParser()
        events = parser.parse(f)
        assert events[0].timestamp.year == 2024
        assert events[0].timestamp.month == 3

    def test_model_detection(self, tmp_path: Path) -> None:
        conv = _chatgpt_conversation(model_slug="gpt-4o")
        f = tmp_path / "conversations.json"
        f.write_text(json.dumps([conv]))
        parser = ChatGPTParser()
        events = parser.parse(f)
        assert events[0].metadata["model"] == "gpt-4o"
