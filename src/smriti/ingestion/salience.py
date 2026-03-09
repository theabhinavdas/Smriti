"""Two-stage salience filter: fast heuristic scoring + optional LLM batch scoring.

Stage 1 (heuristic): Rule-based per-source scoring. Runs synchronously, no LLM.
  - 0.0 = pure noise (drop)
  - 0.2-0.7 = uncertain (pass to Stage 2)
  - 0.8-1.0 = always keep

Stage 2 (LLM): Batched scoring via the model provider. Only called for events
that land in the uncertain range from Stage 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from smriti.models.events import SourceEvent
from smriti.provider import ModelProvider

logger = logging.getLogger(__name__)

DROP_THRESHOLD = 0.2
KEEP_THRESHOLD = 0.7


@dataclass
class ScoredEvent:
    """A source event annotated with a salience score."""

    event: SourceEvent
    score: float

    @property
    def is_salient(self) -> bool:
        return self.score >= DROP_THRESHOLD


class SalienceFilter:
    def __init__(self, provider: ModelProvider | None = None) -> None:
        self._provider = provider

    # ------------------------------------------------------------------
    # Stage 1: Heuristic (no LLM)
    # ------------------------------------------------------------------

    def heuristic_score(self, event: SourceEvent) -> float:
        source = event.source
        if source == "terminal":
            return self._score_terminal(event)
        if source == "browser":
            return self._score_browser(event)
        if source == "cursor":
            return self._score_cursor(event)
        return 0.3

    @staticmethod
    def _score_terminal(event: SourceEvent) -> float:
        cmd = event.raw_content.strip()
        exit_code = event.metadata.get("exit_code")
        if exit_code is not None and exit_code != 0:
            return 1.0
        if cmd.startswith(("git commit", "git push", "git merge")):
            return 0.9
        if cmd.startswith(("ls", "cd", "pwd", "clear", "echo")):
            return 0.0
        if any(kw in cmd for kw in ("install", "build", "deploy", "docker")):
            return 0.8
        return 0.3

    @staticmethod
    def _score_browser(event: SourceEvent) -> float:
        etype = event.event_type
        if etype == "text_selected":
            return 0.9
        if etype == "search":
            return 0.8
        dwell = event.metadata.get("dwell_seconds", 0)
        if dwell < 5:
            return 0.0
        if dwell > 30:
            return 0.6
        return 0.3

    @staticmethod
    def _score_cursor(event: SourceEvent) -> float:
        etype = event.event_type
        if etype in ("file_created", "file_deleted", "agent_transcript"):
            return 0.9
        if etype == "file_edited":
            return 0.7
        if etype == "diagnostic_change":
            return 0.6
        return 0.3

    # ------------------------------------------------------------------
    # Stage 2: LLM batch scoring
    # ------------------------------------------------------------------

    async def llm_score(
        self, events: list[ScoredEvent], user_context: str = ""
    ) -> list[ScoredEvent]:
        """Re-score uncertain events via an LLM. Requires a model provider."""
        if not self._provider or not events:
            return events

        event_descriptions = []
        for i, se in enumerate(events):
            event_descriptions.append(
                f"{i}. [{se.event.source}/{se.event.event_type}] {se.event.raw_content[:200]}"
            )

        prompt = (
            "Rate the salience of each event on a scale of 0-10, where 0 is noise "
            "and 10 is critically important for the user's long-term memory. "
            "Return a JSON array of integers, one per event.\n\n"
        )
        if user_context:
            prompt += f"User context: {user_context}\n\n"
        prompt += "Events:\n" + "\n".join(event_descriptions)

        try:
            raw = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            scores = json.loads(raw)
            for i, llm_score in enumerate(scores):
                if i < len(events):
                    events[i].score = float(llm_score) / 10.0
        except Exception:
            logger.warning("LLM salience scoring failed, keeping heuristic scores", exc_info=True)

        return events

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    async def filter(self, events: list[SourceEvent]) -> list[ScoredEvent]:
        """Run both stages. Returns only salient events."""
        scored: list[ScoredEvent] = []
        uncertain: list[ScoredEvent] = []

        for event in events:
            score = self.heuristic_score(event)
            se = ScoredEvent(event=event, score=score)
            if score >= KEEP_THRESHOLD:
                scored.append(se)
            elif score >= DROP_THRESHOLD:
                uncertain.append(se)

        if uncertain:
            uncertain = await self.llm_score(uncertain)
            scored.extend(se for se in uncertain if se.is_salient)

        return scored
