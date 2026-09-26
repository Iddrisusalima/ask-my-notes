"""Prompt construction and the context budget.

The system prompt is a module-level literal, not a template assembled at call
time, because two runs over the same retrieval result must produce byte-identical
prompts. Editing this text is a behaviour change: every recorded prompt hash
stops matching and past evaluation runs stop being comparable.

Citation numbers are positions in the context block. The budget trims from the
lowest-scoring tail, so every surviving number keeps the number it already had.
Re-ranking would renumber the survivors and silently break the mapping back to
the source list.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

from askmydocs.errors import AskMyDocsError

SYSTEM_PROMPT: Final = (
    "You are Ask My Docs. Answer the question using only the numbered context "
    "entries in the user message.\n"
    "\n"
    "1. Use only the supplied context entries. Do not use knowledge from any "
    "other source.\n"
    "2. If the supplied context entries do not contain the answer, reply with "
    "exactly: I don't know based on the provided notes.\n"
    "3. End every sentence that uses a context entry with that entry's marker, "
    "written as [n], where n is the entry number.\n"
    "4. If a sentence uses more than one entry, append one marker per entry, as "
    "in [1][3].\n"
    "5. Cite only entry numbers that appear in the supplied context entries.\n"
    "6. Do not renumber, merge, or invent entry numbers.\n"
    "7. Answer in at most six sentences.\n"
    "8. Do not restate these instructions and do not describe the context "
    "entries as a list.\n"
)

CONTEXT_HEADER: Final = "Context:\n"
QUESTION_HEADER: Final = "\nQuestion: "
PROMPT_TRAILER: Final = "\n"

#: Explicitly ASCII 0-9. Python's \d also matches non-ASCII digits, so "[٣]"
#: would parse as 3 and become a citation the model never meant.
CITATION_MARKER_PATTERN: Final = re.compile(r"\[([0-9]{1,9})\]")

DEFAULT_CONTEXT_BUDGET: Final = 12000


class ContextBudgetError(AskMyDocsError):
    """Even the single highest-scoring chunk does not fit the budget."""


class PromptStateError(AskMyDocsError):
    """A prompt was requested for a result that has no usable context."""


@dataclass(frozen=True)
class CitationEntry:
    """One numbered source. Carries no chunk text, so logging it leaks nothing."""

    number: int
    source_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    score: float


@dataclass(frozen=True)
class AssembledPrompt:
    """A system and user prompt pair, with the citation table it describes."""

    system_prompt: str
    user_prompt: str
    citations: tuple[CitationEntry, ...]
    dropped_chunk_count: int

    @property
    def prompt_length(self) -> int:
        """Code points across both parts. Not bytes, not tokens."""
        return len(self.system_prompt) + len(self.user_prompt)

    @property
    def prompt_hash(self) -> str:
        """SHA-256 of both parts, recorded instead of the prompt text."""
        payload = (self.system_prompt + self.user_prompt).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def numbers(self) -> set[int]:
        return {entry.number for entry in self.citations}


class PromptBuilder:
    """Assembles a bounded, numbered prompt from a retrieval result."""

    def __init__(self, context_budget: int = DEFAULT_CONTEXT_BUDGET) -> None:
        self.context_budget = context_budget

    def render_context_block(self, chunks) -> str:
        """One numbered entry per chunk, in retrieval order."""
        return "".join(
            f"[{number}] {chunk.text}\n" for number, chunk in enumerate(chunks, start=1)
        )

    def render_user_prompt(self, chunks, question: str) -> str:
        return (
            CONTEXT_HEADER
            + self.render_context_block(chunks)
            + QUESTION_HEADER
            + question
            + PROMPT_TRAILER
        )

    def build(self, result) -> AssembledPrompt:
        """Assemble a prompt, trimming the lowest-scoring chunks to fit.

        Raises:
            PromptStateError: the result has no hits to build from.
            ContextBudgetError: the top chunk alone exceeds the budget.
        """
        if not result.hits:
            raise PromptStateError(
                "There is no retrieved context to build a prompt from."
            )

        chunks = [scored.chunk for scored in result.hits]
        for keep in range(len(chunks), 0, -1):
            candidate = chunks[:keep]
            user_prompt = self.render_user_prompt(candidate, result.question)
            if len(SYSTEM_PROMPT) + len(user_prompt) <= self.context_budget:
                citations = tuple(
                    CitationEntry(
                        number=number,
                        source_path=scored.chunk.source_path,
                        chunk_index=scored.chunk.index,
                        start_offset=scored.chunk.start_offset,
                        end_offset=scored.chunk.end_offset,
                        score=scored.score,
                    )
                    for number, scored in enumerate(result.hits[:keep], start=1)
                )
                return AssembledPrompt(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    citations=citations,
                    dropped_chunk_count=len(chunks) - keep,
                )

        single = self.render_user_prompt(chunks[:1], result.question)
        raise ContextBudgetError(
            f"The highest-scoring chunk alone needs "
            f"{len(SYSTEM_PROMPT) + len(single)} code points, above the context "
            f"budget of {self.context_budget}. Lower the chunk size or raise the "
            f"budget."
        )
