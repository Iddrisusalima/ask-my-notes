"""Citation extraction, validation, and the source list.

A marker the model invented is the one failure mode that makes a wrong answer
look verifiable: a reader who trusts the source list stops checking. So a
fabricated number does not get quietly dropped -- it marks the whole answer
unverified.

An answer with no markers at all is also unverified. It may be correct, but
nothing about it demonstrates it came from the notes rather than from recall.
"""

from __future__ import annotations

from dataclasses import dataclass

from askmydocs.generation.prompting import (
    CITATION_MARKER_PATTERN,
    AssembledPrompt,
    CitationEntry,
)


@dataclass(frozen=True)
class CitationReport:
    """What the answer cited, and whether it can be trusted."""

    markers: tuple[int, ...]       # document order, duplicates kept
    cited: tuple[int, ...]         # ascending, distinct
    dangling: tuple[int, ...]      # cited but absent from the table
    verified: bool

    @property
    def has_citations(self) -> bool:
        return bool(self.markers)


class CitationValidator:
    """Checks every marker against the table the prompt actually supplied."""

    def extract_markers(self, answer: str) -> tuple[int, ...]:
        """Every [n] in document order, duplicates preserved."""
        return tuple(int(m.group(1)) for m in CITATION_MARKER_PATTERN.finditer(answer))

    def validate(self, answer: str, prompt: AssembledPrompt) -> CitationReport:
        markers = self.extract_markers(answer)
        cited = tuple(sorted(set(markers)))
        supplied = prompt.numbers()
        dangling = tuple(n for n in cited if n not in supplied)
        return CitationReport(
            markers=markers,
            cited=cited,
            dangling=dangling,
            verified=bool(markers) and not dangling,
        )


def build_source_list(
    report: CitationReport, prompt: AssembledPrompt
) -> tuple[CitationEntry, ...]:
    """The cited, resolvable entries in ascending number order.

    Uncited entries are omitted: listing a source the answer never used invites
    a reader to attribute a claim to it.
    """
    by_number = {entry.number: entry for entry in prompt.citations}
    return tuple(by_number[n] for n in report.cited if n in by_number)
