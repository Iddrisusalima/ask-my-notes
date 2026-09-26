"""Rendering answers and refusals as lines.

Returns lines rather than printing, so every string still passes through the
Reporter and its redaction, and so the output is testable by comparing tuples.
"""

from __future__ import annotations

from askmydocs.retrieval.retriever import RetrievalOutcome

INGEST_COMMAND = "python scripts/05_ingest.py"


def render_answer(answer: str, report, sources, prompt, budget: int) -> list[str]:
    """Warnings first, then the answer, then the numbered sources."""
    lines: list[str] = []

    if report.dangling:
        listed = ", ".join(f"[{n}]" for n in report.dangling)
        lines.append(
            f"WARNING: the answer cites {listed}, which was not supplied as "
            f"context. Treat this answer as unverified."
        )
    elif not report.has_citations:
        lines.append(
            "WARNING: the answer carries no citations, so it cannot be traced "
            "back to your notes. Treat it as unverified."
        )

    if prompt.dropped_chunk_count:
        lines.append(
            f"Note: {prompt.dropped_chunk_count} lower-scoring chunk(s) were "
            f"dropped to fit the context budget of {budget} characters."
        )

    lines.append("")
    lines.append(answer.strip())
    lines.append("")

    if sources:
        lines.append("Sources:")
        for entry in sources:
            lines.append(
                f"  [{entry.number}] {entry.source_path} "
                f"chunk {entry.chunk_index}, chars {entry.start_offset}-"
                f"{entry.end_offset}  score {entry.score:.4f}"
            )
    else:
        lines.append("Sources: none cited.")

    lines.append("")
    lines.append(f"Verified: {'yes' if report.verified else 'no'}")
    return lines


def render_refusal(result) -> list[str]:
    """No source list, ever: there is nothing to attribute."""
    lines = ["", "I don't know based on the provided notes.", ""]

    if result.outcome is RetrievalOutcome.EMPTY_COLLECTION:
        lines.append(
            "Reason: the collection is empty, so there is nothing indexed to "
            "search."
        )
        lines.append(f"Run '{INGEST_COMMAND}' to index your notes.")
    else:
        lines.append(
            f"Reason: the closest chunk scored {result.top_score:.4f}, below the "
            f"relevance threshold of {result.threshold:.2f}."
        )
        lines.append(
            "Either your notes do not cover this, or the question is phrased "
            "differently from how the notes put it."
        )
    return lines
