"""Markdown loading: the file's full text, with all markup preserved."""

from __future__ import annotations

from pathlib import Path

from askmydocs.loading.base import decode_utf8, normalize_newlines
from askmydocs.models import Document
from askmydocs.reporting import Reporter


class MarkdownLoader:
    """Returns the whole file. Markdown is already the text we want to chunk.

    No markup stripping: a heading is a useful retrieval signal, and removing it
    would shift every offset away from the bytes on disk (Requirement 7.2).
    """

    def load(self, path: Path, relative_path: str, reporter: Reporter) -> Document:
        text, replacements = decode_utf8(path.read_bytes())
        if replacements:
            reporter.warning(
                f"{relative_path} is not valid UTF-8; {replacements} byte "
                f"sequence(s) were replaced."
            )
        return Document(
            text=normalize_newlines(text),
            source_path=relative_path,
            file_type=path.suffix.lower().lstrip("."),
        )
