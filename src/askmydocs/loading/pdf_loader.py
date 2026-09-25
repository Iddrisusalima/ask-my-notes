"""PDF loading: per-page text joined in page order.

Exactly one line feed between consecutive pages and nothing else inserted
(Requirement 7.1). Any decoration -- a page number, a separator rule -- would
appear in a chunk as text the source file does not contain, and would shift
every offset after it.
"""

from __future__ import annotations

from pathlib import Path

from askmydocs.errors import DocumentLoadError, EncryptedPdfError
from askmydocs.loading.base import normalize_newlines
from askmydocs.models import Document
from askmydocs.reporting import Reporter


class PdfLoader:
    """Extracts text page by page with pypdf."""

    def load(self, path: Path, relative_path: str, reporter: Reporter) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise DocumentLoadError(f"pypdf is not installed: {exc}") from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise DocumentLoadError(
                f"{relative_path} could not be opened as a PDF: {exc}"
            ) from exc

        if getattr(reader, "is_encrypted", False):
            raise EncryptedPdfError(
                f"{relative_path} is encrypted or password-protected, so its text "
                f"cannot be extracted."
            )

        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                reporter.warning(
                    f"{relative_path} page {number} could not be extracted: {exc}"
                )
                pages.append("")

        return Document(
            text=normalize_newlines("\n".join(pages)),
            source_path=relative_path,
            file_type="pdf",
        )
