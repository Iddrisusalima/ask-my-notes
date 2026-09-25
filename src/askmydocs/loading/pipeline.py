"""Loading with per-file error isolation.

One unreadable file must not stop the run. Each failure is reported and the file
excluded, and loading continues (Requirement 7.7). The checks run cheapest
first: the size probe needs no parse, so an oversized file costs one stat call.
"""

from __future__ import annotations

from pathlib import Path

from askmydocs.errors import DocumentLoadError
from askmydocs.loading.base import MAX_FILE_BYTES
from askmydocs.loading.discovery import DiscoveryResult
from askmydocs.loading.markdown_loader import MarkdownLoader
from askmydocs.loading.pdf_loader import PdfLoader
from askmydocs.models import Document
from askmydocs.reporting import Reporter

_LOADERS = {
    "pdf": PdfLoader(),
    "md": MarkdownLoader(),
    "markdown": MarkdownLoader(),
}


def load_documents(discovery: DiscoveryResult, reporter: Reporter) -> list[Document]:
    """Load every discovered file, skipping and reporting the ones that fail.

    Returns only the documents that are safe to chunk: an empty-text document is
    excluded with a warning (Requirement 7.8).
    """
    documents: list[Document] = []

    for discovered in discovery.files:
        relative = discovered.relative_path
        try:
            size = discovered.absolute_path.stat().st_size
        except OSError as exc:
            reporter.error(f"{relative} could not be read: {exc}")
            continue

        if size > MAX_FILE_BYTES:
            reporter.warning(
                f"{relative} is {size} bytes, above the {MAX_FILE_BYTES} byte "
                f"limit, and was skipped."
            )
            continue

        loader = _LOADERS[discovered.extension]
        try:
            document = loader.load(discovered.absolute_path, relative, reporter)
        except DocumentLoadError as exc:
            reporter.error(str(exc))
            continue
        except Exception as exc:
            reporter.error(f"{relative} could not be loaded: {exc}")
            continue

        if not document.text.strip():
            reporter.warning(f"{relative} holds no extractable text, and was skipped.")
            continue

        documents.append(document)

    return documents
