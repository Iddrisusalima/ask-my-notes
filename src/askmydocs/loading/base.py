"""Shared loading helpers: size limit, decoding, newline normalization.

Newline normalization happens here and nowhere else. Requirement 7.10 requires
chunk offsets to index the returned ``Document.text`` directly, so if the
chunker also transformed text the offsets would refer to a string nobody holds.
One normalization point, one source of truth.
"""

from __future__ import annotations

from typing import Final, Protocol

from askmydocs.models import Document
from askmydocs.reporting import Reporter

#: Files above this size are skipped with a warning. Requirement 7.12.
MAX_FILE_BYTES: Final = 25 * 1024 * 1024

#: Extensions loaded, lower-cased, without the leading period.
SUPPORTED_EXTENSIONS: Final = ("pdf", "md", "markdown")

#: U+FFFD, substituted for each undecodable byte sequence.
REPLACEMENT_CHARACTER: Final = "\ufffd"


def normalize_newlines(text: str) -> str:
    """Convert CRLF and lone CR to a single LF, and change nothing else.

    No trimming, no whitespace collapsing, no Unicode normalization
    (Requirement 7.10).
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_utf8(data: bytes) -> tuple[str, int]:
    """Decode as UTF-8, dropping a leading BOM, and count any replacements.

    Returns the text and the number of replacement characters introduced.
    A clean decode returns a count of 0 (Requirements 7.4, 7.5).
    """
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8"), 0
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        return text, text.count(REPLACEMENT_CHARACTER)


class DocumentLoader(Protocol):
    """One loader per supported format."""

    def load(self, path, relative_path: str, reporter: Reporter) -> Document:
        """Return a Document, or raise a DocumentLoadError subclass."""
        ...
