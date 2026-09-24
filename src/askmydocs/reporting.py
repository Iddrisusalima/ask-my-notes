"""The single console output channel.

Every outgoing string passes through :func:`~askmydocs.config.redact` before it
reaches a stream. That is the whole reason this class exists: Requirement 1.7
applies to "any error message, log record, or console output", and one choke
point is verifiable whereas scattered ``str.replace`` calls are not.
"""

from __future__ import annotations

import sys
from typing import TextIO

from askmydocs.config import redact


class Reporter:
    """Writes progress, warnings, and errors, redacting the API key."""

    def __init__(
        self,
        api_key: str | None = None,
        stream: TextIO | None = None,
        error_stream: TextIO | None = None,
    ) -> None:
        self._api_key = api_key
        self._stream = stream if stream is not None else sys.stdout
        self._error_stream = error_stream if error_stream is not None else sys.stderr

    def _clean(self, message: str) -> str:
        return redact(message, self._api_key)

    def info(self, message: str) -> None:
        """Ordinary progress output."""
        print(self._clean(message), file=self._stream)

    def warning(self, message: str) -> None:
        """A recoverable problem. Requirements 6.4, 7.5, 7.8, 7.12."""
        print(f"WARNING: {self._clean(message)}", file=self._error_stream)

    def error(self, message: str) -> None:
        """A failure that skipped work. Requirements 7.7, 7.11."""
        print(f"ERROR: {self._clean(message)}", file=self._error_stream)

    def progress(self, done: int, total: int) -> None:
        """Embedding progress. Requirement 11.7."""
        print(f"  embedded {done}/{total} chunks", file=self._stream)
