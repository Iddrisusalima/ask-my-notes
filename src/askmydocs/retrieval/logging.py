"""Append-only retrieval log, one JSON object per line.

Holds verbatim note text, which is why ``.gitignore`` excludes it. Every line is
redacted before the write, so a field added later cannot leak a key by being
forgotten at the call site.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from askmydocs.config import redact
from askmydocs.errors import AskMyDocsError

LOG_SCHEMA_VERSION = 1
TEXT_PREVIEW_CHARS = 500


class RetrievalLogError(AskMyDocsError):
    """The retrieval log could not be written."""


class RetrievalLogWriter:
    """Appends one line per retrieval."""

    def __init__(self, path: Path, api_key: str | None = None) -> None:
        self._path = path
        self._api_key = api_key

    def append(self, result, model_name: str, dimensionality: int | None) -> None:
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": result.question,
            "top_k": result.top_k,
            "model": model_name,
            "dimensionality": dimensionality,
            "threshold": result.threshold,
            "collection_count": result.collection_count,
            "outcome": result.outcome.value,
            "top_score": result.top_score,
            "hit_count": len(result.hits),
            "hits": [
                {
                    "rank": rank,
                    "chunk_id": scored.chunk.chunk_id,
                    "source_path": scored.chunk.source_path,
                    "index": scored.chunk.index,
                    "start_offset": scored.chunk.start_offset,
                    "end_offset": scored.chunk.end_offset,
                    "score": scored.score,
                    "below_threshold": scored.below_threshold,
                    "text_length": len(scored.chunk.text),
                    "text": scored.chunk.text[:TEXT_PREVIEW_CHARS]
                    + ("..." if len(scored.chunk.text) > TEXT_PREVIEW_CHARS else ""),
                }
                for rank, scored in enumerate(result.hits, start=1)
            ],
        }
        line = redact(json.dumps(record, ensure_ascii=False), self._api_key) + "\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(
                self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            try:
                os.write(handle, line.encode("utf-8"))
            finally:
                os.close(handle)
        except OSError as exc:
            raise RetrievalLogError(
                f"The retrieval log at {self._path} could not be written: {exc}"
            ) from exc
