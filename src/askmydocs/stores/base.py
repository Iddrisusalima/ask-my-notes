"""The storage seam.

``VectorStoreInterface`` is the single abstraction Week 2 reimplements against
Chroma. It declares exactly the three operations Requirement 10.1 names, so a
persistent store is a drop-in replacement: no loader, chunker, or embedder
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from askmydocs.models import Chunk, SearchHit


class VectorStoreInterface(ABC):
    """Add a batch, report the count, return the top K by similarity."""

    @abstractmethod
    def add(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Add a batch, preserving each chunk-to-vector association (Req 10.4).

        An empty batch is a no-op and must not raise (Req 10.6). Validation is
        atomic: on any failure the stored items and the count are unchanged
        (Req 10.7, 10.8).
        """

    @abstractmethod
    def count(self) -> int:
        """Number of stored items (Req 10.1, 10.5)."""

    @abstractmethod
    def query(self, embedding: Sequence[float], k: int) -> list[SearchHit]:
        """Up to K chunks, descending similarity, ties by ascending insertion index.

        Requirements 10.9, 10.15. Fewer than K stored returns all of them
        (Req 10.10); an empty store returns an empty list (Req 10.11).
        """

    @property
    @abstractmethod
    def dimensionality(self) -> int | None:
        """Vector length fixed by the first non-empty add, else None."""
