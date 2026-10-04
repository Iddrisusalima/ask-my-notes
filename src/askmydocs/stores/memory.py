"""In-memory store: a linear scan, which is exactly the point.

Phase 1 ranks by comparing the query against every stored vector. That is O(n)
and obviously correct, which makes it the reference implementation Phase 2's
Chroma store is checked against.

``add`` validates everything before mutating anything, so the atomicity the
requirements demand is structural rather than a rule to remember.
"""

from __future__ import annotations

from collections.abc import Sequence

from askmydocs.errors import (
    BatchLengthMismatchError,
    DegenerateVectorError,
    InvalidKError,
    VectorLengthError,
)
from askmydocs.models import Chunk, SearchHit, StoredRecord
from askmydocs.similarity import ZERO_NORM_THRESHOLD, cosine_similarity, euclidean_norm
from askmydocs.stores.base import VectorStoreInterface


class InMemoryStore(VectorStoreInterface):
    """Holds chunk text, metadata, and vectors in process memory."""

    def __init__(self) -> None:
        self._records: list[StoredRecord] = []
        self._dimensionality: int | None = None

    @property
    def dimensionality(self) -> int | None:
        return self._dimensionality

    def count(self) -> int:
        return len(self._records)

    def add(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise BatchLengthMismatchError(
                f"Batch supplies {len(chunks)} chunks but {len(embeddings)} "
                f"embeddings; the counts must match."
            )
        if not chunks:
            return  # Requirement 10.6: empty batch is a no-op.

        lengths = {len(vector) for vector in embeddings}
        if len(lengths) > 1:
            raise VectorLengthError(
                f"Batch mixes embedding lengths {sorted(lengths)}; every vector in "
                f"one batch must have the same length."
            )
        incoming = lengths.pop()
        if self._dimensionality is not None and incoming != self._dimensionality:
            raise VectorLengthError(
                f"Batch supplies vectors of length {incoming} but the store holds "
                f"vectors of length {self._dimensionality}."
            )

        # Build first, mutate last: steps below cannot fail.
        start = len(self._records)
        new_records = [
            StoredRecord(
                chunk=chunk,
                embedding=tuple(float(x) for x in vector),
                insertion_index=start + offset,
            )
            for offset, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]
        self._records.extend(new_records)
        self._dimensionality = incoming

    def query(self, embedding: Sequence[float], k: int) -> list[SearchHit]:
        if isinstance(k, bool) or not isinstance(k, int):
            raise InvalidKError(f"k is {k!r}; k must be an integer of at least 1.")
        if k < 1:
            raise InvalidKError(f"k is {k}; k must be at least 1.")
        if not self._records:
            return []  # Requirement 10.11, before any vector validation.

        if len(embedding) != self._dimensionality:
            raise VectorLengthError(
                f"Query vector has length {len(embedding)} but the store holds "
                f"vectors of length {self._dimensionality}."
            )
        if euclidean_norm(embedding) <= ZERO_NORM_THRESHOLD:
            raise DegenerateVectorError(
                "Query vector has a degenerate norm; similarity is undefined."
            )

        scored = [
            SearchHit(
                chunk=record.chunk,
                score=cosine_similarity(embedding, record.embedding),
                insertion_index=record.insertion_index,
            )
            for record in self._records
        ]
        # Descending score, ties by ascending insertion index (Req 10.9, 10.15).
        scored.sort(key=lambda hit: (-hit.score, hit.insertion_index))
        return scored[:k]
