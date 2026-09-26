"""Chroma-backed store: the same interface, persisted to disk.

This is the only module in the project that imports ``chromadb``. Everything
else talks to ``VectorStoreInterface``, which is what made swapping the storage
layer a one-module change.

Two deliberate choices:

``embedding_function=None``. Chroma's default embedding function downloads a
model on first use and would embed text through it, giving a second embedding
path competing with our own and a silent dimensionality mismatch. Passing None
means Chroma raises if we ever forget to supply a vector, turning a possible
silent-wrong-answer bug into a loud error.

``hnsw:space`` set explicitly to cosine. Chroma's default is L2. Cosine compares
direction only, which is what makes a returned score equal the cosine similarity
the Similarity_Calculator computes, so the relevance threshold means one thing
everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from askmydocs.errors import (
    BatchLengthMismatchError,
    DegenerateVectorError,
    InvalidKError,
    StoreError,
    VectorLengthError,
)
from askmydocs.models import Chunk, SearchHit
from askmydocs.similarity import ZERO_NORM_THRESHOLD, euclidean_norm
from askmydocs.stores.base import VectorStoreInterface

#: Below this stored count a query asks for every item, which forces hnswlib's
#: beam wide enough to be exhaustive, so the top-K is exact.
EXACT_SEARCH_LIMIT: Final = 1000

#: Chroma writes are capped by SQLite's bound-variable limit in practice.
MAX_WRITE_BATCH: Final = 2000


class ChromaStore(VectorStoreInterface):
    """A persistent collection behind the Week 1 store interface."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        distance_metric: str = "cosine",
        model_name: str = "",
        dimensionality: int | None = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise StoreError(
                "chromadb is not installed. Install it with "
                "'python -m pip install chromadb==1.1.1'."
            ) from exc

        persist_directory.mkdir(parents=True, exist_ok=True)
        self._persist_directory = persist_directory
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "hnsw:space": distance_metric,
                "askmydocs_metric": distance_metric,
                "askmydocs_model": model_name,
            },
        )
        self._dimensionality = dimensionality

    @property
    def dimensionality(self) -> int | None:
        return self._dimensionality

    def count(self) -> int:
        return int(self._collection.count())

    def _next_insertion_index(self) -> int:
        return self.count()

    def add(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Add a batch. Delegates to upsert semantics for chunk ids."""
        self.upsert(chunks, embeddings)

    def upsert(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Insert or replace by chunk id, validating before writing anything."""
        if len(chunks) != len(embeddings):
            raise BatchLengthMismatchError(
                f"Batch supplies {len(chunks)} chunks but {len(embeddings)} "
                f"embeddings; the counts must match."
            )
        if not chunks:
            return

        lengths = {len(vector) for vector in embeddings}
        if len(lengths) > 1:
            raise VectorLengthError(
                f"Batch mixes embedding lengths {sorted(lengths)}."
            )
        incoming = lengths.pop()
        if self._dimensionality is not None and incoming != self._dimensionality:
            raise VectorLengthError(
                f"Batch supplies vectors of length {incoming} but the collection "
                f"holds vectors of length {self._dimensionality}."
            )

        base = self._next_insertion_index()
        ids, documents, metadatas, vectors = [], [], [], []
        for offset, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            metadata = chunk.to_metadata()
            metadata["insertion_index"] = base + offset
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(metadata)
            vectors.append([float(x) for x in vector])

        for start in range(0, len(ids), MAX_WRITE_BATCH):
            stop = start + MAX_WRITE_BATCH
            self._collection.upsert(
                ids=ids[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
                embeddings=vectors[start:stop],
            )
        self._dimensionality = incoming

    def delete_by_source_path(self, source_path: str) -> int:
        """Remove every chunk of one source file. Returns the number removed."""
        existing = self._collection.get(where={"source_path": source_path})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=list(ids))
        return len(ids)

    def reset(self) -> None:
        """Empty the collection, keeping it configured."""
        existing = self._collection.get()
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=list(ids))

    def query(self, embedding: Sequence[float], k: int) -> list[SearchHit]:
        """Top K by similarity, descending, ties by ascending insertion index."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise InvalidKError(f"k is {k!r}; k must be an integer of at least 1.")
        if k < 1:
            raise InvalidKError(f"k is {k}; k must be at least 1.")

        total = self.count()
        if total == 0:
            return []

        if self._dimensionality is not None and len(embedding) != self._dimensionality:
            raise VectorLengthError(
                f"Query vector has length {len(embedding)} but the collection holds "
                f"vectors of length {self._dimensionality}."
            )
        if euclidean_norm(embedding) <= ZERO_NORM_THRESHOLD:
            raise DegenerateVectorError(
                "Query vector has a degenerate norm; similarity is undefined."
            )

        # Asking for everything below the limit forces an exhaustive scan, which
        # is what makes the top-K exact and lets the tie-break rule apply.
        wanted = total if total <= EXACT_SEARCH_LIMIT else min(total, k + 50)
        response = self._collection.query(
            query_embeddings=[[float(x) for x in embedding]],
            n_results=wanted,
            include=["documents", "metadatas", "distances"],
        )

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # Chroma's cosine space returns 1 - cosine_similarity of the
            # L2-normalized vectors, so this inverts to our own score exactly,
            # independent of the input vector norms.
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            insertion_index = int(metadata.get("insertion_index", 0))
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        text=text,
                        source_path=str(metadata.get("source_path", "")),
                        index=int(metadata.get("index", 0)),
                        start_offset=int(metadata.get("start_offset", 0)),
                        end_offset=int(metadata.get("end_offset", 0)),
                    ),
                    score=score,
                    insertion_index=insertion_index,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.insertion_index))
        return hits[:k]
