"""Provider-independent embedder: validation, batching, retry, dimensionality.

``embed_text`` is *defined* as ``embed_texts([text])[0]``. That makes the
requirement "a batch result at position i equals the single-text result"
structural rather than a behaviour someone has to maintain (Requirement 3.3).

Concrete providers implement only ``_embed_batch``: one request, one segment.
Everything else -- rejecting empty input, enforcing the input length cap,
splitting into segments, retrying, checking the dimensionality never changes --
happens once, here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from askmydocs.config import Configuration
from askmydocs.errors import EmptyInputError, InputTooLongError
from askmydocs.embeddings.retry import RetryPolicy


class Embedder(Protocol):
    """What the rest of the project depends on. No store type appears here."""

    @property
    def dimensionality(self) -> int | None: ...

    def embed_text(self, text: str) -> list[float]: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class BaseEmbedder(ABC):
    """Shared embedder behaviour."""

    def __init__(
        self,
        configuration: Configuration,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._configuration = configuration
        self._retry = retry_policy or RetryPolicy(configuration.max_retry_attempts)
        self._dimensionality: int | None = None
        self._batch_call_count = 0

    @property
    def dimensionality(self) -> int | None:
        """Discovered from the first successful response, then held constant."""
        return self._dimensionality

    @property
    def batch_call_count(self) -> int:
        """Provider requests issued. Reported by the pipeline (Req 11.2, 11.6)."""
        return self._batch_call_count

    @property
    def model_name(self) -> str:
        return self._configuration.model_name

    def embed_text(self, text: str) -> list[float]:
        """One text. Defined through embed_texts so the two cannot diverge."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in input order.

        Validates the whole input before issuing any request, so a bad element
        costs nothing (Requirements 2.6, 2.8, 3.6). Returns ``[]`` for an empty
        sequence without a request (Requirement 3.4).
        """
        if not texts:
            return []

        limit = self._configuration.max_input_length
        for position, text in enumerate(texts):
            if not text or not text.strip():
                raise EmptyInputError(
                    f"Text at position {position} is empty or whitespace-only."
                )
            if len(text) > limit:
                raise InputTooLongError(
                    f"Text at position {position} is {len(text)} code points, above "
                    f"the configured maximum of {limit}."
                )

        size = self._configuration.max_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            segment = list(texts[start : start + size])
            end = start + len(segment) - 1
            result = self._retry.run(
                lambda seg=segment: self._invoke(seg),
                f"embedding texts {start}-{end} with {self.model_name}",
            )
            vectors.extend(result)
        return vectors

    def _invoke(self, segment: list[str]) -> list[list[float]]:
        """One provider request, with the dimensionality check applied."""
        self._batch_call_count += 1
        vectors = self._embed_batch(segment)

        if len(vectors) != len(segment):
            raise ValueError(
                f"Provider returned {len(vectors)} vectors for {len(segment)} texts."
            )
        for vector in vectors:
            if self._dimensionality is None:
                self._dimensionality = len(vector)
            elif len(vector) != self._dimensionality:
                raise ValueError(
                    f"Provider returned a vector of length {len(vector)} but earlier "
                    f"vectors had length {self._dimensionality}."
                )
        return vectors

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Issue exactly one provider request for this segment."""
