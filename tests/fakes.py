"""Deterministic test doubles. No provider, no socket, no key.

``FakeEmbedder`` subclasses ``BaseEmbedder``, so tests exercise the real
validation, batching, and retry path rather than a parallel reimplementation.

Seeding is SHA-256, not ``hash()``: ``hash()`` is salted per process, so the
determinism assertions would pass or fail depending on which interpreter run
they landed in.
"""

from __future__ import annotations

import hashlib
import math

from askmydocs.config import Configuration
from askmydocs.embeddings.base import BaseEmbedder
from askmydocs.embeddings.retry import RetryPolicy

DEFAULT_FAKE_DIM = 16


class FakeEmbedder(BaseEmbedder):
    """Maps text to a unit vector derived from its SHA-256 digest."""

    def __init__(
        self,
        configuration: Configuration,
        dimensionality: int = DEFAULT_FAKE_DIM,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(
            configuration,
            retry_policy or RetryPolicy(configuration.max_retry_attempts, sleep=lambda _: None),
        )
        self._dim = dimensionality
        self.seen: list[list[str]] = []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        """A stable unit vector for ``text``."""
        raw: list[float] = []
        counter = 0
        while len(raw) < self._dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            raw.extend((byte - 127.5) / 127.5 for byte in digest)
            counter += 1
        values = raw[: self._dim]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class ScriptedProvider(BaseEmbedder):
    """Raises a scripted sequence of errors before delegating to a fake."""

    def __init__(
        self,
        configuration: Configuration,
        errors: list[Exception],
        dimensionality: int = DEFAULT_FAKE_DIM,
    ) -> None:
        super().__init__(
            configuration,
            RetryPolicy(configuration.max_retry_attempts, sleep=lambda _: None),
        )
        self._errors = list(errors)
        self._inner = FakeEmbedder(configuration, dimensionality)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._errors:
            raise self._errors.pop(0)
        return [self._inner._vector(text) for text in texts]
