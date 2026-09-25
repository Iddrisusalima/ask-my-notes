"""Embedder selection, driven by the configured provider."""

from __future__ import annotations

from askmydocs.config import Configuration
from askmydocs.embeddings.base import Embedder
from askmydocs.embeddings.retry import RetryPolicy


def build_embedder(
    configuration: Configuration, retry_policy: RetryPolicy | None = None
) -> Embedder:
    """Return the embedder for ``configuration.provider``.

    Each provider module is imported inside its branch, so selecting the local
    provider needs no ``openai`` install and selecting ``openai`` needs no
    PyTorch.
    """
    if configuration.provider == "openai":
        from askmydocs.embeddings.openai_provider import OpenAIEmbedder

        return OpenAIEmbedder(configuration, retry_policy)

    from askmydocs.embeddings.local_provider import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(configuration, retry_policy)
