"""OpenAI embedding provider.

Vectors are re-sorted by the response ``index`` field, so ordering never depends
on the order the API happened to return them in (Requirement 3.2).
"""

from __future__ import annotations

from askmydocs.config import Configuration
from askmydocs.embeddings.base import BaseEmbedder
from askmydocs.embeddings.retry import RetryPolicy
from askmydocs.errors import EmbeddingError


class OpenAIEmbedder(BaseEmbedder):
    """One request per segment against the embeddings endpoint."""

    def __init__(
        self,
        configuration: Configuration,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(configuration, retry_policy)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingError(
                    "openai is not installed. Install it with "
                    "'python -m pip install openai==1.109.1'."
                ) from exc
            self._client = OpenAI(
                api_key=self._configuration.api_key,
                timeout=self._configuration.request_timeout_seconds,
            )
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._get_client().embeddings.create(
            model=self.model_name, input=texts
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(x) for x in item.embedding] for item in ordered]
