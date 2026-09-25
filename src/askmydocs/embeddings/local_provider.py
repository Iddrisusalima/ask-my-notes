"""Local embedding provider: a Sentence Transformers model, no API key.

The model loads lazily, on the first batch, so constructing the embedder is free
and a run that embeds nothing pays no model-loading cost.
"""

from __future__ import annotations

from askmydocs.config import Configuration
from askmydocs.embeddings.base import BaseEmbedder
from askmydocs.embeddings.retry import RetryPolicy
from askmydocs.errors import EmbeddingError


class SentenceTransformersEmbedder(BaseEmbedder):
    """Runs a local model in-process. Reads no API key and opens no socket
    once the model files are cached (Requirement 1.4)."""

    def __init__(
        self,
        configuration: Configuration,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(configuration, retry_policy)
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install it with "
                    "'python -m pip install sentence-transformers==5.1.0', or switch "
                    "provider with ASKMYDOCS_PROVIDER=openai."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [[float(x) for x in vector] for vector in vectors]
