"""Store selection. Phase 2 adds one branch here for Chroma."""

from __future__ import annotations

from askmydocs.config import Configuration
from askmydocs.stores.base import VectorStoreInterface
from askmydocs.stores.memory import InMemoryStore


def build_store(configuration: Configuration) -> VectorStoreInterface:
    """Return the configured store. Phase 1 has exactly one implementation."""
    return InMemoryStore()
