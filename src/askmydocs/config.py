"""Configuration, read from the environment and validated once.

The environment mapping is injected rather than read from ``os.environ``
directly, which makes every validation rule testable as a pure function with no
process-state patching and no risk of a real API key leaking into a test run.

Validation runs in a fixed order so error messages are predictable:

1. provider identifier
2. model default for that provider
3. strict integer parsing of every numeric setting
4. range check of every numeric setting
5. cross-check that the overlap is below the chunk size
6. API key requirement, last

Step 6 is last deliberately. A learner with a broken chunk size and no key gets
the chunk-size message, rather than being sent hunting for a key they may not
even need.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from askmydocs.errors import ConfigurationError

SUPPORTED_PROVIDERS: Final = ("openai", "sentence-transformers")
REDACTION_MARKER: Final = "***REDACTED***"

#: Shortest API key substring that is redacted. Provider errors sometimes echo a
#: truncated key, so whole-string replacement alone is not enough.
MIN_REDACTED_RUN: Final = 8

DEFAULT_MODELS: Final = {
    "openai": "text-embedding-3-small",
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
}

#: setting name -> (env var, default, minimum, maximum)
NUMERIC_SETTINGS: Final = {
    "chunk_size": ("ASKMYDOCS_CHUNK_SIZE", 500, 1, 10000),
    "chunk_overlap": ("ASKMYDOCS_CHUNK_OVERLAP", 50, 0, 9999),
    "request_timeout_seconds": ("ASKMYDOCS_REQUEST_TIMEOUT", 30, 1, 300),
    "max_retry_attempts": ("ASKMYDOCS_MAX_RETRY_ATTEMPTS", 3, 0, 10),
    "max_input_length": ("ASKMYDOCS_MAX_INPUT_LENGTH", 8000, 1, 100000),
    "max_batch_size": ("ASKMYDOCS_MAX_BATCH_SIZE", 64, 1, 2048),
    "max_chunks_per_run": ("ASKMYDOCS_MAX_CHUNKS_PER_RUN", 2000, 1, 1000000),
}

API_KEY_VARIABLE: Final = "OPENAI_API_KEY"


@dataclass(frozen=True)
class Configuration:
    """Every runtime setting, already validated."""

    provider: str
    model_name: str
    api_key: str | None
    notes_folder: Path
    chunk_size: int
    chunk_overlap: int
    request_timeout_seconds: int
    max_retry_attempts: int
    max_input_length: int
    max_batch_size: int
    max_chunks_per_run: int

    @property
    def stride(self) -> int:
        """chunk_size - chunk_overlap. Always at least 1."""
        return self.chunk_size - self.chunk_overlap


def _read(env: Mapping[str, str], name: str) -> str | None:
    """Return a trimmed value, treating absent and blank as identical."""
    raw = env.get(name)
    if raw is None:
        return None
    trimmed = raw.strip()
    return trimmed or None


def _parse_int(env: Mapping[str, str], variable: str, default: int) -> int:
    """Parse strictly: reject '500.0', '5e2', '1_000', 'abc'. Requirement 1.11."""
    raw = _read(env, variable)
    if raw is None:
        return default
    if not (raw.lstrip("+-").isdigit()):
        raise ConfigurationError(
            f"{variable} is {raw!r}, which is not an integer. "
            f"Expected a whole number such as {default}."
        )
    return int(raw)


def load_configuration(env: Mapping[str, str] | None = None) -> Configuration:
    """Read, default, normalize, and validate every setting.

    Raises:
        ConfigurationError: on any violation of Requirements 1.3, 1.5, 1.8,
            1.9, 1.11, or 1.12.
    """
    env = os.environ if env is None else env

    # 1. Provider, matched case-insensitively after trimming (Req 1.10).
    raw_provider = _read(env, "ASKMYDOCS_PROVIDER")
    provider = (raw_provider or "sentence-transformers").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"ASKMYDOCS_PROVIDER is {raw_provider!r}. "
            f"Supported values are {SUPPORTED_PROVIDERS[0]!r} and "
            f"{SUPPORTED_PROVIDERS[1]!r}."
        )

    # 2. Model default depends on the provider (Req 1.2).
    model_name = _read(env, "ASKMYDOCS_MODEL") or DEFAULT_MODELS[provider]

    # 3 and 4. Parse then range-check every numeric setting.
    values: dict[str, int] = {}
    for setting, (variable, default, minimum, maximum) in NUMERIC_SETTINGS.items():
        value = _parse_int(env, variable, default)
        if setting == "chunk_overlap":
            continue  # range-checked against chunk_size below
        if not (minimum <= value <= maximum):
            raise ConfigurationError(
                f"{variable} is {value}; the permitted range is {minimum} to "
                f"{maximum} inclusive."
            )
        values[setting] = value

    # 5. The overlap is bounded by the chunk size, not by a constant (Req 1.9).
    chunk_size = values["chunk_size"]
    overlap_variable, overlap_default, _, _ = NUMERIC_SETTINGS["chunk_overlap"]
    chunk_overlap = _parse_int(env, overlap_variable, overlap_default)
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ConfigurationError(
            f"{overlap_variable} is {chunk_overlap} and ASKMYDOCS_CHUNK_SIZE is "
            f"{chunk_size}; the permitted range for the overlap is 0 to "
            f"{chunk_size - 1} inclusive, so that the stride stays positive."
        )

    notes_folder = Path(_read(env, "ASKMYDOCS_NOTES_FOLDER") or "sample-notes")

    # 6. The API key, last, and only when the provider needs it (Req 1.3, 1.4).
    api_key: str | None = None
    if provider == "openai":
        api_key = _read(env, API_KEY_VARIABLE)
        if api_key is None:
            raise ConfigurationError(
                f"{API_KEY_VARIABLE} is not set, and provider {provider!r} requires "
                f"it. Either set it, or use the local provider by setting "
                f"ASKMYDOCS_PROVIDER=sentence-transformers."
            )

    return Configuration(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        notes_folder=notes_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **{k: v for k, v in values.items() if k != "chunk_size"},
    )


def redact(text: str, api_key: str | None) -> str:
    """Replace the API key, and any run of it at least 8 characters long.

    Requirement 1.7. Returns ``text`` unchanged when there is no key, or when
    the key is too short to redact meaningfully.
    """
    if not api_key or len(api_key) < MIN_REDACTED_RUN:
        return text
    result = text.replace(api_key, REDACTION_MARKER)
    # Longest runs first, so a long match is not broken up by a shorter one.
    for length in range(len(api_key), MIN_REDACTED_RUN - 1, -1):
        for start in range(0, len(api_key) - length + 1):
            run = api_key[start : start + length]
            if run in result:
                result = result.replace(run, REDACTION_MARKER)
    return result


# --- Week 2 additions -------------------------------------------------------
# Appended, never interleaved: every Week 1 field, name, and default above is
# untouched, so existing construction sites and tests stay valid.

SUPPORTED_STORES: Final = ("memory", "chroma")
SUPPORTED_METRICS: Final = ("cosine",)


@dataclass(frozen=True)
class StoreSettings:
    """Persistent vector store settings."""

    selection: str = "chroma"
    persist_directory: Path = Path(".chroma")
    collection_name: str = "ask_my_docs"
    distance_metric: str = "cosine"
    source_manifest: Path = Path(".chroma/ingest-manifest.json")


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval and logging settings."""

    top_k: int = 5
    relevance_threshold: float = 0.30
    retrieval_log: Path = Path("logs/retrievals.jsonl")
    question_set: Path = Path("question-sets/retrieval-questions.txt")


def _repo_root() -> Path:
    """Anchor relative paths on the package, not the working directory."""
    return Path(__file__).resolve().parents[2]


def _resolve(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else (_repo_root() / path)


def load_store_settings(env: Mapping[str, str] | None = None) -> StoreSettings:
    """Read the Week 2 store settings, validated."""
    env = os.environ if env is None else env

    raw_store = _read(env, "ASKMYDOCS_STORE")
    selection = (raw_store or "chroma").lower()
    if selection not in SUPPORTED_STORES:
        raise ConfigurationError(
            f"ASKMYDOCS_STORE is {raw_store!r}. Supported values are "
            f"{SUPPORTED_STORES[0]!r} and {SUPPORTED_STORES[1]!r}."
        )

    raw_metric = _read(env, "ASKMYDOCS_DISTANCE_METRIC")
    metric = (raw_metric or "cosine").lower()
    if metric not in SUPPORTED_METRICS:
        raise ConfigurationError(
            f"ASKMYDOCS_DISTANCE_METRIC is {raw_metric!r}. Week 2 supports only "
            f"'cosine', so that a score means the same thing as the "
            f"Similarity_Calculator's cosine similarity."
        )

    name = _read(env, "ASKMYDOCS_COLLECTION") or "ask_my_docs"
    if not (3 <= len(name) <= 63):
        raise ConfigurationError(
            f"ASKMYDOCS_COLLECTION is {name!r}; the permitted length is 3 to 63 "
            f"characters."
        )
    if not (name[0].isalnum() and name[-1].isalnum()):
        raise ConfigurationError(
            f"ASKMYDOCS_COLLECTION is {name!r}; it must start and end with an "
            f"ASCII letter or digit."
        )
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in name):
        raise ConfigurationError(
            f"ASKMYDOCS_COLLECTION is {name!r}; only ASCII letters, digits, hyphens, "
            f"and underscores are permitted."
        )

    return StoreSettings(
        selection=selection,
        persist_directory=_resolve(_read(env, "ASKMYDOCS_PERSIST_DIR"), Path(".chroma")),
        collection_name=name,
        distance_metric=metric,
        source_manifest=_resolve(
            _read(env, "ASKMYDOCS_SOURCE_MANIFEST"), Path(".chroma/ingest-manifest.json")
        ),
    )


def load_retrieval_settings(env: Mapping[str, str] | None = None) -> RetrievalSettings:
    """Read the Week 2 retrieval settings, validated."""
    env = os.environ if env is None else env

    top_k = _parse_int(env, "ASKMYDOCS_TOP_K", 5)
    if not (1 <= top_k <= 100):
        raise ConfigurationError(
            f"ASKMYDOCS_TOP_K is {top_k}; the permitted range is 1 to 100."
        )

    raw_threshold = _read(env, "ASKMYDOCS_RELEVANCE_THRESHOLD")
    try:
        threshold = float(raw_threshold) if raw_threshold else 0.30
    except ValueError as exc:
        raise ConfigurationError(
            f"ASKMYDOCS_RELEVANCE_THRESHOLD is {raw_threshold!r}, which is not a "
            f"decimal number."
        ) from exc
    if not (-1.0 <= threshold <= 1.0):
        raise ConfigurationError(
            f"ASKMYDOCS_RELEVANCE_THRESHOLD is {threshold}; the permitted range is "
            f"-1.0 to 1.0, because it is compared against a cosine similarity."
        )

    return RetrievalSettings(
        top_k=top_k,
        relevance_threshold=threshold,
        retrieval_log=_resolve(
            _read(env, "ASKMYDOCS_RETRIEVAL_LOG"), Path("logs/retrievals.jsonl")
        ),
        question_set=_resolve(
            _read(env, "ASKMYDOCS_QUESTION_SET"), Path("question-sets/retrieval-questions.txt")
        ),
    )
