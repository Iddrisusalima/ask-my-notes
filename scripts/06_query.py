"""Retrieve for one question and print the hits with their sources.

Prints one outcome line whose wording differs across the three cases, so the
difference between "nothing relevant" and "nothing indexed" is visible rather
than inferred.

Usage:
    python scripts/06_query.py "why does chunk size matter?"
    python scripts/06_query.py "..." --top-k 3
    python scripts/06_query.py "..." --fake
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from askmydocs.config import (
    load_configuration,
    load_retrieval_settings,
    load_store_settings,
)
from askmydocs.errors import AskMyDocsError
from askmydocs.reporting import Reporter
from askmydocs.retrieval.logging import RetrievalLogWriter
from askmydocs.retrieval.retriever import RetrievalOutcome, Retriever
from askmydocs.stores.chroma_store import ChromaStore

PREVIEW_CHARS = 240


def parse_args(argv: list[str]) -> tuple[str, int | None, bool]:
    use_fake = "--fake" in argv
    argv = [a for a in argv if a != "--fake"]
    top_k: int | None = None
    if "--top-k" in argv:
        position = argv.index("--top-k")
        top_k = int(argv[position + 1])
        del argv[position : position + 2]
    question = " ".join(argv).strip()
    return question, top_k, use_fake


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question, requested_k, use_fake = parse_args(argv)

    if not question:
        print(
            'Usage: python scripts/06_query.py "your question" [--top-k N] [--fake]',
            file=sys.stderr,
        )
        return 2

    try:
        configuration = load_configuration()
        store_settings = load_store_settings()
        retrieval_settings = load_retrieval_settings()
    except AskMyDocsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(api_key=configuration.api_key)
    top_k = requested_k or retrieval_settings.top_k

    try:
        if use_fake:
            from tests.fakes import FakeEmbedder

            embedder = FakeEmbedder(configuration)
        else:
            from askmydocs.embeddings.factory import build_embedder

            embedder = build_embedder(configuration)

        store = ChromaStore(
            store_settings.persist_directory,
            store_settings.collection_name,
            store_settings.distance_metric,
            model_name=configuration.model_name,
        )
        retriever = Retriever(
            store, embedder, retrieval_settings.relevance_threshold,
            configuration.max_input_length,
        )
        result = retriever.retrieve(question, top_k)

        RetrievalLogWriter(
            retrieval_settings.retrieval_log, configuration.api_key
        ).append(result, configuration.model_name, embedder.dimensionality)

        reporter.info(f'Question: "{question}"')
        reporter.info("")

        if result.outcome is RetrievalOutcome.EMPTY_COLLECTION:
            reporter.info(
                "The collection is empty, so there is nothing to search. Run "
                "'python scripts/05_ingest.py' to index your notes."
            )
            return 0

        if result.outcome is RetrievalOutcome.NO_RELEVANT_CONTEXT:
            reporter.info(
                f"No relevant context found. The best score was "
                f"{result.top_score:.4f}, below the relevance threshold of "
                f"{result.threshold:.2f}, so the notes do not appear to answer this."
            )
            reporter.info("Closest matches, for reference:")
        else:
            reporter.info(
                f"Relevant context found. Best score {result.top_score:.4f} meets the "
                f"threshold of {result.threshold:.2f}."
            )

        for rank, scored in enumerate(result.hits, start=1):
            flag = "  (below threshold)" if scored.below_threshold else ""
            preview = scored.chunk.text[:PREVIEW_CHARS].replace("\n", " ")
            suffix = "..." if len(scored.chunk.text) > PREVIEW_CHARS else ""
            reporter.info("")
            reporter.info(
                f"  [{rank}] {scored.chunk.source_path} "
                f"chunk {scored.chunk.index}, chars "
                f"{scored.chunk.start_offset}-{scored.chunk.end_offset}  "
                f"score {scored.score:.4f}{flag}"
            )
            reporter.info(f"      {preview}{suffix}")
        return 0
    except AskMyDocsError as exc:
        reporter.error(str(exc))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
