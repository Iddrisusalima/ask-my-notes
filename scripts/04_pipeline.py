"""Phase 1 end-to-end pipeline: discover, load, chunk, embed, store, query.

A thin shell over library code. Every console string goes through the Reporter,
and the library raises typed exceptions rather than exiting, so this module owns
the only ``return`` of an exit status.

Usage:
    python scripts/04_pipeline.py
    python scripts/04_pipeline.py --fake     # deterministic embedder, no model download
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from askmydocs.chunking import Chunker
from askmydocs.config import Configuration, load_configuration
from askmydocs.errors import AskMyDocsError, GuardrailError
from askmydocs.loading.discovery import discover_notes
from askmydocs.loading.pipeline import load_documents
from askmydocs.models import Chunk
from askmydocs.reporting import Reporter
from askmydocs.stores.memory import InMemoryStore

DEMO_QUERY = "Why does chunk size affect the quality of retrieved context?"
DEMO_TOP_K = 3
PREVIEW_CHARS = 200


def build_pipeline_embedder(configuration: Configuration, use_fake: bool):
    """The real provider, or the deterministic fake for a no-download run."""
    if use_fake:
        from tests.fakes import FakeEmbedder

        return FakeEmbedder(configuration)
    from askmydocs.embeddings.factory import build_embedder

    return build_embedder(configuration)


def run(configuration: Configuration, reporter: Reporter, embedder) -> int:
    """Run the pipeline. Returns the exit status."""
    started = time.monotonic()

    discovery = discover_notes(configuration.notes_folder)
    reporter.info(
        f"Discovered {len(discovery.files)} note file(s) in "
        f"{configuration.notes_folder}: {discovery.counts_by_extension or '{}'}"
    )
    if discovery.skipped:
        reporter.info(f"Skipped {len(discovery.skipped)} entr(ies).")
    if len(discovery.files) < 5:
        reporter.warning(
            f"Found {len(discovery.files)} note file(s); 5 to 10 is the recommended "
            f"range for a useful corpus."
        )

    documents = load_documents(discovery, reporter)
    chunker = Chunker(configuration.chunk_size, configuration.chunk_overlap)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))

    reporter.info(f"Loaded {len(documents)} document(s), produced {len(chunks)} chunk(s).")

    if not chunks:
        reporter.info("No chunks to embed. Nothing to do.")
        reporter.info("Documents: 0  Chunks: 0  Store items: 0")
        return 0

    # Guardrail before the first request, so tripping it costs nothing.
    if len(chunks) > configuration.max_chunks_per_run:
        raise GuardrailError(
            f"This run would embed {len(chunks)} chunks, above the configured "
            f"maximum of {configuration.max_chunks_per_run}. Raise "
            f"ASKMYDOCS_MAX_CHUNKS_PER_RUN or narrow the notes folder."
        )

    store = InMemoryStore()
    batch = configuration.max_batch_size
    embedded = 0
    for start in range(0, len(chunks), batch):
        segment = chunks[start : start + batch]
        vectors = embedder.embed_texts([chunk.text for chunk in segment])
        store.add(segment, vectors)
        embedded += len(segment)
        reporter.progress(embedded, len(chunks))

    elapsed = time.monotonic() - started
    reporter.info("")
    reporter.info(
        f"Documents: {len(documents)}  Chunks: {len(chunks)}  "
        f"Dimensionality: {embedder.dimensionality}  "
        f"Store items: {store.count()}  "
        f"Embedder calls: {getattr(embedder, 'batch_call_count', 'n/a')}  "
        f"Elapsed: {elapsed:.1f}s"
    )

    reporter.info("")
    reporter.info(f'Demonstration query: "{DEMO_QUERY}"')
    hits = store.query(embedder.embed_text(DEMO_QUERY), DEMO_TOP_K)
    if not hits:
        reporter.info("No chunks were available for the demonstration query.")
    for rank, hit in enumerate(hits, start=1):
        preview = hit.chunk.text[:PREVIEW_CHARS].replace("\n", " ")
        suffix = "..." if len(hit.chunk.text) > PREVIEW_CHARS else ""
        reporter.info(
            f"  {rank}. {hit.chunk.source_path} "
            f"[chunk {hit.chunk.index}, chars {hit.chunk.start_offset}-"
            f"{hit.chunk.end_offset}] score {hit.score:.4f}"
        )
        reporter.info(f"     {preview}{suffix}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    use_fake = "--fake" in argv

    try:
        configuration = load_configuration()
    except AskMyDocsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(api_key=configuration.api_key)
    try:
        embedder = build_pipeline_embedder(configuration, use_fake)
        return run(configuration, reporter, embedder)
    except GuardrailError as exc:
        reporter.error(str(exc))
        return 4
    except AskMyDocsError as exc:
        reporter.error(str(exc))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
