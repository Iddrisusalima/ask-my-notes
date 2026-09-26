"""Incremental ingest into the persistent store.

Idempotent by design. A manifest records a SHA-256 hash per source file, so a
second run over unchanged notes embeds nothing and costs nothing. A changed file
has its old chunks deleted before the new ones are written -- without that, a
file that shrinks leaves orphan chunks behind that still answer queries.

Usage:
    python scripts/05_ingest.py
    python scripts/05_ingest.py --reset    # rebuild from scratch
    python scripts/05_ingest.py --fake     # deterministic embedder, no model load
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from askmydocs.chunking import Chunker
from askmydocs.config import load_configuration, load_store_settings
from askmydocs.errors import AskMyDocsError, GuardrailError
from askmydocs.loading.discovery import discover_notes
from askmydocs.loading.pipeline import load_documents
from askmydocs.reporting import Reporter
from askmydocs.stores.chroma_store import ChromaStore


def content_hash(path: Path) -> str:
    """SHA-256 of the file bytes, from one sequential read."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AskMyDocsError(
            f"The manifest at {path} is not valid JSON ({exc}). Delete it and "
            f"re-run with --reset to rebuild."
        ) from exc


def save_manifest(path: Path, manifest: dict) -> None:
    """Write atomically: a temp file in the same directory, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    use_fake = "--fake" in argv
    do_reset = "--reset" in argv
    started = time.monotonic()

    try:
        configuration = load_configuration()
        store_settings = load_store_settings()
    except AskMyDocsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(api_key=configuration.api_key)

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
        manifest = {} if do_reset else load_manifest(store_settings.source_manifest)
        if do_reset:
            store.reset()
            reporter.info("Reset: the collection and manifest were emptied.")

        discovery = discover_notes(configuration.notes_folder)
        documents = load_documents(discovery, reporter)
        chunker = Chunker(configuration.chunk_size, configuration.chunk_overlap)

        seen: set[str] = set()
        new_count = changed_count = skipped_count = 0
        planned: list[tuple[str, list]] = []

        for document in documents:
            seen.add(document.source_path)
            absolute = configuration.notes_folder / document.source_path
            digest = content_hash(absolute)
            entry = manifest.get(document.source_path)
            unchanged = (
                entry is not None
                and entry.get("hash") == digest
                and entry.get("chunk_size") == configuration.chunk_size
                and entry.get("chunk_overlap") == configuration.chunk_overlap
                and entry.get("model") == configuration.model_name
            )
            if unchanged:
                skipped_count += 1
                continue
            if entry is None:
                new_count += 1
            else:
                changed_count += 1
            planned.append((document.source_path, chunker.chunk_document(document)))
            manifest[document.source_path] = {
                "hash": digest,
                "chunk_size": configuration.chunk_size,
                "chunk_overlap": configuration.chunk_overlap,
                "model": configuration.model_name,
                "chunks": len(planned[-1][1]),
            }

        deleted_sources = [path for path in list(manifest) if path not in seen]
        for path in deleted_sources:
            store.delete_by_source_path(path)
            manifest.pop(path, None)

        total_chunks = sum(len(chunks) for _, chunks in planned)
        if total_chunks > configuration.max_chunks_per_run:
            raise GuardrailError(
                f"This run would embed {total_chunks} chunks, above the configured "
                f"maximum of {configuration.max_chunks_per_run}."
            )

        embedded = 0
        for source_path, chunks in planned:
            store.delete_by_source_path(source_path)  # delete before write
            batch = configuration.max_batch_size
            for start in range(0, len(chunks), batch):
                segment = chunks[start : start + batch]
                vectors = embedder.embed_texts([c.text for c in segment])
                store.upsert(segment, vectors)
                embedded += len(segment)
                reporter.progress(embedded, total_chunks)
            save_manifest(store_settings.source_manifest, manifest)

        elapsed = time.monotonic() - started
        reporter.info("")
        reporter.info(
            f"New: {new_count}  Changed: {changed_count}  Unchanged: {skipped_count}  "
            f"Deleted: {len(deleted_sources)}"
        )
        reporter.info(
            f"Chunks embedded: {embedded}  Collection items: {store.count()}  "
            f"Embedder calls: {getattr(embedder, 'batch_call_count', 'n/a')}  "
            f"Elapsed: {elapsed:.1f}s"
        )
        reporter.info(f"Collection: {store_settings.persist_directory}")
        return 0
    except GuardrailError as exc:
        reporter.error(str(exc))
        return 4
    except AskMyDocsError as exc:
        reporter.error(str(exc))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
