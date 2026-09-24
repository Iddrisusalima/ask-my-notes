# Repository structure and conventions

```
pyproject.toml            pinned dependencies, editable install
README.md
.env.example
sample-notes/             note corpus; only README.md is tracked
question-sets/            one question per line
learning-notes/           the learner's written explanations
reports/                  topk-experiment.md, relevance-review*.csv
logs/                     retrievals.jsonl
scripts/                  01_embed_one .. 08_relevance_review
src/askmydocs/
  config.py  reporting.py  similarity.py  chunking.py  models.py  errors.py
  loading/      base.py, pdf_loader.py, markdown_loader.py
  embeddings/   base.py and the provider implementations
  stores/       base.py, admin.py, memory_store.py, chroma_store.py, factory.py
  retrieval/    retriever, result types, retrieval log writer
  ingest/       manifest, change detection, dimensionality
  evaluation/   question sets, top-K experiment, precision@K
tests/
```

## Module responsibilities

- **config** — reads every setting from the environment, validates it, resolves paths.
- **reporting** — the single output channel; redacts the API key.
- **similarity** — cosine similarity over float64 vectors.
- **chunking** — splits text into overlapping chunks with code point offsets.
- **loading** — discovers note files and extracts text per format.
- **embeddings** — provider-agnostic embedder interface plus implementations.
- **stores** — vector storage behind one interface; in-memory and Chroma.
- **retrieval** — embeds a question, ranks hits, applies the threshold, logs the retrieval.
- **ingest** — incremental chunk-embed-upsert driven by a source manifest.
- **evaluation** — question sets, top-K tuning, relevance review, precision@K.

## The architectural seam

`Vector_Store_Interface` in `stores/base.py` is the only storage abstraction. `InMemoryStore` and
`ChromaStore` both implement it. The chunker and the embedder never import a store type, and
`stores/chroma_store.py` is the only module that imports `chromadb`.

## Frozen modules

`chunking.py`, `similarity.py`, `loading/base.py`, `loading/pdf_loader.py`,
`loading/markdown_loader.py`, everything under `embeddings/`, and `models.py` are byte-identical
from the end of Week 1 onward. Extend them by adding a new module or a subclass, never by editing
them. `config.py` and `stores/factory.py` are the only pre-existing modules later weeks may change,
and only additively.

## Conventions

- Library code raises typed exceptions and never calls `sys.exit`.
- Scripts are thin `main(argv) -> int` shells over library code.
- Data models are frozen dataclasses.
- Chunk offsets are Unicode code point counts, start inclusive and end exclusive.
- Every console string passes through the `Reporter`.
