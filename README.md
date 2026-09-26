# Ask My Docs

A from-scratch Retrieval-Augmented Generation tool over your own PDF and markdown notes.
It splits documents into overlapping chunks, embeds each chunk, stores them in a local
vector database, retrieves the chunks most similar to your question, and answers using
only that retrieved context - with citations back to the source file and character range.
Every stage is hand-written: no LangChain, no LlamaIndex, no framework that hides a step.

The committed `sample-notes/` corpus is six short markdown notes on embeddings, chunking,
vector databases, the RAG pipeline, RAG versus fine-tuning, and retrieval evaluation, so
you can clone this repository and run it immediately.

## Status

Week 1 of three is partially built. The pure layers are done and tested: configuration,
cosine similarity, the chunker, and the in-memory vector store behind the interface that
Week 2 swaps a persistent Chroma store into. 19 property-based tests pass, each running
200 generated examples.

Not yet built: document loaders, the embedder, the runnable scripts, and Weeks 2 and 3
(persistent storage, retrieval, generation, citations).

## Setup

Requires Python 3.10 to 3.12.

```powershell
git clone <your-fork-url> ask-my-docs
cd ask-my-docs
python -m pip install -e .
Copy-Item .env.example .env     # then edit if you want to change a default
```

`sentence-transformers` pulls in PyTorch, roughly 2 GB. To work on the pure layers and
run the test suite without it:

```powershell
python -m pip install -e . --no-deps
python -m pip install numpy==2.1.3 pypdf==6.1.3 python-dotenv==1.1.1 pytest==8.4.2 hypothesis==6.140.3
```

## Running the tests

```powershell
python -m pytest -q
```

19 property tests, about 15 seconds. The suite needs no API key and opens no socket: a
session fixture patches `socket` so an accidental network call fails loudly instead of
quietly reaching a provider.

## Adding your own documents

`sample-notes/` is the shareable corpus and is committed. Keep your own notes out of it:

```powershell
mkdir my-notes            # git-ignored
# copy your PDFs and markdown files in, then point the tool at them:
$env:ASKMYDOCS_NOTES_FOLDER = "my-notes"
```

Supported extensions are `.pdf`, `.md`, and `.markdown`, matched case-insensitively and
found recursively.

## Configuration

Every setting is an environment variable with a documented default. See `.env.example`
for the full list with ranges. The ones you are most likely to change:

| Variable | Default | What it controls |
|---|---|---|
| `ASKMYDOCS_PROVIDER` | `sentence-transformers` | Embedding provider. The local model is the default, so the paid path is always an explicit choice. |
| `ASKMYDOCS_NOTES_FOLDER` | `sample-notes` | Where the notes live. |
| `ASKMYDOCS_CHUNK_SIZE` | `500` | Chunk length in Unicode code points. |
| `ASKMYDOCS_CHUNK_OVERLAP` | `50` | Code points repeated from the previous chunk. |
| `OPENAI_API_KEY` | none | Required only for the `openai` provider. Never logged, never committed. |

## How it is built

Specifications live in `.kiro/specs/`, one per week, each with requirements, a design, and
a task list. `.kiro/steering/` holds the rules every task follows, including the frozen
modules Weeks 2 and 3 must not edit. `.kiro/hooks/`, `.kiro/agents/`, and `.kiro/skills/`
hold the automation, the scoped agents, and the workflow guides.

## Licence

Not yet chosen.
