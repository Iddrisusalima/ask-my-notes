# Technology and configuration

## Runtime and packaging

- Python 3.10 through 3.12.
- `src/askmydocs/` layout, installed editable: `pip install -e .`
- Every dependency version is pinned exactly in `pyproject.toml`. No open ranges.

## Allowed third-party libraries

| Library | Used for |
|---|---|
| `openai` | embeddings endpoint and chat completions only |
| `sentence-transformers` | local embedding model (default provider) |
| `chromadb` | local persistent vector store |
| `pypdf` | PDF page text extraction |
| `numpy` | float64 vector arithmetic |
| `python-dotenv` | loading `.env` |
| `pytest`, `hypothesis` | tests |

## Not allowed

RAG or agent orchestration frameworks - LangChain, LlamaIndex, Haystack, or any package supplying
pre-built loading, chunking, retrieval, re-ranking, or prompt-orchestration pipelines - and hosted
vector database clients. If a task seems to need one, the task is wrong.

## Configuration

Configuration is environment variables only, read through `load_configuration`, with documented
defaults. Week 1 variables carry their exact names; Week 2 settings are named by the design's
setting name, and `.env.example` plus the README are the source of truth for both.

| Setting | Variable / setting name | Default |
|---|---|---|
| Embedding provider | `ASKMYDOCS_PROVIDER` | `sentence-transformers` |
| Model name | `ASKMYDOCS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` (local); `text-embedding-3-small` (openai) |
| API key | `OPENAI_API_KEY` | none - required only for the `openai` provider |
| Notes folder | `ASKMYDOCS_NOTES_FOLDER` | `sample-notes` |
| Chunk size | `ASKMYDOCS_CHUNK_SIZE` | `500` |
| Chunk overlap | `ASKMYDOCS_CHUNK_OVERLAP` | `50` |
| Request timeout | `ASKMYDOCS_REQUEST_TIMEOUT` | `30` seconds |
| Max retry attempts | `ASKMYDOCS_MAX_RETRY_ATTEMPTS` | `3` |
| Max input length | `ASKMYDOCS_MAX_INPUT_LENGTH` | `8000` |
| Max batch size | `ASKMYDOCS_MAX_BATCH_SIZE` | `64` |
| Max chunks per run | `ASKMYDOCS_MAX_CHUNKS_PER_RUN` | `2000` |
| Embedding dimensionality | `ASKMYDOCS_EMBEDDING_DIM` | unset - resolved offline for known models |
| Store selection | store selection | `chroma` |
| Persist directory | `Persist_Directory` | `.chroma` |
| Collection name | `Collection_Name` | `ask_my_docs` |
| Distance metric | `Distance_Metric` | `cosine` |
| Top K | `Top_K` | `5` |
| Relevance threshold | `Relevance_Threshold` | `0.30` |
| Retrieval log | `Retrieval_Log` | `logs/retrievals.jsonl` |
| Source manifest | `Source_Manifest` | `.chroma/ingest-manifest.json` |
| Question set | `Question_Set` | `question-sets/week2-questions.txt` |

## Commands

```powershell
pip install -e .                         # install, editable
python -m pytest -q                      # whole test suite

python scripts/01_embed_one.py           # embed one sentence
python scripts/02_compare_sentences.py   # compare sentence similarity
python scripts/03_chunking_experiment.py # chunking experiment
python scripts/04_pipeline.py            # Week 1 in-memory pipeline
python scripts/05_ingest.py              # incremental ingest into Chroma
python scripts/06_query.py "question"    # retrieve for a question
python scripts/07_topk_experiment.py     # top-K tuning experiment
python scripts/08_relevance_review.py    # relevance review (generate / score)
```

## Shell

The development shell is PowerShell on Windows. Chain commands with `;`, not `&&`. Do not start
long-running watchers from a tool call.
