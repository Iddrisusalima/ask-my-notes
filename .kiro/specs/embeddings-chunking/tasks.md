# Implementation Plan: Week 1 — Embeddings, Loading, and Chunking

## Overview

The plan builds the Week 1 layers bottom-up in Python 3.10–3.12, in the order the dependency
graph allows: repository scaffold → errors and models → configuration, reporting, redaction →
cosine similarity → chunker → discovery → loaders → store interface and `In_Memory_Store` →
embedder core and providers → the four runnable scripts → the layering guard → written notes and
README. Each layer lands with its tests before the next layer starts, so every step is validated
by code rather than by inspection.

Property-based tests use Hypothesis and follow the numbered Correctness Properties of the design
document. Every property test runs at least 100 examples (Requirement 13.7); the `pure` profile
runs 200 and the three filesystem-backed properties run 100. No test issues a network request:
the substitute embedder (`tests/fakes.py::FakeEmbedder`) and a session-scoped socket guard make
that structural rather than aspirational (Requirements 13.5, 13.6).

Build order is dependency-driven, not calendar-driven. The Week 1 day each epic serves is noted
on the epic and summarised in the day-mapping section at the end.

## Tasks

- [x] 1. Repository scaffold, dependency manifest, and test harness
  - Serves the Mon–Tue setup step. Nothing else can be imported or tested until this exists.

  - [x] 1.1 Create `pyproject.toml`
    - Pin an exact version for every direct dependency: `openai`, `sentence-transformers`,
      `pypdf`, `numpy`, `python-dotenv`, `pytest`, `hypothesis`
    - Declare `requires-python = ">=3.10,<3.13"` and the `src/` layout package discovery
    - Include no dependency that supplies pre-built loading, chunking, retrieval, or
      prompt-orchestration pipelines
    - Configure pytest: `testpaths = ["tests"]`, `--strict-markers`, `--durations=15`
    - _Requirements: 13.1, 13.2, 13.4, 13.8_

  - [x] 1.2 Create the package and directory skeleton
    - `src/askmydocs/__init__.py`, `src/askmydocs/loading/__init__.py`,
      `src/askmydocs/embeddings/__init__.py`, `src/askmydocs/stores/__init__.py`
    - Empty `scripts/`, `reports/`, `learning-notes/`, `sample-notes/`, `tests/` directories
    - _Requirements: 13.1, 13.3_

  - [x] 1.3 Create `.gitignore` and `.env.example`
    - `.gitignore`: `.env`, `.hypothesis/`, `__pycache__/`, and `sample-notes/*` with a
      negation for `sample-notes/README.md`
    - `.env.example`: every `ASKMYDOCS_*` variable plus `OPENAI_API_KEY=sk-your-key-here`,
      placeholder values only, no real secret
    - _Requirements: 1.6, 6.2_

  - [x] 1.4 Create `sample-notes/README.md`
    - State the expected count of 5 to 10 note files, the supported extensions `.pdf`, `.md`,
      `.markdown`, and that this README is not itself counted as a note file
    - _Requirements: 6.1, 6.2_

  - [x] 1.5 Create `tests/conftest.py`
    - Register the Hypothesis `pure` profile (200 examples, 500 ms deadline) and `filesystem`
      profile (100 examples, 3 s deadline, `too_slow` health check suppressed); load `pure`
      by default
    - Session-scoped autouse fixture that pops `OPENAI_API_KEY`, `OPENAI_ORG_ID`,
      `OPENAI_BASE_URL` and patches `socket.socket` and `socket.create_connection` to raise
    - _Requirements: 13.5, 13.6, 13.7_

  - [ ]* 1.6 Write `tests/test_repository.py` scaffold checks
    - Assert `.gitignore` covers `.env` and `sample-notes/*` while keeping the folder README
    - Assert `.env.example` contains no string matching a plausible API key shape
    - Assert `pyproject.toml` pins exact versions, declares 3.10–3.12, and lists no
      orchestration-framework dependency
    - Assert the network guard is active by attempting a connection and expecting `RuntimeError`
    - Assert `sample-notes/README.md` states the count range and the supported extensions
    - _Requirements: 1.6, 6.1, 6.2, 13.1, 13.2, 13.6_

- [x] 2. Exception hierarchy and data models

  - [x] 2.1 Implement `src/askmydocs/errors.py`
    - `AskMyDocsError` root plus the full hierarchy from the design: `ConfigurationError`,
      `NotesFolderError`, `DocumentLoadError` (`EncryptedPdfError`, `FileTooLargeError`,
      `EmptyDocumentError`), `ChunkerConfigError`, `SimilarityError`
      (`DimensionMismatchError`, `EmptyVectorError`, `NonFiniteValueError`, `ZeroNormError`),
      `EmbeddingError` (`EmptyInputError`, `InputTooLongError`, `EmbeddingFailedError`),
      `StoreError` (`BatchLengthMismatchError`, `VectorLengthError`, `DegenerateVectorError`,
      `InvalidKError`), `GuardrailError`
    - No `sys.exit` anywhere in library code
    - _Requirements: 1.3, 4.6, 4.7, 4.8, 4.9, 6.5, 7.7, 7.11, 7.12, 10.7, 10.8, 10.13, 10.14, 11.9_

  - [x] 2.2 Implement `src/askmydocs/models.py`
    - Frozen dataclasses `Document(text, source_path, file_type)`,
      `Chunk(text, source_path, index, start_offset, end_offset)` with `length`, `chunk_id`
      (`"{source_path}#{index}"`), and `to_metadata()`,
      `StoredRecord(chunk, embedding: tuple[float, ...], insertion_index)`,
      `SearchHit(chunk, score, insertion_index)`
    - _Requirements: 7.3, 8.10, 8.11, 8.12, 10.2, 10.4, 10.15_

  - [ ]* 2.3 Write `tests/test_models.py`
    - `chunk_id` format, `to_metadata()` key set and value types, `length` equals
      `end_offset - start_offset`, immutability of `StoredRecord.embedding`
    - _Requirements: 8.10, 8.11, 10.2, 10.4_

- [x] 3. Configuration, Reporter, and API key redaction
  - Serves Mon: nothing can select a provider or chunk size until configuration parses.

  - [x] 3.1 Implement `Configuration` and `load_configuration` in `src/askmydocs/config.py`
    - Frozen `Configuration` dataclass with all eleven settings plus the `stride` property
    - `load_configuration(env: Mapping[str, str] | None = None)` reading from the injected
      mapping, defaulting to `os.environ` merged over `.env` values
    - Fixed validation order: provider identifier → provider-specific model default → strict
      integer parsing → range checks → `chunk_overlap < chunk_size` cross-check → API key
      requirement last
    - Strict integer parsing: accept surrounding whitespace, reject `"500.0"`, `"5e2"`,
      `"1_000"`, `"abc"`; treat absent and empty identically for every setting but the API key
    - Error messages name the variable, the rejected value, and the permitted range or
      expected type; the overlap message names both variables and both values
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 1.9, 1.10, 1.11, 1.12_

  - [x] 3.2 Implement `REDACTION_MARKER` and `redact()` in `src/askmydocs/config.py`
    - Replace the whole API key and every contiguous substring of the key of at least 8
      characters with the marker; return the text unchanged when the key is `None` or shorter
      than 8 characters
    - _Requirements: 1.7_

  - [x] 3.3 Implement `Reporter` in `src/askmydocs/reporting.py`
    - `info`, `warning`, `error`, `progress(done, total)` writing to injected streams
    - Every outgoing string passes through `redact()` before reaching a stream, so redaction
      has exactly one choke point
    - _Requirements: 1.7, 6.4, 7.5, 7.7, 7.8, 7.11, 7.12, 11.7_

  - [ ]* 3.4 Write `tests/test_config.py` example and boundary tests
    - Every default applied when a variable is absent and when it is empty
    - Exact range boundaries for chunk size, timeout, retry attempts, input length, batch size,
      chunks per run — one accepted and one rejected value at each edge
    - Unparsable numeric values, rejected provider identifier message content, missing API key
      with the OpenAI provider raising before any provider object is constructed
    - Local provider path never reads the API key variable and leaves `api_key` as `None`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.11, 1.12_

  - [ ]* 3.5 Write property test for API key redaction
    - **Property 20: The API key never survives redaction**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Generated key of 8–64 characters, a carrier message, a drawn substring window of at
      least 8 characters, and an insertion position; assert the key and the window are both
      absent from the output and the marker is present
    - **Validates: Requirements 1.7**

  - [ ]* 3.6 Write property test for chunk configuration acceptance and rejection
    - **Property 21: Invalid chunk configurations are always rejected, valid ones always accepted**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Size drawn from 1–10000; invalid region drawn from negatives and values at or above the
      size, valid region from 0 to size − 1; invalid raises `ConfigurationError` naming both
      variables and both values, valid preserves both values
    - **Validates: Requirements 1.9, 1.8**

  - [ ]* 3.7 Write property test for provider identifier resolution
    - **Property 22: Provider identifiers resolve case-insensitively after trimming**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Sampled supported identifier × random per-character case map × whitespace padding drawn
      from `" \t\n\r"`; assert the resolved provider equals the canonical identifier
    - **Validates: Requirements 1.10**

  - [ ]* 3.8 Write `tests/test_reporting.py`
    - Warnings and errors go to the error stream, info and progress to the output stream
    - A message embedding the API key and a message embedding an 8-character fragment of it
      both emerge redacted
    - `progress` prints the cumulative and total counts
    - _Requirements: 1.7, 11.7_

- [ ] 4. Checkpoint — configuration layer green
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Cosine similarity and shared test strategies
  - Serves Mon–Tue: the numeric core behind the sentence comparison and the store's ranking.

  - [x] 5.1 Implement `src/askmydocs/similarity.py`
    - `ZERO_NORM_THRESHOLD = 1e-12`, `euclidean_norm` over float64 via numpy
    - `cosine_similarity` with the fixed validation order: length mismatch → empty vector →
      non-finite element → degenerate norm, each raising the matching typed error and naming
      which input violated the rule
    - Return the dot product over the product of norms, clamped into `[-1.0, 1.0]`
    - _Requirements: 4.1, 4.2, 4.6, 4.7, 4.8, 4.9_

  - [x] 5.2 Implement `tests/strategies.py`
    - `finite_element`, `vector` (degenerate draws repaired, never filtered), `vector_pair`,
      `scale_factor` (log-uniform over 1e-6 to 1e6), `chunk_config`, `document_text`,
      `unicode_text` over the mixed alphabet, `multi_chunk_case`, `add_batches` (zero-sized
      batches allowed), and a `make_chunk` helper
    - _Requirements: 13.7_

  - [ ]* 5.3 Write `tests/test_similarity_examples.py`
    - Hand-computed values: orthogonal → 0.0, opposite → −1.0, 45° → 0.7071, one-dimensional
      pairs
    - Validation ordering: two empty vectors raise the empty-vector error while one empty and
      one non-empty raises the length mismatch; a `NaN` element raises before the norm check
    - Error messages report both lengths and identify which input was at fault
    - _Requirements: 4.1, 4.6, 4.7, 4.8, 4.9_

  - [ ]* 5.4 Write property test for the cosine similarity bound
    - **Property 1: Cosine similarity is bounded**
    - Minimum 100 examples; run at 200 under the `pure` profile, seeded with `@example` cases
      for unit, opposite, and one-dimensional pairs
    - Strategy `vector_pair()`; assert the result lies within `[-1 - 1e-9, 1 + 1e-9]`
    - **Validates: Requirements 4.2**

  - [ ]* 5.5 Write property test for symmetry
    - **Property 2: Cosine similarity is symmetric**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Strategy `vector_pair()`; assert `abs(sim(a, b) - sim(b, a)) <= 1e-9`
    - **Validates: Requirements 4.3**

  - [ ]* 5.6 Write property test for self-similarity
    - **Property 3: Self-similarity is one**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Strategy `vector()`; assert `abs(sim(a, a) - 1.0) <= 1e-9`
    - **Validates: Requirements 4.4**

  - [ ]* 5.7 Write property test for scale invariance
    - **Property 4: Positive scaling does not change similarity**
    - Minimum 100 examples; run at 200 under the `pure` profile, seeded with `@example` at
      k = 1e-6, k = 1.0, k = 1e6
    - Strategy `vector()` × `scale_factor`; assert `abs(sim(a, k·a) - 1.0) <= 1e-9`
    - **Validates: Requirements 4.5**

- [x] 6. Fixed-size chunker with overlap
  - Serves Wed–Thu: the boundary arithmetic, written and proved by test rather than delegated.

  - [x] 6.1 Implement `src/askmydocs/chunking.py`
    - Frozen `Chunker(chunk_size, chunk_overlap)` whose `__post_init__` rejects
      `chunk_size < 1` and overlap outside `[0, chunk_size - 1]` with `ChunkerConfigError`
    - `stride`, `expected_chunk_count(text_length)` closed form, `chunk_text(text, source_path)`
      following the design pseudocode (empty → `[]`, fits-in-one → single full-text chunk,
      otherwise constant stride with the `end == n` stop condition), `chunk_document(document)`
    - All lengths and offsets in Unicode code points: plain `str` slicing only, no encoding,
      no `unicodedata.normalize`, no grapheme grouping
    - Imports only `models` and `errors` — no store type appears in any signature
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 10.3_

  - [ ]* 6.2 Write `tests/test_chunking_examples.py` from the design's worked examples
    - Example A: `"abcdefghijkl"`, size 5, overlap 2 → 4 chunks with the tabulated offsets
    - Example B: 13 characters, size 10, overlap 4 → 2 chunks, final length 7
    - Example C: `"abcdefgh"`, size 5, overlap 2 → exactly 2 chunks, no spurious trailing chunk
    - Example D: 20 characters, size 500, overlap 50 → one chunk shorter than the overlap
    - Example E: `"a😀b😀c"`, size 3, overlap 1 → 2 chunks of 3 code points each
    - Empty text → empty sequence; invalid `Chunker` settings raise
    - _Requirements: 8.8, 8.9, 8.10, 8.11, 8.13, 8.14_

  - [ ]* 6.3 Write property test for chunk length bounds
    - **Property 5: Chunk length bounds**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `document_text` × `chunk_config()` for the universal bound and `multi_chunk_case()` for
      the multi-chunk clause; every chunk at most the size, all but the last exactly the size,
      the last strictly greater than the overlap
    - **Validates: Requirements 8.2, 8.3**

  - [ ]* 6.4 Write property test for the constant stride
    - **Property 6: Consecutive Chunks advance by exactly the stride**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `document_text` × `chunk_config()`; consecutive start offsets differ by exactly
      size − overlap, which also establishes strictly ascending starts
    - **Validates: Requirements 8.4, 8.1**

  - [ ]* 6.5 Write property test for text reconstruction
    - **Property 7: Chunks reconstruct the Document text exactly**
    - Minimum 100 examples; run at 200 under the `pure` profile, seeded with `@example` at
      text lengths of `size`, `size + 1`, `size + stride - 1`, `size + stride`, `size + stride + 1`
    - First chunk plus every later chunk minus its leading overlap equals the document text;
      empty text yields no chunks
    - **Validates: Requirements 8.5**

  - [ ]* 6.6 Write property test for the overlap invariant
    - **Property 8: Overlap invariant between consecutive Chunks**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `multi_chunk_case()`; trailing overlap characters of each chunk equal the leading overlap
      characters of the next, skipped when the overlap is 0
    - **Validates: Requirements 8.6**

  - [ ]* 6.7 Write property test for deterministic boundaries
    - **Property 9: Chunk boundaries are deterministic**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Two separately constructed `Chunker` instances with identical settings produce equal
      `list[Chunk]` values
    - **Validates: Requirements 8.7**

  - [ ]* 6.8 Write property test for offset indexing
    - **Property 10: Offsets index the Document text**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `text[c.start_offset:c.end_offset] == c.text`, `end - start == len(text)`,
      `c.source_path` carried through, `0 <= start <= end <= len(text)`
    - **Validates: Requirements 8.11, 8.10**

  - [ ]* 6.9 Write property test for ordinal indices
    - **Property 11: Ordinal indices are consecutive from zero**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `[c.index for c in chunks] == list(range(len(chunks)))`
    - **Validates: Requirements 8.12**

  - [ ]* 6.10 Write model-based property test for the chunk count
    - **Property 12: Chunk count matches the closed-form model**
    - Minimum 100 examples; run at 200 under the `pure` profile, seeded with `@example` at
      text lengths 0, 1, exactly `size`, and strictly less than `overlap`
    - Count equals `expected_chunk_count`; the single-chunk case holds the full text; no chunk
      other than the last ends at the text length and the last always does
    - **Validates: Requirements 8.13, 8.8, 8.9, 8.1**

  - [ ]* 6.11 Write property test for code-point counting
    - **Property 13: All counting is in Unicode code points**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `unicode_text` (astral-plane characters and combining sequences) ×
      `chunk_config(max_size=40)`; length bound, reconstruction, and offset indexing all hold,
      plus a guard asserting the UTF-8 byte length differs from the code-point length on
      non-ASCII samples so byte counting cannot pass silently
    - **Validates: Requirements 8.14**

- [ ] 7. Sample notes discovery
  - Serves Wed–Thu: deterministic, cross-platform file discovery.

  - [ ] 7.1 Implement `src/askmydocs/loading/discovery.py`
    - `SUPPORTED_EXTENSIONS`, frozen `DiscoveredFile`, `SkippedEntry`, `DiscoveryResult`
    - `discover_notes(notes_folder)`: recursive walk; case-insensitive extension match;
      exclude the folder README, dot-prefixed entries, and symbolic links, each recorded as a
      skipped entry with a reason; forward-slash relative paths sorted by ascending code-point
      comparison; counts grouped by lower-cased extension and a total skipped count
    - Raise `NotesFolderError` naming the resolved absolute path and which condition applied
      when the path is missing or is not a directory
    - _Requirements: 6.3, 6.5, 6.6, 6.7, 6.8, 6.9, 7.3, 7.6_

  - [ ]* 7.2 Write `tests/test_discovery.py` example and boundary tests
    - Missing path and a path that is a file, each with the expected message content
    - Warning text and continuation at 4 and at 11 supported files; no warning at 5 and 10
    - Empty folder reports zero documents and is not an error
    - Symbolic link excluded and listed among skipped entries with its reason, skipped on
      platforms that deny symlink creation
    - Unsupported extensions listed with the offending extension
    - _Requirements: 6.4, 6.5, 6.6, 6.9, 7.6_

  - [ ]* 7.3 Write property test for discovery completeness, filtering, and ordering
    - **Property 29: Discovery is complete, filtered, and deterministically ordered**
    - Minimum 100 examples; run at 100 under the `filesystem` profile with a 3-second deadline
    - Generated trees of 0–12 entries up to depth 3 with adversarial names (`Z.md`, `a.md`,
      `_x.md`, `B/A.md`), randomized extension case, unsupported extensions, dot-prefixed
      files and directories, and a root `README.md`; assert the returned relative-path list
      equals the independently computed expectation sorted by code point, paths contain no
      backslash and are unique, extension counts match, and the skipped set matches
    - **Validates: Requirements 6.3, 6.7, 6.8, 7.3**

- [ ] 8. PDF and markdown document loaders
  - Serves Wed–Thu: real notes reaching the chunker as plain text with source metadata.

  - [ ] 8.1 Implement `src/askmydocs/loading/base.py`
    - `MAX_FILE_BYTES = 25 * 1024 * 1024`, `DocumentLoader` protocol
    - `normalize_newlines(text)` converting CRLF and lone CR to a single LF and applying no
      other transformation
    - `decode_utf8(data)` stripping a leading byte-order mark and falling back to
      `errors="replace"`, returning the text and the replacement count
    - _Requirements: 7.4, 7.5, 7.10, 7.12_

  - [ ] 8.2 Implement `src/askmydocs/loading/markdown_loader.py`
    - Return a `Document` with the file's full text, all markup, indentation, and blank lines
      preserved, decoded via `decode_utf8` and newline-normalized
    - Populate `source_path` as the forward-slash relative path and `file_type` as the
      lower-cased extension without the leading period
    - _Requirements: 7.2, 7.3, 7.4, 7.9, 7.10_

  - [ ] 8.3 Implement `src/askmydocs/loading/pdf_loader.py`
    - Join per-page extracted text in ascending page order with exactly one LF between pages
      and nothing else inserted
    - Raise `EncryptedPdfError` when the reader reports encryption
    - _Requirements: 7.1, 7.3, 7.9, 7.11_

  - [ ] 8.4 Implement `src/askmydocs/loading/pipeline.py`
    - `load_documents(discovery, notes_folder, reporter)` loading in discovery order with
      per-file isolation in the design's order: size check before opening, then encryption,
      then open/parse failure, then the whitespace-only check
    - Oversize, encrypted, unparsable, and whitespace-only files are reported and excluded;
      invalid UTF-8 files are included with a replacement-count warning; loading always
      continues with the remaining files
    - _Requirements: 7.5, 7.7, 7.8, 7.11, 7.12_

  - [ ]* 8.5 Write `tests/test_loading.py` example tests
    - Page joining with a stubbed page reader and one small committed PDF fixture: exactly one
      LF between pages, nothing else inserted
    - Invalid UTF-8 bytes produce the replacement character and a warning naming the path and
      the replacement count
    - Loading the same file twice yields character-identical text and identical metadata
    - Unsupported files appear in the skip listing with their extension
    - _Requirements: 7.1, 7.5, 7.6, 7.9_

  - [ ]* 8.6 Write property test for decoding and newline normalization
    - **Property 30: Decoding and newline normalization is a faithful round trip**
    - Minimum 100 examples; run at 100 under the `filesystem` profile
    - Text assembled from an alphabet including `"\r\n"`, `"\r"`, `"\n"`, `"\t"`, `"\u00a0"`,
      `"e\u0301"`, `"\u00e9"`, and `"\U0001F600"` × a byte-order-mark flag; assert the loaded
      text equals the CR-normalized original, contains no CR, does not start with a BOM, and
      that `"e\u0301"` has not been normalized to `"\u00e9"`
    - **Validates: Requirements 7.2, 7.4, 7.10**

  - [ ]* 8.7 Write property test for per-file error isolation
    - **Property 31: A failing file never prevents the other files from loading**
    - Minimum 100 examples; run at 100 under the `filesystem` profile
    - 1–8 valid markdown files plus 1–3 bad files (unparsable, encrypted, oversize,
      whitespace-only) injected at drawn positions with the size probe and PDF reader stubbed;
      assert the returned `source_path` list equals the expected valid subset in discovery
      order, one report per bad file names its path and reason, and no exception escapes
    - **Validates: Requirements 7.7, 7.8, 7.11, 7.12**

- [ ] 9. Checkpoint — pure layers and loading green
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Vector store interface and In_Memory_Store
  - Serves Fri wrap-up, and is the single seam Week 2 replaces with Chroma.

  - [ ] 10.1 Implement `src/askmydocs/stores/base.py`
    - Abstract `VectorStoreInterface` declaring exactly `add(chunks, embeddings)`, `count()`,
      `query(embedding, k)`, and a `dimensionality` property
    - Docstrings state the contract: empty batch is a no-op, add validation is atomic, query
      returns descending similarity with ties broken by ascending insertion index
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 10.2 Implement `src/askmydocs/stores/memory.py`
    - `InMemoryStore` holding `list[StoredRecord]` and a lazily fixed `_dimensionality`
    - `add` validates in order before mutating anything: count match, empty-batch early
      return, uniform incoming vector length, match against the established dimensionality;
      then builds records and extends, so atomicity is structural
    - `query` scores every record with `cosine_similarity`, sorts by
      `(-score, insertion_index)`, slices to `k`, and returns `SearchHit` values; rejects
      non-integer or `k < 1`, and a query vector of the wrong length or degenerate norm; an
      empty store returns `[]` without raising even though no dimensionality is established
    - _Requirements: 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 10.13, 10.14, 10.15_

  - [ ] 10.3 Implement `src/askmydocs/stores/factory.py`
    - `build_store(configuration)` returning `InMemoryStore`, with the single branch point
      Week 2 extends
    - _Requirements: 10.1, 10.2_

  - [ ]* 10.4 Write `tests/test_store_examples.py`
    - `InMemoryStore` satisfies the interface and exposes no operation beyond the four members
    - `k = 0`, negative `k`, `k = 1.5`, and `k = True`-style non-integers each raise
      `InvalidKError` stating the permitted range and returning no result
    - A query vector of the wrong length reports both lengths; a zero-norm query vector raises
      `DegenerateVectorError` naming the violated property
    - _Requirements: 10.1, 10.2, 10.13, 10.14_

  - [ ]* 10.5 Write property test for count additivity
    - **Property 14: Stored item count is additive over add operations**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `add_batches()` with zero-sized batches allowed; after each add the count equals the
      running sum and an empty batch neither changes the count nor raises
    - **Validates: Requirements 10.6, 10.5**

  - [ ]* 10.6 Write property test for chunk-to-vector association
    - **Property 15: Chunk-to-vector association is preserved**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Distinct chunk texts and vectors per index; the flattened stored
      `(chunk, embedding)` list equals the flattened input pairs in order and
      `insertion_index` equals the position
    - **Validates: Requirements 10.4, 10.2**

  - [ ]* 10.7 Write property test for add atomicity
    - **Property 16: Add validation is atomic**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - A valid prior state from `add_batches()` then a drawn invalid batch of one of the three
      kinds; the expected error type is raised, its message contains both numbers, and the
      count and full record list equal a snapshot taken before the failing call
    - **Validates: Requirements 10.7, 10.8**

  - [ ]* 10.8 Write model-based property test for top-K ranking
    - **Property 17: Query returns the top-K prefix in descending similarity order, deterministically**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - 0–40 pairs of fixed dimension, a drawn query vector, and `k` drawn from
      `[1, count + 20]`; assert the hit count equals `min(k, count)`, scores are
      non-increasing, the result equals the independently computed reference ranking
      `sorted(records, key=(-score, insertion_index))[:k]`, and a repeated query is equal
    - **Validates: Requirements 10.9, 10.10**

  - [ ]* 10.9 Write property test for self-retrieval and tie ordering
    - **Property 18: A stored vector retrieves its own Chunk, ties by insertion index**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Regime (a): 1–20 pairwise-distinct vectors built distinct by construction; querying with
      record *i*'s vector and `k = 1` returns chunk *i*. Regime (b): one vector duplicated at
      2–5 insertion positions; the maximal-score hits carry strictly ascending insertion
      indices starting at the smallest
    - **Validates: Requirements 10.12, 10.15**

  - [ ]* 10.10 Write property test for the empty store
    - **Property 19: An empty store returns no results without raising**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - `vector()` × `st.integers(1, 100)` against a fresh store; `query` returns `[]` and raises
      nothing, exercising the "no dimensionality established" branch
    - **Validates: Requirements 10.11**

- [ ] 11. Embedder core: retry, batching, and the substitute embedder
  - Serves the Mon–Tue embedding goal; sequenced here because it depends on configuration.

  - [ ] 11.1 Implement `src/askmydocs/embeddings/retry.py`
    - `FailureKind` enum, `classify_failure(error)` mapping rate-limit, timeout, and connection
      errors to transient and credential or rejected-input errors to terminal
    - `RetryPolicy(max_retry_attempts, sleep=time.sleep)` with
      `delay_for_attempt(n) = min(30.0, max(1.0, 2.0 ** n))` and `run(operation, description)`
      retrying transient failures only, re-raising terminal failures with zero retries, and
      raising `EmbeddingFailedError` naming provider, attempts made, and reason on exhaustion
    - _Requirements: 2.7, 2.9, 2.10_

  - [ ] 11.2 Implement `src/askmydocs/embeddings/base.py`
    - `Embedder` protocol and `BaseEmbedder(ABC)` with `dimensionality` discovered from the
      first successful response then asserted constant, `batch_call_count`, and the abstract
      `_embed_batch`
    - `embed_texts`: empty input returns `[]` with no request; validate every element before
      any request, reporting the first offending zero-based position and the reason; split into
      consecutive segments of at most the configured batch size; run each segment through the
      retry policy; concatenate in input order; on segment failure raise `EmbeddingFailedError`
      naming the provider, the segment's input positions, and the reason, returning nothing
      partial
    - `embed_text(text)` defined as `embed_texts([text])[0]`
    - Imports no store type; returns `list[list[float]]` only
    - _Requirements: 2.1, 2.4, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 10.3_

  - [ ] 11.3 Implement `tests/fakes.py`
    - `FakeEmbedder(BaseEmbedder)` seeded from SHA-256 of the text (not `hash()`), fixed
      dimensionality defaulting to 8 and parameterizable to 3, guaranteed non-zero norm,
      recording its segments
    - `ScriptedProvider` raising a caller-supplied sequence of exceptions before succeeding,
      `SpyEmbedder` failing the test if invoked, `SleepRecorder` capturing requested delays
    - _Requirements: 13.5, 13.6_

  - [ ]* 11.4 Write retry, timeout, and classification example tests in `tests/test_embedder.py`
    - Transient failures retried exactly `max_retry_attempts` times for maxima 0, 1, 3, 10
    - Every recorded delay between 1 and 30 seconds; retry-then-success returns vectors with
      the expected number of recorded delays
    - Exhaustion message names provider, attempt count, and reason; terminal failure yields one
      call and no delays; a timeout exception is treated as transient
    - `classify_failure` called directly with each synthetic exception type; segment failure
      reporting names the segment's zero-based input positions and returns no partial list
    - _Requirements: 2.7, 2.9, 2.10, 3.7_

  - [ ]* 11.5 Write property test for embedder output shape
    - **Property 23: Embedder output shape is constant and well-formed**
    - Minimum 100 examples; run at 200 under the `pure` profile, substitute embedder only
    - Non-empty non-whitespace texts and lists thereof; element count equals the reported
      dimensionality, all elements finite, norm above 0, and one shared length across a batch
    - **Validates: Requirements 2.1, 2.4**

  - [ ]* 11.6 Write property test for embedding determinism
    - **Property 24: Embedding is deterministic for identical text**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Embedding the same text twice agrees within 1e-6 per element, and two independently
      constructed substitute embedders return equal vectors across instances and processes
    - **Validates: Requirements 2.5, 13.5**

  - [ ]* 11.7 Write property test for batch and single agreement
    - **Property 25: Batch embedding agrees with single embedding and preserves order**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - Lists of 1–40 valid texts × batch sizes 1–8 so multi-segment splits are common; element
      agreement within 1e-6 at every position and a shuffled input yields a correspondingly
      shuffled output
    - **Validates: Requirements 3.3, 3.1, 3.2**

  - [ ]* 11.8 Write property test for batch segmentation arithmetic
    - **Property 26: Batch segmentation arithmetic**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - N drawn from 0–200 × batch size 1–32 against the recording substitute provider; request
      count equals `ceil(N / B)` and 0 when N is 0, every segment holds at most B texts, and
      concatenated segments equal the input
    - **Validates: Requirements 3.5, 3.4**

  - [ ]* 11.9 Write property test for invalid batch element reporting
    - **Property 27: Invalid batch elements are reported by first position, with no request issued**
    - Minimum 100 examples; run at 200 under the `pure` profile
    - 1–30 valid texts with 1–3 invalid elements injected at drawn positions, invalid drawn
      from a whitespace alphabet including `U+00A0`, `U+2003`, `U+3000` or from over-length
      text; the error names the smallest injected position and the provider call count is 0
    - **Validates: Requirements 3.6, 2.6, 2.8**

  - [ ]* 11.10 Write property test for retry delay bounds
    - **Property 28: Retry delays stay within the mandated bounds**
    - Minimum 100 examples; run at 200 under the `pure` profile, pure arithmetic with no
      sleeping and no network
    - Attempts 0–10; each delay between 1.0 and 30.0 seconds and the sequence non-decreasing
    - **Validates: Requirements 2.9**

- [ ] 12. Embedding providers and factory

  - [ ] 12.1 Implement `src/askmydocs/embeddings/local_provider.py`
    - `SentenceTransformersEmbedder` loading the model once, lazily, in `_embed_batch`
    - Never reads the API key variable and opens no socket to the OpenAI API
    - _Requirements: 1.4, 2.1, 2.4_

  - [ ]* 12.2 Implement `src/askmydocs/embeddings/openai_provider.py`
    - `OpenAIEmbedder._embed_batch` issuing one request per segment with
      `timeout=configuration.request_timeout_seconds` and returning vectors sorted by the
      response `index` field so ordering never depends on response order
    - Optional: skip this sub-task if the learner works only with the local model. The factory
      must then raise a clear `ConfigurationError` when the OpenAI provider is selected
    - _Requirements: 1.3, 2.1, 2.9, 2.10, 3.2_

  - [ ] 12.3 Implement `src/askmydocs/embeddings/factory.py`
    - `build_embedder(configuration, retry_policy=None)` selecting the provider implementation
      from `configuration.provider`, defaulting to the local provider
    - _Requirements: 1.4, 1.5_

  - [ ]* 12.4 Write `tests/test_providers.py`
    - Local provider completes with the API key variable absent and `socket.socket` patched to
      raise, with the model loader stubbed so no weights are downloaded
    - OpenAI provider with a stubbed client: out-of-order response indices still yield input
      order, and the recorded request kwargs carry the configured timeout
    - `build_embedder` returns the implementation matching the configured provider
    - _Requirements: 1.3, 1.4, 1.5, 2.9, 3.2_

- [ ] 13. Runnable scripts
  - Each script is a thin `main(argv, embedder=None) -> int` that builds the `Configuration` and
    `Reporter`, calls library functions, catches typed exceptions, and maps them to the design's
    exit statuses. The optional `embedder` parameter is how tests inject the substitute.

  - [ ] 13.1 Implement `scripts/01_embed_one.py`
    - Embed one sentence; print the dimensionality and the first 5 elements, or all elements
      when the dimensionality is below 5
    - Mon–Tue deliverable
    - _Requirements: 2.2, 2.3_

  - [ ] 13.2 Implement `scripts/02_compare_sentences.py`
    - Define 3 similar and 3 unrelated sentences of 40–200 characters each, sharing no content
      word across groups
    - Embed all 6 in a single batch call; print 3 within-group lines, 9 cross-group lines, both
      labelled means, and a final verdict line, all values rounded to 4 decimal places
    - On embedder failure print no similarity or mean values, name the provider and reason, and
      exit with a failure status
    - Mon–Tue deliverable
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ] 13.3 Implement `scripts/03_chunking_experiment.py`
    - Chunk the loaded documents once for each of the 6 combinations of sizes 200, 500, 1000
      and overlaps 0, 50; issue no embedder call at all
    - Print per combination the size, overlap, total chunk count, mean length to 1 decimal
      place, and minimum and maximum length; print the first 2 chunks truncated to 200
      characters with a truncation marker, or all chunks when fewer than 2 were produced
    - Rewrite `reports/chunking-experiment.md` wholesale with the statistics table
    - With no supported files: print a document count of 0, leave any existing statistics file
      unchanged, exit 0
    - Wed–Thu deliverable
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_

  - [ ] 13.4 Implement `scripts/04_pipeline.py`
    - Discover, load, chunk, then compare the total chunk count against `Max_Chunks_Per_Run`
      and terminate with exit 4 naming both numbers before any request when exceeded
    - Embed in batches of at most the configured batch size, printing cumulative and total
      counts after each batch returns; add every chunk with its vector to the store built by
      `build_store`
    - Run the demonstration query and print the query text plus the top 3 hits (or all stored
      chunks when fewer than 3) with source path, similarity to 4 decimal places, and text
      truncated to 200 characters
    - Print the summary: document count, chunk count, dimensionality, store item count, batch
      call count, and elapsed seconds to 1 decimal place
    - Empty folder prints zero counts, issues no embedder call, prints the no-chunks line for
      the demonstration query, and exits 0; embedder failure discards the store, reports the
      provider, reason, and chunks embedded before the failure, and exits 5
    - Fri wrap-up deliverable
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10_

  - [ ]* 13.5 Write `tests/test_scripts.py` coverage for scripts 01 and 02
    - Script 01 prints the dimensionality and exactly 5 elements at dimensionality 8 and all
      elements at dimensionality 3
    - Script 02 emits exactly 3 within-group and 9 cross-group lines, both means, the verdict
      line, and exactly one batch call for all 6 sentences; sentence lengths and the
      no-shared-content-word rule are asserted on the defined data
    - Embedder failure yields a failure status with no similarity or mean output
    - _Requirements: 2.2, 2.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 13.6 Write `tests/test_scripts.py` coverage for script 03
    - All 6 combinations reported with the required statistics; sample output truncated at 200
      characters with the marker; the statistics file fully replaced on a second run
    - Empty folder leaves an existing statistics file byte-identical and exits 0
    - A spy embedder asserts the experiment issues no provider call
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_

  - [ ]* 13.7 Write `tests/test_scripts.py` coverage for script 04
    - Over a fixture folder of small files: store count equals the total chunk count, batch
      call count equals `ceil(total / max_batch_size)`, and two runs produce the same chunk
      count and the same ordered chunk texts
    - Progress lines appear after each batch; the summary prints all six values
    - Chunk count above the guardrail exits 4 with zero embedder calls; a scripted mid-run
      failure exits 5, reports the count embedded before failure, and leaves no populated store
    - Empty folder exits 0 with the no-chunks demonstration line and zero embedder calls
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10_

- [ ] 14. Layering enforcement

  - [ ] 14.1 Write `tests/test_layering.py`
    - AST-parse `src/askmydocs/chunking.py` and every module under `src/askmydocs/embeddings/`
      and assert no import statement names `askmydocs.stores` or any store symbol
    - Resolve the type hints of the public `Chunker` and `Embedder` members and assert no
      parameter or return annotation references `VectorStoreInterface` or `InMemoryStore`
    - _Requirements: 10.3_

- [ ] 15. Learning notes and README
  - Fri wrap-up: the written artifacts and the setup document that names them.

  - [ ] 15.1 Write `learning-notes/why-split-documents.md`
    - A section under its own heading of at least 100 words on why documents are split,
      addressing both model context length limits and retrieval relevance precision
    - Record, for at least 2 of the 6 experiment combinations, the observation of how size and
      overlap changed which sentences landed in one chunk, using the output of script 03
    - _Requirements: 12.1, 9.5_

  - [ ] 15.2 Write `learning-notes/semantic-search.md`
    - A section under its own heading explaining semantic search in 3–4 sentences with no
      formulas, no mathematical operators, and no variable symbols
    - Record the selected provider, model name, dimensionality, and the reason for the choice;
      the within-group and cross-group means printed by script 02 to at least 4 decimal places
      together with the 6 sentences; and the chosen chunk size and overlap with the reason
    - _Requirements: 12.2, 12.3, 12.4, 12.5_

  - [ ] 15.3 Write `README.md`
    - Setup commands including `pip install -e .`, and why the `src/` layout requires it
    - Every environment variable with its default and permitted range, and the model default
      per provider
    - One command per script, the location of `reports/chunking-experiment.md`, the single test
      command, and the file containing each required learning-notes section
    - _Requirements: 1.2, 9.4, 12.6, 13.3, 13.4_

  - [ ]* 15.4 Extend `tests/test_repository.py` with artifact checks
    - Word count of the splitting section at or above 100; the semantic-search section has 3–4
      sentences and contains no mathematical operator or variable symbol
    - Both notes files exist under version control and the README names the file holding each
      required section
    - The README states the test command and one command per script
    - _Requirements: 12.1, 12.2, 12.6, 13.3, 13.4_

- [ ] 16. Final checkpoint — full suite green within the time budget
  - Ensure all tests pass, ask the user if questions arise.

## Week 1 day mapping

| Day | Goal | Tasks |
|---|---|---|
| Mon | Project setup, configuration, redaction | 1, 2, 3, 4 |
| Mon–Tue | Embeddings and cosine similarity comparison | 5, 11, 12, 13.1, 13.2, 13.5 |
| Wed–Thu | Loading and chunking pipeline | 6, 7, 8, 9, 13.3, 13.6 |
| Fri | Wrap-up pipeline, in-memory store, written notes | 10, 13.4, 13.7, 14, 15, 16 |

Tasks are ordered by dependency rather than by day, so the embedder and scripts 01 and 02 —
Mon–Tue deliverables — are built after the pure layers they rely on. A learner following the
calendar can run tasks 1–5 and 11–13.2 first and come back for the chunking work.

## Notes

- Tasks marked with `*` are optional. All of them are tests except 12.2, the OpenAI provider,
  which is genuinely optional for a learner working only with the local Sentence Transformers
  model. Skipping the test sub-tasks yields a faster MVP at the cost of the verification
  Requirements 13.4 and 13.7 ask for.
- Each of the 31 correctness properties from the design is its own sub-task, tagged with its
  property number, its minimum example count, and the criteria it validates. Properties 1–19
  cover the FOR ALL criteria of Requirements 4, 8, and 10 that Requirement 13.7 mandates;
  properties 20–31 are supplementary.
- Property tests and example tests are complementary. Properties carry the universal statements;
  examples carry the hand-computed cosine values, the exact configuration boundaries, validation
  ordering, error message contents, and script output formats — none of which a generator can pin.
- No task issues a network request. The substitute embedder and the session socket guard are in
  place before any embedding test runs.
- The four worked chunking examples in task 6.2 are transcribed verbatim from the design, so the
  document and the suite cannot drift apart.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["1.5", "2.1", "2.2"] },
    { "id": 2, "tasks": ["1.6", "2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3"] },
    { "id": 4, "tasks": ["3.4", "5.1"] },
    { "id": 5, "tasks": ["3.5", "5.2", "6.1"] },
    { "id": 6, "tasks": ["3.6", "3.8", "5.3", "6.2", "7.1"] },
    { "id": 7, "tasks": ["3.7", "5.4", "6.3", "7.2", "8.1"] },
    { "id": 8, "tasks": ["5.5", "6.4", "7.3", "8.2", "8.3"] },
    { "id": 9, "tasks": ["5.6", "6.5", "8.4", "10.1"] },
    { "id": 10, "tasks": ["5.7", "6.6", "8.5", "10.2", "11.1"] },
    { "id": 11, "tasks": ["6.7", "8.6", "10.3", "11.2"] },
    { "id": 12, "tasks": ["6.8", "8.7", "10.4", "11.3", "12.1"] },
    { "id": 13, "tasks": ["6.9", "10.5", "11.4", "12.2", "12.3"] },
    { "id": 14, "tasks": ["6.10", "10.6", "11.5", "12.4", "13.1"] },
    { "id": 15, "tasks": ["6.11", "10.7", "11.6", "13.2", "13.3"] },
    { "id": 16, "tasks": ["10.8", "11.7", "13.4", "14.1"] },
    { "id": 17, "tasks": ["10.9", "11.8", "13.5", "15.1"] },
    { "id": 18, "tasks": ["10.10", "11.9", "13.6", "15.2"] },
    { "id": 19, "tasks": ["11.10", "13.7", "15.3"] },
    { "id": 20, "tasks": ["15.4"] }
  ]
}
```
