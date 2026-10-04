# Implementation Plan: Phase 2 — Vector Database and Retrieval

## Overview

Phase 2 adds a second implementation of the Phase 1 `Vector_Store_Interface` backed by a persistent Chroma collection, then builds incremental ingest, retrieval with a relevance threshold, an append-only retrieval log, and two measurement scripts on top of it. The language is Python, as the design specifies throughout.

The build order follows the design's module dependency graph strictly: repository and dependency pin → configuration extension → Phase 2 errors → `Store_Admin_Interface` and `InMemoryAdminStore` → `chroma_compat` and `ChromaStore` → factory branch → conformance suite across both stores → ingest → retrieval → evaluation → scripts → layering and frozen-module checks → written notes. Nothing is implemented before the module it depends on exists.

Grouping is aligned with the Phase 2 day plan:

- **Mon–Tue** (tasks 1–11): vector database setup, `Chroma_Store`, the conformance suite, and incremental ingest.
- **Wed–Thu** (tasks 13–18): the `Retriever`, the `Query_Script`, and the Top-K tuning experiment plus the relevance review.
- **Fri** (tasks 20–22): retrieval-log integrity and concurrency verification, Phase 1 isolation checks, and the `Comparison_Note` and `Learning_Notes`.

The `RetrievalLogWriter` is *implemented* Wed–Thu because the `Retriever` cannot append a record without it (Requirement 14.1); Friday's group covers the log's integrity properties, the concurrent-append tests, and the documentation of the record schema.

Two hard constraints apply to the whole plan and are repeated in the tasks they bind:

- **No Phase 1 frozen module may be edited.** `chunking.py`, `similarity.py`, `loading/base.py`, `loading/pdf_loader.py`, `loading/markdown_loader.py`, `embeddings/*`, and `models.py` stay byte-identical (Requirements 18.1, 18.8, 18.9). Exactly two pre-existing modules change, both additively: `config.py` and `stores/factory.py`.
- **Every property test runs at least 100 generated examples** and reports the seed and shrunk input on failure (Requirement 19.3). Properties that touch a Chroma collection run 100 under the `chroma` profile; pure properties run 200 under the Phase 1 `pure` profile.

## Tasks

- [ ] 1. Mon–Tue — Dependency pin and repository configuration

  - [ ] 1.1 Pin `chromadb` and register the Phase 2 test profile in `pyproject.toml`
    - Add an exact `chromadb==<version>` pin; add no hosted vector database client package and no pre-built retrieval, re-ranking, or prompt-orchestration dependency
    - Register the `chroma` Hypothesis profile (`max_examples=100`, `deadline=None`) and the `slow` marker; keep Phase 1's `testpaths`, `--strict-markers`, and default `pure` profile
    - _Requirements: 2.1, 19.1, 19.3_

  - [ ] 1.2 Extend `.env.example` and `.gitignore` for Phase 2
    - List every environment variable of Requirement 1.1 with its default and permitted range, plus `ASKMYDOCS_EMBEDDING_DIM` with the note that a learner on a default model never needs to set it
    - Exclude `.chroma/`, `logs/`, and `reports/relevance-review*.csv` from version control
    - _Requirements: 1.10, 1.11, 14.7_

  - [ ] 1.3 Write the Phase 2 README sections in `README.md`
    - Install command with the exact pin, the `Ingest_Script` command, the `Query_Script` command; that Chroma needs no API key, that the `Persist_Directory` is safe to delete, and the command sequence that rebuilds the collection after deletion
    - Every Phase 2 environment variable with default and range; the resolved `Retrieval_Log` path, one-JSON-object-per-line format, `Log_Schema_Version` value, every record field name, the 500-character truncation limit, and that the log holds verbatim source text and is git-ignored
    - The whole-suite test command, the conformance-suite-only command, and the `Comparison_Note` path
    - _Requirements: 1.10, 2.5, 2.6, 14.10, 17.8, 19.9_

  - [ ]* 1.4 Write repository-configuration tests in `tests/test_repository_phase2.py`
    - Assert the `chromadb` pin is exact and no hosted client package is declared; assert the `.gitignore` entries; assert `.env.example` names every Requirement 1.1 variable; assert the README contains the install, ingest, query, reset, and test command strings and the log schema field list
    - _Requirements: 1.10, 1.11, 2.1, 2.5, 2.6, 14.10, 19.1, 19.9_

- [ ] 2. Mon–Tue — Configuration extension and offline dimensionality resolution

  - [ ] 2.1 Add `VectorStoreSettings` and `RetrievalSettings` to `src/askmydocs/config.py`
    - Two frozen dataclasses with the documented defaults; append them to `Configuration` as two defaulted fields, last, so every Phase 1 construction site and every Phase 1 test still builds a valid `Configuration` unchanged
    - Add `SUPPORTED_STORE_SELECTIONS`, `SUPPORTED_DISTANCE_METRICS`, and `COLLECTION_NAME_PATTERN`
    - _Requirements: 1.1, 1.2, 18.10_

  - [ ] 2.2 Add the Phase 2 validation and path-resolution phase to `load_configuration`
    - Run after the Phase 1 phases in the design's order: store selection → distance metric → collection name → `Top_K` → `Relevance_Threshold` → path resolution; trim and lower-case before matching selection and metric
    - Reject an unsupported selection, a metric other than `cosine`, an out-of-range or unparsable `Top_K`, an out-of-range or unparsable `Relevance_Threshold` (rejecting `nan` and `inf`), and a `Collection_Name` violating the character set, first-and-last-character, or 3-to-63 length rule, each with the documented message contents and exit status 2
    - Resolve `Persist_Directory`, `Retrieval_Log`, `Source_Manifest`, and `Question_Set` paths against the repository root when not absolute, anchored on the package location rather than the process working directory
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [ ] 2.3 Implement `resolve_dimensionality` in `src/askmydocs/ingest/dimensionality.py`
    - Four-step resolution stopping at the first success: configured `ASKMYDOCS_EMBEDDING_DIM`, `embedder.dimensionality` guarded against provider exceptions, the pinned `KNOWN_MODEL_DIMENSIONALITY` table, then `ConfigurationError` naming the variable and the configured model
    - Issue no probe embedding and no provider network call, so a no-change ingest run still reports zero Embedder calls
    - _Requirements: 1.12, 4.2, 8.6, 9.5_

  - [ ]* 2.4 Write configuration unit tests in `tests/test_config_phase2.py`
    - Cover every default, every boundary of the `Top_K` and `Relevance_Threshold` ranges, the 2/3/63/64-character `Collection_Name` boundaries, the strict-parse rejections (`"5.0"`, `"5e0"`, `"1_0"`), validation ordering, path resolution, and each dimensionality resolution step
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

- [ ] 3. Mon–Tue — Phase 2 error hierarchy and exit statuses

  - [ ] 3.1 Create `src/askmydocs/errors_phase2.py`
    - Define the Phase 2 subtree under Phase 1's roots without editing Phase 1's `errors.py`: the five `StoreError` subclasses, `DuplicateChunkIdError`, `MetadataValueError`, `ManifestError`, the `IngestError` subtree, the `RetrievalError` subtree, `QuestionSetError`, and the `ReviewError` subtree including `InconsistentTopKError`
    - Reuse Phase 1's `VectorLengthError`, `DegenerateVectorError`, `InvalidKError`, and `GuardrailError` unchanged so the conformance suite can assert one expected type per rule for both stores
    - Add `DOCUMENTED_RESET_COMMAND` as the single module constant every `StoreError` message and the README quote
    - _Requirements: 2.7, 2.8, 2.9, 2.10, 4.6, 4.8, 7.5, 8.11, 8.13, 8.14, 8.19, 10.7, 10.8, 10.9, 10.10, 14.9, 15.8, 16.7, 16.8, 16.12_

  - [ ]* 3.2 Write error-mapping tests in `tests/test_errors_phase2.py`
    - Assert each new type's base class, assert the exit-status mapping for statuses 6 through 11, and assert every `StoreError` message quotes `DOCUMENTED_RESET_COMMAND`
    - _Requirements: 2.7, 2.9, 4.6, 4.8, 7.5, 12.6_

- [ ] 4. Mon–Tue — Store_Admin_Interface and InMemoryAdminStore

  - [ ] 4.1 Create `src/askmydocs/stores/admin.py`
    - `StoredItem` with `insertion_index`, `source_path`, and `state_key(float_places=5)`; `CollectionState`; `StoreAdminInterface` with the five Requirement 3.13 operations plus `iter_items` as the single abstract read primitive and `get_by_ids`, `count_by_source_path`, `collection_state` concrete on the ABC
    - Shared `validate_batch` and `validate_metadata` used by both stores, rejecting non-permitted metadata types by `type(value) in PERMITTED_METADATA_TYPES` and rejecting `nan`/`inf` floats, naming every offending key
    - Define the `AdminStore` Protocol; leave `stores/base.py` with exactly its three Phase 1 operations
    - _Requirements: 3.13, 3.19, 18.4_

  - [ ] 4.2 Create `src/askmydocs/stores/memory_admin.py`
    - `InMemoryAdminStore(InMemoryStore, StoreAdminInterface)` as a subclass, so `stores/memory.py` is never edited and `isinstance(store, InMemoryStore)` still holds
    - Implement `iter_items`, `upsert` in build-then-apply order with insertion-index retention for replaced ids, `delete_by_ids`, `delete_by_source_path`, and `reset`
    - _Requirements: 3.13, 3.14, 3.15, 3.16, 3.17, 5.6, 5.7, 18.4, 18.5_

  - [ ]* 4.3 Write `InMemoryAdminStore` example tests in `tests/test_memory_admin.py`
    - Cover index retention across a mixed upsert, delete of an absent id and an absent source path as no-ops, reset emptying the store, and the `0..count-1` insertion-index sequence surviving any mix of upserts
    - _Requirements: 3.15, 3.16, 3.17, 5.6, 5.7_

- [ ] 5. Mon–Tue — Chroma version shim, test fixtures, and ChromaStore construction

  - [ ] 5.1 Create `tests/strategies_phase2.py`
    - `bounded_vector` reaching Requirement 6.3's norm window by rescaling rather than filtering, `chunk_text` capped at 600 characters, `source_path` with non-ASCII characters, `chunk_batch`, `multi_source_batch`, `invalid_batch` tagged by violated rule, `admin_operations`, `corpus`, `corpus_mutation` with the shrink case weighted up, and `label_matrix`
    - Reuse Phase 1's `finite_element`, `vector`, `MIXED_ALPHABET`, and `unicode_text` unchanged; register the `chroma` settings profile
    - _Requirements: 19.3, 19.4_

  - [ ] 5.2 Add Phase 2 fixtures to `tests/conftest.py` and create `tests/fakes_phase2.py`
    - Module-scoped `chroma_store_module` fixture over `tmp_path_factory` with `close()` on teardown, and a per-example `clean_chroma` fixture using `reset()`; set `ANONYMIZED_TELEMETRY=False`; inherit Phase 1's autouse no-network, no-key fixture unchanged
    - Reuse Phase 1's `FakeEmbedder`, `ScriptedProvider`, and `SpyEmbedder` without modification; add only `DroppingStore`, which silently discards a write's final record so Requirement 8.19's detection is reachable
    - _Requirements: 6.8, 19.5, 19.6, 19.7_

  - [ ] 5.3 Create `src/askmydocs/stores/chroma_compat.py`
    - `collection_kwargs(chromadb_module, distance_metric, fingerprint)` returning the pinned version's create keywords with `hnsw:space` set explicitly to the configured value and `hnsw:search_ef` pinned, plus the fingerprint keys; `max_write_batch(client)` clamped to `DEFAULT_MAX_WRITE_BATCH`
    - Receive the module as a parameter rather than importing it, so `chroma_store.py` stays the only module importing `chromadb`
    - _Requirements: 3.18, 4.1, 4.2, 18.3_

  - [ ] 5.4 Implement `CollectionFingerprint` and `ChromaStore.open`/`close` in `src/askmydocs/stores/chroma_store.py`
    - Fingerprint `to_metadata`, `from_metadata` naming every absent or unreadable field, and `differences` returning every differing field with both values
    - `open` performs the design's six ordered checks before any add, upsert, or query: directory type, creation with parents, a real create-and-remove write probe rather than `os.access`, branch-local `import chromadb`, `PersistentClient` with `anonymized_telemetry=False`, then `get_or_create_collection(embedding_function=None, **collection_kwargs(...))` and fingerprint validation; write the fingerprint once at creation and never mutate collection metadata
    - Map each failure to its typed error with the documented message contents, leaving the `Persist_Directory` content and `Collection_State` unmodified; `close()` releases the client so a later store can open the same directory
    - _Requirements: 2.2, 2.3, 2.7, 2.8, 2.9, 2.10, 3.3, 4.1, 4.2, 4.6, 4.8, 12.1, 12.6, 18.3_

  - [ ]* 5.5 Write Chroma construction and failure-table tests in `tests/test_chroma_store.py`
    - One test per failure-table row asserting the type, the exit status, and the message contents; hash every file under the `Persist_Directory` before and after a failed open and assert equality
    - The metric probe: store `v`, query `2·v`, assert the returned distance is within 1e-5 of 0, so a collection silently indexed with `l2` fails loudly; assert `collection._embedding_function is None`; assert the network guard is still armed
    - Cross-process persistence: close the first store, open a second against the same path with a `SpyEmbedder` that fails the test if called, and assert the stored item count and `Collection_State`
    - _Requirements: 2.2, 2.3, 2.7, 2.8, 2.9, 2.10, 3.3, 4.1, 4.2, 4.6, 4.8, 5.1, 12.1, 12.6, 19.7, 19.10_

- [ ] 6. Mon–Tue — ChromaStore write path, query path, and Chroma-specific properties

  - [ ] 6.1 Implement the index cache and the write path in `stores/chroma_store.py`
    - One lazily built `_index: dict[str, IndexEntry]` serving duplicate-id rejection, next insertion index as `1 + max(existing)`, upsert index retention, delete-by-source-path, per-source counts, and the orphan sweep; keep it current on every add, upsert, and delete
    - `add` rejects ids already stored and ids repeated within the batch before any write; `upsert` replaces text, metadata, and vector while retaining the index and leaving the count unchanged; `stored_metadata` adds only the store-owned insertion index, adding no field to `Chunk`
    - Assign insertion indices across the whole batch before writing any segment, then write and delete in consecutive segments of `max_write_batch`; empty batch is a no-op; take an `RLock` around every client call
    - _Requirements: 3.1, 3.2, 3.4, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 5.6, 5.7, 18.8_

  - [ ] 6.2 Implement `query`, `count`, and `score_from_distance` in `stores/chroma_store.py`
    - Validate K (rejecting `bool`), return `[]` at count 0, then validate the vector length against the fingerprint dimensionality and the norm against `ZERO_NORM_THRESHOLD`, all before any Chroma query
    - Fetch `n_results = count` while `count <= EXACT_SEARCH_LIMIT`, else `min(count, k + TIE_MARGIN)`; include documents, metadatas, and distances but not embeddings; convert with `clamp(1 - distance, -1, 1)`; sort on `(-score, insertion_index)` and slice to K
    - Reconstruct each `Chunk` from the returned document and metadata as a straight field mapping
    - _Requirements: 3.5, 3.6, 3.7, 3.8, 3.20, 4.3, 4.7, 5.5, 12.2_

  - [ ]* 6.3 Write property test for the distance-to-score conversion in `tests/test_store_properties_phase2.py`
    - **Property 1: Chroma's Similarity_Score equals the Phase 1 Cosine_Similarity**
    - At least 100 examples (`chroma` profile); oracle is the unmodified Phase 1 `similarity.py`, so this is a cross-implementation comparison; seeded `@example` cases at identical, negated, and orthogonal pairs, a pair whose cosine is exactly 0.5, and dimensionalities 384 and 1536; includes the positive-scalar rescaling clause
    - **Validates: Requirements 4.3, 4.4**
    - _Requirements: 4.3, 4.4, 19.3, 19.4_

  - [ ]* 6.4 Write property test for the score interval
    - **Property 2: Every Similarity_Score is within [-1, 1]**
    - At least 100 examples (`chroma` profile); exact interval assertion, not tolerance-bounded, because the conversion clamps; asserted for both stores
    - **Validates: Requirements 4.7**
    - _Requirements: 4.7, 19.3_

  - [ ]* 6.5 Write property test for round-trip fidelity across a process boundary
    - **Property 3: Round-trip fidelity of text, metadata, and vector, across a process boundary**
    - At least 100 examples (`chroma` profile); character-for-character text equality, all four metadata fields, the two-regime element tolerance, then close, reopen, and assert count and `Collection_State` with a `SpyEmbedder` that fails the test if called; seeded cases for text containing `"`, `\n`, and `\\`, an empty-text chunk, and dimensionality 1536
    - **Validates: Requirements 2.4, 3.2, 5.1, 5.2, 5.3, 5.4**
    - _Requirements: 2.4, 3.2, 5.1, 5.2, 5.3, 5.4, 19.3, 19.4, 19.10_

  - [ ]* 6.6 Write property test for self-retrieval
    - **Property 4: A stored Chunk's own vector retrieves that Chunk**
    - At least 100 examples (`chroma` profile); model the tie group within 1e-9 exhaustively and assert the returned id is the group's minimum-insertion-index member; seeded cases containing duplicate chunk text so the tie branch is reached deliberately
    - **Validates: Requirements 5.5**
    - _Requirements: 5.5, 19.3, 19.4_

  - [ ]* 6.7 Write property test for query determinism
    - **Property 11: Repeated identical queries are deterministic**
    - At least 100 examples (`chroma` profile); identical id sequences and positional score equality within 1e-9, not 1e-5, because the same store answers the same query
    - **Validates: Requirements 3.5, 10.12**
    - _Requirements: 3.5, 10.12, 19.3_

  - [ ]* 6.8 Write property test for insertion-index assignment under segmentation
    - **Property 26: Insertion indices are strictly increasing and survive segmentation**
    - At least 100 examples (`chroma` profile); inject a maximum write batch of 7 so nearly every example is segmented; include deletes that remove the highest-indexed items, which is the case separating `1 + max` from `count`
    - **Validates: Requirements 3.4, 3.18**
    - _Requirements: 3.4, 3.18, 19.3_

- [ ] 7. Mon–Tue — Factory branch and Phase 1 pipeline isolation

  - [ ] 7.1 Add the `chroma` branch to `src/askmydocs/stores/factory.py`
    - One import swap to `InMemoryAdminStore` and one branch returning `ChromaStore.open(configuration)` for selection `chroma`; keep the `chroma_store` import local to the branch so selecting `memory` works with no `chromadb` installed and `ChromaUnavailableError` has a single raise site
    - Do not import `chromadb` in this module; this and `config.py` are the only two pre-existing modules Phase 2 changes, and both changes are additive
    - _Requirements: 2.7, 18.3, 18.5_

  - [ ] 7.2 Keep the Phase 1 Pipeline_Script on an In_Memory_Store in `scripts/04_pipeline.py`
    - **No Phase 1 frozen module may be edited.** `chunking.py`, `similarity.py`, `loading/base.py`, `loading/pdf_loader.py`, `loading/markdown_loader.py`, `embeddings/*`, and `models.py` stay byte-identical; do not add a `load_bytes` entry point to any loader and add no field to `Chunk`
    - Because the default store selection is `chroma`, `scripts/04_pipeline.py` must construct `InMemoryStore()` **directly**. If it currently obtains its store from `build_store`, change that one line to a direct construction — scripts are not in the frozen set and the change preserves Phase 1 Requirement 11's output exactly
    - _Requirements: 18.1, 18.6, 18.8_

  - [ ]* 7.3 Write factory and pipeline tests in `tests/test_factory_phase2.py`
    - Assert `chroma` returns a `ChromaStore`, `memory` returns a value that is an `InMemoryStore`, selecting `memory` does not import `chromadb`, and the Pipeline_Script populates an in-memory store and produces the Phase 1 output under a Phase 2 configuration whose selection is `chroma`
    - _Requirements: 18.5, 18.6, 18.10_

- [ ] 8. Checkpoint — both stores construct, write, and query
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Mon–Tue — Store_Conformance_Suite across both stores

  - [ ] 9.1 Build the conformance harness in `tests/test_store_conformance.py`
    - A parameterized `store` fixture over `InMemoryAdminStore` and `ChromaStore`, so every case runs once per store from the same test code, each Chroma case using its own `tmp_path` `Persist_Directory` removed afterwards and issuing no remote provider request
    - Example cases pinning messages, validation order, and documented boundaries, each carrying a `@pytest.mark.criteria(...)` marker; a meta-test asserting the union of markers covers the declared criterion list of Phase 1 Requirement 10 and Phase 2 Requirement 3 and failing with every uncovered criterion; a meta-test asserting the fixture directory is under `tmp_path` and not under the repository
    - _Requirements: 6.1, 6.2, 6.8, 19.6_

  - [ ]* 9.2 Write property test for count and fetchable-id agreement
    - **Property 5: Stored item count equals the number of distinct fetchable chunk ids**
    - At least 100 examples (`chroma` profile); driven from identical operation scripts against both stores; includes deletes of absent ids and paths, and `reset` in the operation alphabet
    - **Validates: Requirements 3.16, 3.17, 5.6, 5.7**
    - _Requirements: 3.16, 3.17, 5.6, 5.7, 6.1, 19.3_

  - [ ]* 9.3 Write property test for differential top-K agreement
    - **Property 6: Chroma_Store and In_Memory_Store agree on top-K ids and scores**
    - At least 100 examples (`chroma` profile); 1 to 50 chunks, K from 1 to 50, positional id comparison with the 1e-5 score-tolerance escape clause and positional score comparison within 1e-5
    - **Validates: Requirements 3.5, 6.3, 6.4**
    - _Requirements: 3.5, 6.1, 6.3, 6.4, 19.3, 19.4_

  - [ ]* 9.4 Write property test for score ordering
    - **Property 7: Scores are monotonically non-increasing**
    - At least 100 examples (`chroma` profile); exact comparison with no tolerance, asserted for both stores
    - **Validates: Requirements 6.5, 10.5**
    - _Requirements: 6.5, 10.5, 19.3, 19.4_

  - [ ]* 9.5 Write property test for prefix consistency across K
    - **Property 8: Results are prefix-consistent across values of Top_K**
    - At least 100 examples (`chroma` profile); `K1 < K2` in 1..50 against both stores, with the score-tolerance escape clause; seeded `@example(K1=3, K2=10)`, which is Requirement 15.7's exact instance
    - **Validates: Requirements 6.6, 15.7**
    - _Requirements: 6.6, 15.7, 19.3, 19.4_

  - [ ]* 9.6 Write property test for equal stored item counts
    - **Property 9: The two stores report equal stored item counts**
    - At least 100 examples (`chroma` profile); shares its generator body with Property 6 and adds the one-line count assertion for a readable failure
    - **Validates: Requirements 6.7**
    - _Requirements: 6.7, 19.3_

  - [ ]* 9.7 Write property test for rejected writes leaving the store unchanged
    - **Property 24: A rejected write leaves the store completely unchanged**
    - At least 100 examples (`chroma` profile); each generated batch is tagged with the single rule it violates, so the assertion names the expected error type rather than accepting any exception; run against both stores to verify identical behaviour
    - **Validates: Requirements 3.9, 3.11, 3.12, 3.14, 3.19**
    - _Requirements: 3.9, 3.11, 3.12, 3.14, 3.19, 6.1, 19.3_

  - [ ]* 9.8 Write property test for upsert replacement and index retention
    - **Property 25: Upsert replaces content and retains the insertion index**
    - At least 100 examples (`chroma` profile); mixed batches of already-stored and new ids against both stores; assert retained index, replaced text/metadata/vector within tolerance, and a count delta equal to the number of previously absent ids
    - **Validates: Requirements 3.15**
    - _Requirements: 3.15, 6.1, 19.3_

  - [ ]* 9.9 Write the recall-floor integration test in `tests/test_store_conformance.py`
    - One `slow`-marked test at 1200 stored items and K 10 over several query vectors, using `InMemoryAdminStore` as the exhaustive oracle; report the measured overlap count and the proportion whether or not the assertion passes, and assert `overlap >= floor(0.95 × K)` in place of chunk-id sequence equality
    - _Requirements: 3.20, 6.9_

- [ ] 10. Mon–Tue — Ingest hashing, reading, manifest, and the pure planner

  - [ ] 10.1 Implement `src/askmydocs/ingest/hashing.py`
    - `read_and_hash(path)` returning the bytes and the lower-case hex SHA-256 of exactly those bytes from one sequential read pass, plus `hash_bytes`
    - _Requirements: 7.1, 7.8, 7.9_

  - [ ] 10.2 Implement `src/askmydocs/ingest/reading.py`
    - `document_from_bytes(discovered, data, reporter)` building a Phase 1 `Document` from bytes already in memory, composing Phase 1's public `decode_utf8` and `normalize_newlines` for markdown and constructing `PdfReader(io.BytesIO(data))` for PDFs, joining page text with a single line feed
    - Edit no Phase 1 loader; preserve Phase 1's per-file isolation behaviour for unopenable, unparsable, encrypted, oversize, and whitespace-only files
    - _Requirements: 7.1, 9.7, 18.1, 18.9_

  - [ ] 10.3 Implement `src/askmydocs/ingest/manifest.py`
    - `ManifestEntry` with all seven Requirement 7.2 fields and `SourceManifest`; `to_json_bytes` with `sort_keys=True`, `ensure_ascii=False`, and a single trailing newline so identical content serializes byte-identically
    - `load_manifest` treating an absent file as empty and raising `ManifestError` with the path, the offending key, and the reset command for unparsable JSON or a missing field; `save_manifest` writing a temp file in the same directory, `fsync`, then `os.replace`, and opportunistically removing stale `*.tmp*` siblings
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ] 10.4 Implement `src/askmydocs/ingest/planner.py`
    - `SourceClass`, `IngestPlan`, and a pure `classify(discovered, hashes, manifest, stored_counts, configuration)` performing no I/O
    - Compare all five fields — content hash, chunk size, chunk overlap, provider, model name — before declaring a source unchanged; demote an otherwise-unchanged source to `CHANGED` and record it in `reconciled` when the recorded chunk count disagrees with the stored count; classify manifest paths absent from discovery as `DELETED`; collect stored source paths with no manifest entry as `orphans`
    - _Requirements: 8.1, 8.9, 8.16, 8.17_

  - [ ]* 10.5 Write manifest unit tests in `tests/test_manifest.py`
    - Absent file, unparsable JSON, each missing-field case, the atomic replace leaving previous content intact when `os.replace` raises, and the full field set of a written entry
    - _Requirements: 7.2, 7.4, 7.5, 7.6_

  - [ ]* 10.6 Write the loader-equivalence test in `tests/test_ingest_examples.py`
    - For every committed fixture file assert `document_from_bytes(d, path.read_bytes(), r) == foundation_loader.load(d, r)`, so the duplicated PDF logic cannot drift from the frozen Phase 1 loader
    - _Requirements: 7.1, 18.1, 18.9_

  - [ ]* 10.7 Write property test for the content hash in `tests/test_ingest_properties.py`
    - **Property 17: Content_Hash is deterministic and content-sensitive**
    - At least 100 examples; runs 200 under the `pure` profile with no store and no filesystem; also asserts every digest matches `^[0-9a-f]{64}$`, since the manifest comparison is a string equality
    - **Validates: Requirements 7.8, 7.9**
    - _Requirements: 7.8, 7.9, 19.3_

  - [ ]* 10.8 Write property test for manifest serialization
    - **Property 18: Manifest serialization is deterministic and code-point ordered**
    - At least 100 examples; runs 200 under the `pure` profile; draws non-ASCII source paths so `sort_keys=True` and `ensure_ascii=False` are pinned together; asserts byte-identical re-serialization and a load-then-save round trip
    - **Validates: Requirements 7.3**
    - _Requirements: 7.3, 19.3_

- [ ] 11. Mon–Tue — Ingest runner, recovery, and reporting

  - [ ] 11.1 Implement the orchestration path in `src/askmydocs/ingest/runner.py`
    - Open the store, honour the reset option, load the manifest, run the startup orphan deletion then the reconciliation, discover and hash each file in one read pass, classify, process deleted sources first, chunk every new and changed source, and enforce `Max_Chunks_Per_Run` before any Embedder call
    - Per-source commit loop in ascending source path order: unconditional delete-before-write, embed in segments of `Max_Batch_Size`, upsert, verify the per-source stored count, re-hash from a further read pass, then write the manifest entry atomically
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.10, 8.11, 8.12, 8.15, 8.16, 8.17, 8.18, 7.1, 7.2, 7.10_

  - [ ] 11.2 Implement rollback and run reporting in `ingest/runner.py`
    - `rollback` in the fixed order delete items → drop entry → persist manifest, so a crash during rollback lands on the recoverable side; terminate with exit 8 naming the source path, the reason, and the count of sources committed before the failure, leaving committed entries in place; on a mid-run byte change warn, roll that source back, and continue
    - `IngestReport` printing all twelve counts of Requirement 9.1 plus elapsed seconds to one decimal, the resolved absolute `Persist_Directory`, and the `Collection_Name`; per-batch progress lines; all output through the Phase 1 `Reporter` so the API key is redacted
    - _Requirements: 7.11, 8.13, 8.14, 8.19, 9.1, 9.2, 9.3, 9.4, 9.6_

  - [ ]* 11.3 Transcribe worked scenarios A–F into `tests/test_ingest_examples.py`
    - One test per scenario with the document's exact counts, including the shrink case that leaves stale chunks without delete-before-write, the five interrupted-run variants, and the mid-run byte change; use `ScriptedProvider` and `DroppingStore` for the injected faults
    - _Requirements: 7.10, 7.11, 8.4, 8.5, 8.12, 8.13, 8.14, 8.16, 8.17, 8.18, 8.19, 9.1, 9.5_

  - [ ]* 11.4 Write property test for ingest idempotence in `tests/test_ingest_properties.py`
    - **Property 13: Ingest is idempotent**
    - At least 100 examples (`chroma` profile); asserts unchanged `Collection_State`, unchanged count, byte-identical manifest, and **zero** recorded Embedder segments on the second run, plus the configuration-drift and reconciliation classifications against the pure `classify`, and that every batch carries at most `Max_Batch_Size` texts
    - **Validates: Requirements 8.2, 8.6, 8.9, 8.10, 8.16**
    - _Requirements: 8.2, 8.6, 8.9, 8.10, 8.16, 9.5, 19.3, 19.4_

  - [ ]* 11.5 Write property test for incremental ingest equalling a full rebuild
    - **Property 14: Incremental ingest equals a full rebuild**
    - At least 100 examples (`chroma` profile); 1 to 4 drawn mutations each followed by an incremental run, compared against reset-plus-one-run in a second `Persist_Directory` on `Collection_State`, count, and manifest bytes
    - **Validates: Requirements 8.7, 8.15**
    - _Requirements: 8.7, 8.15, 19.3, 19.4_

  - [ ]* 11.6 Write property test for the manifest-sum invariant
    - **Property 15: The manifest chunk counts sum to the stored item count**
    - At least 100 examples (`chroma` profile); draws a fault injection of none, an Embedder failure, a dropped record, or a mid-run byte change, and asserts the invariant both after the run completes and again after the next run's startup sweep, verifying the convergence claim
    - **Validates: Requirements 7.7, 8.14, 8.17, 8.19**
    - _Requirements: 7.7, 8.14, 8.17, 8.19, 19.3, 19.4_

  - [ ]* 11.7 Write property test for the stored id and source-path sets
    - **Property 16: The stored id and source-path sets exactly match the current corpus**
    - At least 100 examples (`chroma` profile); shrink mutation weighted up and orphans pre-seeded directly into the store so the sweep clause is reachable on every example; four set equalities, one per clause
    - **Validates: Requirements 8.4, 8.5, 8.8, 8.17**
    - _Requirements: 8.4, 8.5, 8.8, 8.17, 19.3_

- [ ] 12. Checkpoint — ingest is incremental, idempotent, and crash-convergent
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Wed–Thu — Retrieval log record and writer

  - [ ] 13.1 Implement the record schema in `src/askmydocs/retrieval/logging.py`
    - `LOG_SCHEMA_VERSION`, `LOG_TEXT_LIMIT`, and `build_record(result, timestamp)` producing every field of Requirements 14.2 and 14.3 plus the `outcome` string and the per-hit `text_truncated` and `below_threshold` marks
    - `text_length` is the full untruncated code-point count; `text` holds the first 500 code points with the marker appended only when longer
    - _Requirements: 14.2, 14.3, 11.7, 12.7_

  - [ ] 13.2 Implement `RetrievalLogWriter` in `retrieval/logging.py`
    - `threading.Lock` around the whole append; create missing parent directories; `os.open` with `O_WRONLY | O_CREAT | O_APPEND | getattr(os, "O_BINARY", 0)` and one `os.write` of the UTF-8 encoded line, checking the returned byte count; never read, seek, or truncate
    - Serialize with `json.dumps` without `indent` so no chunk text can split a record across lines; apply `redact()` to the whole serialized line; raise `RetrievalLogError` naming the resolved absolute path and the reason
    - _Requirements: 14.4, 14.5, 14.6, 14.7, 14.9, 14.12_

- [ ] 14. Wed–Thu — Retriever, outcomes, and the relevance threshold

  - [ ] 14.1 Define the retrieval models in `src/askmydocs/retrieval/retriever.py`
    - `RetrievalOutcome` as a closed three-value enum, `ScoredHit` wrapping Phase 1's `SearchHit` with `below_threshold`, and `RetrievalResult` carrying hits, outcome, `top_score`, threshold, collection count, and provenance, with `search_hits` and `has_relevant_context`
    - Add no field to Phase 1's `SearchHit` or `Chunk`; depend on `VectorStoreInterface`, never on `chroma_store`
    - _Requirements: 10.2, 10.11, 11.2, 12.3, 18.1, 18.3, 18.8_

  - [ ] 14.2 Implement `retrieve` and `retrieve_with_embedding`
    - Validate `Top_K` then the question's non-whitespace content then its code-point length against `Max_Input_Length`, each before any Embedder call; embed exactly once; short-circuit an empty collection to `EMPTY_COLLECTION` with zero hits and no top score, skipping the threshold check entirely
    - Decide the outcome on `top_score >= threshold` and the per-hit mark on `score < threshold`; return `min(Top_K, count)` hits, omitting none on the ground of its score; append exactly one log record as the last step before returning, so a failure before a result writes nothing
    - `retrieve_with_embedding` exists so the Top-K experiment can reuse one question embedding across K while each invocation still writes its own record
    - _Requirements: 10.1, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 11.1, 11.2, 11.7, 11.8, 12.2, 12.3, 12.7, 14.1, 14.11_

  - [ ]* 14.3 Write Retriever unit tests in `tests/test_retriever.py`
    - The three outcomes, the inclusive threshold boundary, exactly one Embedder call per question, the default `Top_K` path, each error message's contents, and the empty-collection reporting of no top score and no no-relevant-context outcome
    - _Requirements: 10.1, 10.2, 10.3, 10.6, 10.7, 10.8, 10.9, 10.10, 11.1, 11.8, 12.2, 12.3_

  - [ ]* 14.4 Write property test for K clamping
    - **Property 10: K clamping**
    - At least 100 examples (`chroma` profile); batch sizes drawn from 0 so the empty-collection case is generated rather than special-cased; asserted at the store level for both stores and at the Retriever level; seeded at `count = 0`, `k = count`, `k = count + 1`, and `k = 100`
    - **Validates: Requirements 3.6, 10.4, 11.2, 12.2**
    - _Requirements: 3.6, 10.4, 11.2, 12.2, 19.3, 19.4_

  - [ ]* 14.5 Write property test for the relevance outcome
    - **Property 12: The relevance outcome is exactly determined by the top score and the threshold**
    - At least 100 examples (`chroma` profile); biconditional assertion, the per-hit mark, the hit count regardless of outcome, and the empty-collection triple; seeded `@example` cases setting the threshold to exactly the observed top score, and at -1.0, 0.0, and 1.0
    - **Validates: Requirements 10.11, 11.1, 11.2, 11.3, 11.8, 12.3**
    - _Requirements: 10.11, 11.1, 11.2, 11.3, 11.8, 12.3, 19.3, 19.4_

  - [ ]* 14.6 Write property test for invalid retrieval input
    - **Property 23: Invalid retrieval input raises before any side effect**
    - At least 100 examples; runs 200 under the `pure` profile against a spy store; whitespace drawn from the full Unicode whitespace set including `\u00a0`, `\u2028`, and `\u3000`; asserts the expected type, no Embedder call, no store query, and byte-identical log content
    - **Validates: Requirements 10.7, 10.8, 10.9, 14.11**
    - _Requirements: 10.7, 10.8, 10.9, 14.11, 19.3_

- [ ] 15. Wed–Thu — Command-line query script

  - [ ] 15.1 Implement `scripts/06_query.py`
    - Thin `main(argv) -> int` taking a positional question and `--top-k`; check `Path.exists()` on the resolved `Persist_Directory` before any store construction and print the path plus the ingest command, exiting 0
    - Print the header block and, per hit, the one-based rank, source path, chunk id, score to four decimals, and chunk text truncated at 200 characters with a marker; print exactly one outcome line whose wording differs across the three cases, with below-threshold hits under their own heading; map each typed error to its exit status and print no result
    - Route all output through the Phase 1 `Reporter` so the API key is redacted
    - _Requirements: 11.4, 11.5, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9_

  - [ ]* 15.2 Write Query_Script tests in `tests/test_scripts_phase2.py`
    - Missing question usage line and non-zero exit, out-of-range `--top-k`, the three distinct outcome lines, the 200-character truncation marker, the missing-`Persist_Directory` path, and redaction of a planted key
    - _Requirements: 11.4, 11.5, 12.4, 12.5, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9_

- [ ] 16. Checkpoint — retrieval returns scored hits and reports absence
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Wed–Thu — Question set and Top-K tuning experiment

  - [ ] 17.1 Write the Question_Set file `question-sets/retrieval-questions.txt`
    - UTF-8, one question per non-empty line, 5 to 10 distinct questions over the sample notes, including at least one question expected to fall below the `Relevance_Threshold`
    - _Requirements: 1.2, 15.9, 15.11_

  - [ ] 17.2 Implement `src/askmydocs/evaluation/question_set.py`
    - Load, decode, trim, de-duplicate, and number: `Question_Identifier` is the one-based position over the distinct non-empty lines in file order, so identifiers agree across the report, the review file, and the precision report
    - Terminate with `QuestionSetError` for an absent, unopenable, undecodable, or empty file, naming the resolved absolute path, the reason, and the expected format; warn on a count outside 5 to 10 and on duplicate text, naming every occurrence's line number, and continue
    - _Requirements: 15.8, 15.9, 15.11_

  - [ ] 17.3 Implement `src/askmydocs/evaluation/topk.py`
    - One embedding per distinct question reused across K in the ordered list 3, 5, 10 via `retrieve_with_embedding`, with three real store queries per question rather than one slice
    - `TopKObservation` per combination with returned count, highest, lowest, mean to four decimals, below-threshold count, and distinct source count; the per-K aggregates; report rendering that carries an existing tradeoff section forward and otherwise emits a placeholder with a word-count reminder
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.6, 15.12_

  - [ ] 17.4 Implement `scripts/07_topk_experiment.py`
    - Print the no-chunks line and the ingest command and exit 0 with the report unchanged when the collection is empty; otherwise print the statistics and write `reports/topk-experiment.md` to a temp file in the report's directory then `os.replace`
    - _Requirements: 15.2, 15.5, 15.10_

  - [ ]* 17.5 Write Top-K experiment tests in `tests/test_topk_experiment.py`
    - Empty-collection path, the absent and empty question-set errors, the count and duplicate warnings, the atomic replace preserving a previous report when rendering raises, the K=3 prefix of K=10 over the real question set, and the structural checks on the tradeoff section: present, at least 100 words, a value of K named
    - _Requirements: 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11_

  - [ ]* 17.6 Write property test for Embedder call accounting
    - **Property 21: Embedder calls equal the distinct question count**
    - At least 100 examples; runs 200 under the `pure` profile driven against `InMemoryAdminStore`; asserts `n` rather than `3n` Embedder calls across three values of K, first-occurrence processing order, and identifiers `1..n`; draws lines differing only by surrounding whitespace
    - **Validates: Requirements 10.6, 15.1, 15.11, 15.12**
    - _Requirements: 10.6, 15.1, 15.11, 15.12, 19.3_

- [ ] 18. Wed–Thu — Relevance review and precision at K

  - [ ] 18.1 Implement generate mode in `src/askmydocs/evaluation/review.py`
    - One Retriever invocation per question at the configured default `Top_K`, one `ReviewRow` per hit, rows ordered by ascending question identifier then rank
    - CSV via `csv.writer` with `QUOTE_ALL`, `lineterminator="\n"`, and `newline=""`; replace every line feed and carriage return with one space **before** truncating to 300 characters and appending the marker
    - Never modify an existing review file: write new rows to a timestamped sibling, carry over each label matched on `(question identifier, chunk id)` rather than rank, and print both resolved absolute paths and the carried-over count
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [ ] 18.2 Implement score mode and precision in `evaluation/review.py`
    - Validate the whole file before computing, in the fixed order absent file → invalid label → empty label, each naming every offending row and leaving any existing report unchanged; add `InconsistentTopKError` when one question's rows record differing `Top_K` values
    - `Precision_At_K` as the `y` count over the row count present for that question, not over `Top_K`; a question with no rows reports not defined, is excluded from the mean, and is counted among the exclusions; render the `Precision_Report`
    - _Requirements: 16.5, 16.6, 16.7, 16.8, 16.9, 16.10, 16.12, 16.13_

  - [ ] 18.3 Implement `scripts/08_relevance_review.py`
    - `generate` and `score` subcommands as a thin `main(argv) -> int`, mapping each `ReviewError` to exit status 11 and printing through the `Reporter`
    - _Requirements: 16.1, 16.6, 16.7, 16.8, 16.12_

  - [ ]* 18.4 Write relevance review tests in `tests/test_relevance_review.py`
    - Generate mode against an existing file leaving its bytes untouched, the carry-over count, each validation error's message contents and exit status, the row-count denominator when fewer hits than `Top_K` were returned, and the excluded-question reporting
    - _Requirements: 16.1, 16.4, 16.5, 16.6, 16.7, 16.8, 16.12, 16.13_

  - [ ]* 18.5 Write property test for precision at K
    - **Property 22: Precision_At_K is a well-defined proportion**
    - At least 100 examples; runs 200 under the `pure` profile over generated label matrices; asserts the `[0, 1]` bound for every value and the mean, the all-`y` and all-`n` identities, the undefined/excluded/counted triple, and that one empty label among many refuses and leaves the report bytes unchanged; seeded all-`y`, all-`n`, single-row, and all-zero-row cases
    - **Validates: Requirements 16.5, 16.7, 16.9, 16.10, 16.13**
    - _Requirements: 16.5, 16.7, 16.9, 16.10, 16.13, 19.3_

  - [ ]* 18.6 Write property test for the review file round trip
    - **Property 27: Review rows survive a write-and-reparse cycle, and labels carry over by chunk id**
    - At least 100 examples; runs 200 under the `pure` profile; re-parses the written file with `csv.reader` and asserts field-for-field equality, no line feed or carriage return in any text field, the 300-character truncation rule, the sort order, the old file's bytes unchanged, and the `(question identifier, chunk id)` carry-over map
    - **Validates: Requirements 16.2, 16.3, 16.4**
    - _Requirements: 16.2, 16.3, 16.4, 19.3_

- [ ] 19. Checkpoint — measurement scripts produce their reports
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Fri — Retrieval log integrity and concurrency

  - [ ]* 20.1 Write property test for log append integrity in `tests/test_retrieval_log.py`
    - **Property 19: N completed retrievals append exactly N complete, parsable log lines**
    - At least 100 examples (`chroma` profile); pre-existing log content drawn as bytes so the prefix assertion is meaningful; asserts the line-count delta, the full field set of Requirements 14.2 and 14.3 per line, `new_bytes.startswith(old_bytes)`, the untruncated `text_length` with the 500-character truncation rule, and that a planted key appears nowhere while the redaction marker appears where it would have; seeded at text lengths 499, 500, 501 and with text containing a line feed, a double quote, and an astral-plane character
    - **Validates: Requirements 11.7, 12.7, 14.1, 14.2, 14.3, 14.4, 14.5, 14.7, 14.8**
    - _Requirements: 11.7, 12.7, 14.1, 14.2, 14.3, 14.4, 14.5, 14.7, 14.8, 19.3_

  - [ ]* 20.2 Write property test for line count under mixed success and failure
    - **Property 20: Log line count is exactly the completed-invocation count under repetition**
    - At least 100 examples (`chroma` profile); each drawn invocation is valid or invalid by one of the four failure modes; asserts the delta equals the valid count and every line parses, which is what fails if the append is placed before the store query
    - **Validates: Requirements 14.1, 14.11**
    - _Requirements: 14.1, 14.11, 19.3_

  - [ ]* 20.3 Write the concurrent-append integration tests in `tests/test_retrieval_log.py`
    - Threads inside one process sharing one store and one writer, parameterized at 2, 8, and 16 threads and repeated; size each record above `PIPE_BUF` with five hits of 500 characters each so the case `O_APPEND` alone does not protect is the case exercised; assert exactly one parsable line per completed invocation
    - _Requirements: 14.12, 19.11_

- [ ] 21. Fri — Phase 1 isolation and layering verification

  - [ ] 21.1 Create `tests/foundation_baseline.json` and `tests/test_foundation_unmodified.py`
    - Baseline of CRLF-normalized SHA-256 digests for every frozen module, including `models.py`, generated at the Phase 1 completion revision; the test recomputes and fails naming every differing path
    - Add the git cross-check reading each module at the recorded revision via `git show`, skipped with a clear reason when git is unavailable; state in a comment that `config.py` and `stores/factory.py` are deliberately absent from the frozen list, naming Requirements 1.1 and 18.5
    - _Requirements: 18.1, 18.8, 18.9, 18.10_

  - [ ] 21.2 Extend `tests/test_layering_phase2.py`
    - AST-scan the `src/` tree for three rules: no frozen Phase 1 module imports `chromadb`, `askmydocs.stores`, `askmydocs.retrieval`, or `askmydocs.ingest`; `chromadb` is imported by exactly one module, counting plain, from, and `importlib.import_module("chromadb")` forms; `retrieval/retriever.py` does not import `chroma_store`
    - Fail with every offending module and every offending symbol, not just the first; add the import test covering the `retrieval/logging.py` stdlib shadowing risk
    - _Requirements: 18.2, 18.3, 18.7_

- [ ] 22. Fri — Comparison note and Learning_Notes

  - [ ] 22.1 Write `learning-notes/vector-db-comparison.md`
    - Compare Chroma against at least one of LanceDB and Pinecone across local versus hosted, cost, persistence model, index type, metadata filtering, and operational effort; state per database whether it runs locally without an account and whether it needs an API key; name HNSW where it applies; state how metadata filtering is expressed and which Phase 1 `Chunk` metadata fields can be filtered on
    - A section of at least 150 words on how a vector database performs similarity search, covering the shared embedding space, the role of the distance metric, HNSW proximity-graph behaviour as an approximate index, and the latency-versus-recall tradeoff
    - State the selected database and the reason, the condition under which the learner would move to a hosted database, the configured `Distance_Metric`, and the distance-to-score formula; record the path in the README
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8_

  - [ ] 22.2 Write the Phase 2 Learning_Notes sections
    - The `Distance_Metric`, the conversion formula, and the 1e-5 tolerance at which the conversion is verified against the `Similarity_Calculator`
    - The selected `Relevance_Threshold`, one question that produced no relevant context, one question that scored at or above the threshold, and the reason for the value
    - The mean `Precision_At_K`, the `Top_K` at which the review was performed, the labelled row count, and one observation about a chunk labelled `n` with the reason
    - _Requirements: 4.5, 11.6, 16.11_

  - [ ]* 22.3 Write structural checks for the written artifacts in `tests/test_learning_artifacts_phase2.py`
    - Assert the required headings are present, the 150-word and 100-word minimums are met, `HNSW` is named, the conversion formula appears, and a value of K is named; assert structure only, since correctness of the explanations is not machine-checkable
    - _Requirements: 4.5, 11.6, 15.6, 16.11, 17.1, 17.3, 17.5, 17.7_

- [ ] 23. Final checkpoint — whole suite green, offline, and within budget
  - Ensure all tests pass, ask the user if questions arise.
  - Run the whole suite with no API key present and confirm no network request is issued and the wall clock stays under 300 seconds, pulling the design's documented levers in order if it does not.
  - _Requirements: 19.2, 19.7, 19.8_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Every property, unit, and integration test sub-task is marked optional; every implementation and every written repository artifact is not. The two isolation checks in task 21 are **not** optional, because Requirements 18.7 and 18.9 make them deliverables rather than confidence measures.
- No Phase 1 frozen module may be edited at any point: `chunking.py`, `similarity.py`, `loading/base.py`, `loading/pdf_loader.py`, `loading/markdown_loader.py`, `embeddings/*`, and `models.py` stay byte-identical, and no field is added to `Chunk`. Only `config.py` and `stores/factory.py` change, both additively.
- `scripts/04_pipeline.py` must construct `InMemoryStore()` directly. The default store selection is `chroma`, so routing the Pipeline_Script through `build_store` would break Requirement 18.6.
- Every property test runs at least 100 generated examples and reports the seed and shrunk input on failure. `chroma`-profile properties run 100; `pure`-profile properties run 200.
- Each property from the design has exactly one property-based test. Properties 5, 6, 7, 8, 9, 10, 24, and 25 are parameterized over both stores from the same test code, which is Requirement 6.1.
- Three groups are deliberately covered by integration tests rather than properties, as the design's demotion section sets out: the recall floor above the exact search limit (9.9), concurrent log appends (20.3), and the cross-process persistence check (5.5).
- Checkpoints sit after the store layer, after ingest, after the measurement scripts, and at the end.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "2.1", "3.1"] },
    { "id": 1, "tasks": ["1.4", "2.2", "3.2", "4.1"] },
    { "id": 2, "tasks": ["2.3", "2.4", "4.2", "5.1", "5.3"] },
    { "id": 3, "tasks": ["4.3", "5.2", "5.4"] },
    { "id": 4, "tasks": ["5.5", "6.1"] },
    { "id": 5, "tasks": ["6.2"] },
    { "id": 6, "tasks": ["6.3", "7.1", "10.1", "10.2"] },
    { "id": 7, "tasks": ["6.4", "7.2", "10.3", "10.4"] },
    { "id": 8, "tasks": ["6.5", "7.3", "10.5", "10.6"] },
    { "id": 9, "tasks": ["6.6", "9.1", "10.7"] },
    { "id": 10, "tasks": ["6.7", "10.8", "11.1"] },
    { "id": 11, "tasks": ["6.8", "11.2"] },
    { "id": 12, "tasks": ["9.2", "11.3", "13.1"] },
    { "id": 13, "tasks": ["9.3", "11.4", "13.2", "14.1"] },
    { "id": 14, "tasks": ["9.4", "11.5", "14.2"] },
    { "id": 15, "tasks": ["9.5", "11.6", "14.3", "15.1"] },
    { "id": 16, "tasks": ["9.6", "11.7", "14.4", "15.2", "17.1", "17.2"] },
    { "id": 17, "tasks": ["9.7", "14.5", "17.3", "18.1"] },
    { "id": 18, "tasks": ["9.8", "14.6", "17.4", "18.2", "20.1"] },
    { "id": 19, "tasks": ["9.9", "17.5", "18.3", "20.2", "21.1"] },
    { "id": 20, "tasks": ["17.6", "18.4", "20.3", "21.2", "22.1"] },
    { "id": 21, "tasks": ["18.5", "22.2"] },
    { "id": 22, "tasks": ["18.6", "22.3"] }
  ]
}
```
