# Implementation Plan: Week 3 — Generation and Citations

## Overview

Week 3 turns a Week 2 `RetrievalResult` into a cited answer or an honest refusal. Nothing is added to
the storage or retrieval layer. The new work is a `generation/` package of seven modules, one
append-only log writer, one evaluation module, two thin scripts, and the written repository artifacts
the submission needs. The language is Python, as the design specifies throughout.

The build order follows the design's module dependency graph strictly:

`pyproject.toml` / `.env.example` / `.gitignore` → `config.py` `GenerationSettings` (additive) →
`errors_week3.py` → `generation/prompting.py` → `generation/chat_client.py` →
`generation/generator.py` → `generation/citations.py` → `generation/presenter.py` →
`generation/answer_log.py` → `generation/pipeline.py` → `scripts/09_ask.py` →
`evaluation/answer_eval.py` → `scripts/10_evaluate.py` → layering and frozen-module verification →
the README and the written learning artifacts. Nothing is implemented before the module it depends on
exists.

Grouping follows the Week 3 day plan:

- **Mon–Tue** (tasks 1–7): repository configuration, the configuration extension, the error subtree,
  prompt construction and the context budget, the chat client and its doubles, the `AnswerGenerator`,
  and the grounding fixtures — the corpus and the `Evaluation_Question_Set` that carry the
  Positive_Grounding_Question and the Negative_Grounding_Question.
- **Wed** (tasks 9–10): the `CitationValidator`, verification, the presenter, and the `Source_List`.
- **Thu** (tasks 12–15): the `Answer_Log`, `answer_question`, the `Ask_Script`, and the evaluation run
  in both modes.
- **Fri–Sat** (tasks 17–19): layering and frozen-module verification, the exit-status table in the
  README, the README sections, the written learning artifacts, and the demo shot list.
- **Sun** (task 20): the `Submission_Checklist` and the `Self_Check_Checklist`.

The two grounding integration assertions land in task 13 with the pipeline, because
`answer_question` is what they exercise; Mon–Tue builds the fixture corpus and the grounding questions
they run against.

Four hard constraints apply to the whole plan and are repeated in the tasks they bind:

- **`config.py` changes additively only**, once: one new frozen `GenerationSettings` dataclass, one new
  defaulted `Configuration` field appended last, one new validation phase appended after the Week 2
  phase. Every Week 1 and Week 2 setting name, resolved value, and validation behaviour stays unchanged
  (Requirements 1.2, 16.5).
- **`stores/factory.py` needs no change.** Week 3 adds no store. The requirement text permits an edit
  here; the design does not need one, so no task makes one (Requirement 16.5).
- **Every frozen Week 1 and Week 2 module stays byte-identical**: `chunking.py`, `similarity.py`,
  `models.py`, `loading/*`, everything under `embeddings/`, and the Week 2 retrieval and store modules.
  `embeddings/retry.py` is *imported* for `RetryPolicy`, `classify_failure`, and `FailureKind`, and
  gains nothing (Requirements 4.8, 16.5).
- **The default install now needs `OPENAI_API_KEY`.** `ASKMYDOCS_PROVIDER` still defaults to
  `sentence-transformers` and runs offline, but `ASKMYDOCS_CHAT_PROVIDER` defaults to `openai`, so a
  clone that ran Weeks 1 and 2 with no key at all cannot answer in Week 3. This is the only new
  credential the week introduces, it is caught at wiring time by `build_chat_client` rather than at
  configuration parse time, and `.env.example` and the README both say so in as many words
  (Requirements 1.7, 1.8, 4.4, 12.4, 13.2).

Every property test uses Hypothesis, runs at least 100 generated examples, carries the tag comment
`# Feature: generation-citations, Property N: <property text>`, and implements exactly one design
property per test function. Property 1 runs 200, as the design specifies.

## Tasks

- [ ] 1. Mon–Tue — Repository configuration, dependency pins, and ignores

  - [ ] 1.1 Pin the chat dependency and register the Week 3 test profile in `pyproject.toml`
    - Keep the existing exact `openai==<version>` pin as the chat completions client dependency and add no new third-party package; add no orchestration, prompt-template, or re-ranking library
    - Register the `week3` Hypothesis profile with a 500 ms per-test deadline and the `too_slow` health check disabled for the two properties that build prompts near the 200000 code point budget ceiling; keep the Week 1 `pure` and Week 2 `chroma` profiles, `testpaths`, and `--strict-markers` unchanged
    - _Requirements: 16.3_

  - [ ] 1.2 Extend `.env.example` with the six Week 3 variables
    - `ASKMYDOCS_CHAT_PROVIDER` (`openai`), `ASKMYDOCS_CONTEXT_BUDGET` (`12000`, range 1000–200000), `ASKMYDOCS_CHAT_MODEL` (`gpt-4o-mini`), `ASKMYDOCS_ANSWER_LOG` (`logs/answers.jsonl`), `ASKMYDOCS_EVALUATION_QUESTION_SET` (`question-sets/evaluation-questions.txt`), `ASKMYDOCS_EVALUATION_REPORT` (`reports/evaluation.md`) — six variables, not eight, because the run file and the ratings file are derived paths rather than settings
    - State that `OPENAI_API_KEY` is now required by a default install because the `Chat_Provider` defaults to `openai`, independently of `ASKMYDOCS_PROVIDER`, and that every combination of the two providers is permitted
    - _Requirements: 1.1, 1.7, 1.8_

  - [ ] 1.3 Extend `.gitignore` for the Week 3 artifacts
    - Exclude `logs/answers.jsonl` (the `Answer_Log` holds verbatim note text), `reports/evaluation-run.json`, and `reports/evaluation-ratings.csv`; keep the Week 1 and Week 2 entries for `.env`, the personal note files of `sample-notes/` other than its README, the Chroma `Persist_Directory`, the `Retrieval_Log`, and `reports/relevance-review*.csv`
    - Keep `docs/example-answer.png` tracked, which is why the image lives outside `reports/`
    - _Requirements: 10.7, 16.2_

  - [ ]* 1.4 Write the repository-hygiene checks in `tests/test_docs_week3.py`
    - Assert the `openai` pin is exact and that no orchestration or hosted vector database package is declared; assert each Week 3 `.gitignore` entry; assert `.env.example` names all six Week 3 variables with their defaults and the `Context_Budget` range, and holds the `OPENAI_API_KEY` note
    - _Requirements: 1.8, 10.7, 16.2, 16.3_

- [ ] 2. Mon–Tue — Configuration extension, additive only

  - [ ] 2.1 Add `GenerationSettings` to `src/askmydocs/config.py`
    - One new frozen dataclass with the six documented defaults, plus `CONTEXT_BUDGET_MIN`, `CONTEXT_BUDGET_MAX`, and `SUPPORTED_CHAT_PROVIDERS = ("openai",)`
    - Append `generation: GenerationSettings = field(default_factory=GenerationSettings)` to `Configuration` as the last field, so every Week 1 and Week 2 construction site and test still builds a valid `Configuration` unchanged; add nothing above the new code and change no existing field or order
    - Add `evaluation_run_path` and `evaluation_ratings_path` as derived `with_name` properties rather than settings, keeping the new-setting count at exactly six
    - _Requirements: 1.1, 1.2, 1.6, 16.5_

  - [ ] 2.2 Append the Week 3 validation and path-resolution phase to `load_configuration`
    - Run after the Week 2 phase: trim and lower-case `Chat_Provider` then require it to equal `openai`, naming the setting, the rejected value, and the supported value; parse `Context_Budget` strictly in the Week 1 sense (accept `"12000 "`, reject `"12000.0"`, `"1.2e4"`, `"12_000"`, and `True`) and require 1000–200000 inclusive, naming the setting, the value, and the range; require `Chat_Model` non-empty after `strip()`
    - Resolve `Answer_Log`, `Evaluation_Question_Set`, and `Evaluation_Report` against the repository root using the Week 2 package-location anchor and return absolute paths
    - Validate `Chat_Provider` without reading `ASKMYDOCS_PROVIDER`, so local embeddings plus a remote chat model resolves cleanly
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.3 Write configuration tests in `tests/test_config_week3.py`
    - Every Week 3 default; the 999/1000/200000/200001 `Context_Budget` boundaries and the strict-parse rejections; the `Chat_Provider` rejection message contents; the empty `Chat_Model` rejection; path resolution independent of the process working directory; the local-embeddings-plus-`openai`-chat combination resolving with no error and with no key present; a Week 1 and a Week 2 `Configuration` construction site still valid unchanged
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [ ] 3. Mon–Tue — Week 3 error subtree and the single exit-status table

  - [ ] 3.1 Create `src/askmydocs/errors_week3.py`
    - Define the Week 3 subtree under the existing `AskMyDocsError` root without editing `errors.py`: `PromptError` with `ContextBudgetError` and `PromptStateError`; `GenerationError` with `MissingApiKeyError`, `ChatCompletionFailedError`, and `MalformedModelOutputError`; `AnswerLogError`; `EvaluationError` with `QuestionCountError`, `QualityRatingError`, `NoRatingsError`, `RefusalPathMissingError`, and `EvaluationArtifactError`; re-export into the existing root exactly as `errors_week2.py` does
    - `MissingApiKeyError` is a `GenerationError`, not a `ConfigurationError` — the settings are valid, the key is only needed when a chat call is about to happen; a rejected `Chat_Provider` value stays a `ConfigurationError`
    - Add the one shared exit-status mapping for statuses 12 through 22 as a module constant both scripts read, with no per-script override, and keep Requirement 12.1's condition on the Week 1 status 3
    - Raise typed exceptions only; call `sys.exit` from no library module
    - _Requirements: 3.6, 4.4, 10.1, 11.4, 11.5, 11.6, 11.8, 12.3, 12.4, 12.5, 12.6, 12.8, 12.9_

  - [ ]* 3.2 Write error-mapping tests in `tests/test_errors_week3.py`
    - Assert each new type's base class, assert the statuses 12 through 22 map to the conditions of the design's table, and assert the statuses are pairwise distinct and disjoint from the Week 1 and Week 2 ranges
    - _Requirements: 12.8, 12.9_

- [ ] 4. Mon–Tue — Prompt construction and the context budget

  - [ ] 4.1 Create `src/askmydocs/generation/prompting.py` — types, literals, and rendering
    - `SYSTEM_PROMPT` as a module-level `Final` literal holding the design's eight numbered instructions verbatim; `CONTEXT_HEADER`, `QUESTION_HEADER`, `ENTRY_TEMPLATE`, `PROMPT_TRAILER`, and `CITATION_MARKER_PATTERN = re.compile(r"\[([0-9]{1,9})\]")` using the explicit ASCII class rather than `\d`
    - Frozen `CitationEntry` (number, source path, chunk index, start and end offsets, score, carrying **no** chunk text), `CitationTable` asserting numbers are exactly `1..n` with `has`, `numbers`, `__getitem__`, `__len__`, and `AssembledPrompt` with `prompt_length` and `prompt_hash` as properties and the Requirement 3.1 bound asserted in `__post_init__`
    - `render_context_block`, `render_user_prompt`, `prompt_length` as code points via `len()`, and `build_citation_table` as the single number-assignment site
    - Import `RetrievalResult`, `RetrievalOutcome`, `ScoredHit`, and `Chunk` and nothing from `askmydocs.stores` or `chromadb`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.8_

  - [ ] 4.2 Implement `PromptBuilder.build` and the budget trim in `src/askmydocs/generation/prompting.py`
    - Require `RELEVANT_CONTEXT` and raise `PromptStateError` otherwise; scan `n` downward from `len(hits)`, re-rendering the user prompt each step rather than subtracting, and stop at the largest fitting prefix
    - Drop from the lowest-scoring tail so every retained number keeps the number it already had; report `dropped_chunk_count` as the complement and `0` when nothing was dropped
    - At `n == 1` still over budget, raise `ContextBudgetError` naming the `Context_Budget`, the `Prompt_Length`, and the `Chunk_Size`, with the remedy arithmetic the design states
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7_

  - [ ] 4.3 Create `tests/strategies_week3.py`
    - `chunk_texts` deliberately including `[`, `]`, newlines, non-ASCII characters, and the literal strings `Context:` and `Question:`; `source_paths` with intended duplicates; `scores`, `chunks` with `0 <= start <= end`, `scored_hits`, `hit_sequences` (1–8), `questions`, `budgets` over 1000–200000, `outcomes`, `marker_multisets` over 0–20, `api_keys`, `reported_counts` including `None` and negatives, and `ratings`
    - Add the budget densification helper that draws half the budgets between "only the top hit fits" and "everything fits", plus explicit `@example` seeds at 10 and 11 hits for the entry-number digit boundary
    - Reuse the Week 1 and Week 2 strategies unchanged; register nothing that alters the existing profiles
    - _Requirements: 2.2, 2.3, 3.1, 3.2_

  - [ ]* 4.4 Write prompt example tests in `tests/test_prompting_examples.py`
    - **Pin the `System_Prompt`**: assert `len(SYSTEM_PROMPT) == 766` and assert its SHA-256 digest, with a comment stating that editing the literal is a behaviour change — every existing `Prompt_Hash` stops matching and evaluation runs before and after the edit stop being comparable, so a deliberate edit must update the pin and regenerate the report
    - Assert the exact `Context_Block` layout, the eight instruction lines of Requirement 2.4, worked example A (five hits, one dropped, `dropped_chunk_count == 1`) and worked example B (`Context_Budget = 1000`, `Chunk_Size = 500`, `ContextBudgetError` naming all three numbers)
    - _Requirements: 2.4, 2.6, 3.4, 3.6_

  - [ ]* 4.5 Write property test for the Context_Block contents in `tests/test_prompting_properties.py`
    - **Property 2: The Context_Block holds only supplied chunk text, each chunk exactly once**
    - At least 100 examples; exact string equality against an independently reconstructed expected value, plus a count check that each dropped hit's text is absent unless a supplied chunk carries the same text; chunk texts include bracket characters, newlines, and the literal `Context:` and `Question:` so a delimiter-based implementation fails
    - **Validates: Requirements 2.1, 2.3, 2.5**
    - _Requirements: 2.1, 2.3, 2.5_

  - [ ]* 4.6 Write property test for the context budget bound in `tests/test_prompting_properties.py`
    - **Property 3: `Prompt_Length` never exceeds the `Context_Budget`**
    - At least 100 examples; preconditioned on the single top hit fitting, with half the budgets densified into the interesting region; asserted on the returned value, which the `AssembledPrompt.__post_init__` bound makes a type invariant as well
    - **Validates: Requirements 3.1, 3.2**
    - _Requirements: 3.1, 3.2_

  - [ ]* 4.7 Write property test for prefix trimming in `tests/test_prompting_properties.py`
    - **Property 4: The retained chunk set is always a prefix, and the drop count is its complement**
    - At least 100 examples; asserts the prefix identity, `dropped_chunk_count == len(hits) - len(supplied_chunks)`, the zero case, and **maximality** by rendering over one additional hit and checking it exceeds the budget — which is what catches an implementation that trims one hit too many
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.7**
    - _Requirements: 3.2, 3.3, 3.4, 3.7_

  - [ ]* 4.8 Write property test for prompt determinism in `tests/test_prompting_properties.py`
    - **Property 9: Prompt construction is deterministic for the same `RetrievalResult` and `Configuration`**
    - At least 100 examples; two separately constructed `PromptBuilder` instances so a builder that memoizes state cannot pass; asserts identical system prompt, user prompt, `Citation_Table`, and `Prompt_Hash`; chunk texts and source paths include duplicates, the usual source of accidental nondeterminism
    - **Validates: Requirement 2.6**
    - _Requirements: 2.6_

- [ ] 5. Mon–Tue — Chat client and the scripted test doubles

  - [ ] 5.1 Create `src/askmydocs/generation/chat_client.py`
    - Frozen `ChatCompletion` with nullable `content` and nullable token counts; the `ChatClient` Protocol; `OpenAIChatClient` issuing one `chat.completions.create` call with two messages system-then-user, `temperature=0.0`, and the `Request_Timeout` as the `timeout` keyword; no `max_tokens`, no `seed`, no `response_format`, no tools
    - A tolerant `_adapt` that folds "no `choices`" and "no `message`" into `content is None`, keeping provider-shape handling in this module
    - `build_chat_client` selecting on `configuration.generation.chat_provider` alone, never on the `Embedding_Provider`, and raising `MissingApiKeyError` naming `OPENAI_API_KEY` and the setting name `Chat_Provider` before any request is issued
    - This is the only Week 3 module that imports `openai`
    - _Requirements: 4.1, 4.2, 4.4, 1.7, 12.4_

  - [ ] 5.2 Create `tests/fakes_week3.py`
    - `FakeChatModel(replies)` implementing `ChatClient`, consuming queued `ChatCompletion` values in order with the last repeating, and recording every call's `model`, `system_prompt`, `user_prompt`, and `timeout` so "the `Request_Timeout` was applied" is asserted without a clock; `FailingChatModel(errors, then=...)` raising a scripted exception sequence before delegating; a `reply(text, prompt_tokens=, completion_tokens=)` helper
    - Script the whole answer library the design tabulates: a **well-cited answer** (`"... [1]."`), a **cited subset** of a four-entry table, a **dangling marker** (`[7]` against four entries), the `[0]` case, mixed valid-and-dangling with a duplicate, **no markers at all**, the model's own `"I don't know based on the provided notes."` fallback, the near-miss forms `[ 1 ]` / `[1,2]` / `[source]`, **malformed output** as both `content is None` and `"   \n\t "`, the token zero-fill and negative-count replies, two transient failures then success, retry exhaustion, and a terminal auth failure
    - Construct `RetryPolicy` with `sleep=lambda _: None` everywhere, so backoff is exercised as recorded decisions rather than elapsed time; construct no `OpenAIChatClient` and read no real key
    - _Requirements: 4.2, 4.5, 4.6, 4.7, 7.2, 7.3, 12.3, 12.6_

- [ ] 6. Mon–Tue — Answer generation

  - [ ] 6.1 Create `src/askmydocs/generation/generator.py`
    - Frozen `TokenUsage` with `total_tokens` as `init=False` derived in `__post_init__`, rejecting negatives and non-ints, plus `TokenUsage.zero()`; frozen `GeneratedAnswer`
    - `normalize_usage` zero-filling only when all three reported counts are absent, coercing a negative or non-integer count to 0, and never reading a reported total
    - `AnswerGenerator.generate` wrapping one `client.complete(...)` in `RetryPolicy.run`, importing `RetryPolicy` and `classify_failure` from the frozen `embeddings/retry.py` and adding nothing to it, and translating post-exhaustion `EmbeddingFailedError` into `ChatCompletionFailedError` naming the `Chat_Model`, the attempt count, and the final failure category in one `except` clause
    - Detect `Malformed_Model_Output` **outside** the retry wrapper, in order `content is None` then `content.strip() == ""`, raising `MalformedModelOutputError` naming the `Chat_Model` and which condition applied — never retried, because a successful HTTP call is not a transport fault
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7, 4.8, 12.3, 12.6_

  - [ ]* 6.2 Write generator example tests in `tests/test_generator.py`
    - The four retry scenarios (two transient then success, exhaustion, terminal no-retry, and the client's own timeout exception type); the recorded `timeout` value equalling the `Request_Timeout`; the two-message order and `temperature=0.0`; both malformed-output conditions raising with no second attempt; `ChatCompletionFailedError` message contents; `build_chat_client` raising `MissingApiKeyError` with the key absent and with it blank, having recorded zero calls
    - _Requirements: 4.1, 4.2, 4.4, 12.3, 12.4, 12.6_

  - [ ]* 6.3 Write property test for token usage in `tests/test_generator_properties.py`
    - **Property 8: `Token_Usage` counts are non-negative and the total equals the sum of its parts**
    - At least 100 examples over all eight absence combinations including negatives; asserts both directions of the zero-fill biconditional — all-absent gives `(0, 0, 0)`, any present count is preserved — and `total_tokens == prompt_tokens + completion_tokens`
    - **Validates: Requirements 4.5, 4.6, 4.7**
    - _Requirements: 4.5, 4.6, 4.7_

- [ ] 7. Mon–Tue — Grounding fixtures and the evaluation question set

  - [ ] 7.1 Write the `Evaluation_Question_Set` at `question-sets/evaluation-questions.txt`
    - UTF-8, one question per non-empty line, at least 10 distinct non-empty lines over the committed sample notes, unchanged in format from Week 2
    - Include at least one Positive_Grounding_Question whose answer appears in the notes and nowhere in the `System_Prompt`, and at least one Negative_Grounding_Question whose answer appears nowhere in the notes, so the refusal path is exercised and score mode's `RefusalPathMissingError` check passes
    - _Requirements: 5.1, 5.2, 5.3, 11.6_

  - [ ] 7.2 Commit the grounding fixture corpus and confirm the sample notes answer the README example
    - Add the small three-file fixture notes folder the grounding integration tests ingest into a temporary Chroma collection with the Week 1 `FakeEmbedder`
    - Confirm `sample-notes/` holds small non-sensitive placeholder notes sufficient to answer the README example question, with only its README tracked for personal files
    - _Requirements: 5.1, 5.2, 16.1_

- [ ] 8. Checkpoint — prompt, budget, and generation verified against fakes
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Wed — Citation validation and the source list

  - [ ] 9.1 Create `src/askmydocs/generation/citations.py`
    - Frozen `CitationReport` holding `markers` in document order with duplicates kept, `cited` as the ascending distinct set, `dangling` as the ascending unresolvable subset, and `verified = bool(markers) and not dangling`
    - `CitationValidator.extract_markers` over `CITATION_MARKER_PATTERN` and `validate` de-duplicating **before** the table lookup, so a marker repeated ten times yields one `Dangling_Citation`
    - `build_source_list` as one expression: cited numbers filtered to resolvable, de-duplicated, ascending — satisfying one line per cited number, no entry absent from the table, uncited numbers omitted, and ascending order at once
    - Import nothing from `generation/chat_client.py`; this module stays runnable with no client at all
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.2 Write the marker non-match table tests in `tests/test_citations.py`
    - One case per row of the design's table: `[1]` and `[12]` match; `[0]` and an out-of-range number match then dangle; `[ 1 ]`, `[1 ]`, `[1,2]`, `[1-3]`, `[1.5]`, `[1;2]`, `[]`, `[n]`, `[source]`, `[citation needed]`, and a 10-digit run do not match; `[[1]]` matches the inner marker; `[1](https://example.com)` matches; `[١]` and `[三]` do not match, which is why the pattern uses `[0-9]` rather than `\d`
    - _Requirements: 7.1, 7.2_

  - [ ]* 9.3 Write property test for citation resolution in `tests/test_citations.py`
    - **Property 1: Every citation marker of a verified answer maps to exactly one supplied chunk**
    - **At least 200 examples** — double the floor, because this is the property the whole feature exists for; answer texts built by interleaving generated prose with a rendered `marker_multisets` so out-of-range and `[0]` markers occur in roughly half the cases
    - Asserts the table is a bijection onto the supplied prefix with numbers exactly `1..n`, that each resolved entry's source path, ordinal index, offsets, and score equal the chunk's, that `set(markers) == set(cited)`, that `dangling == cited - table.numbers`, and that `verified` implies empty dangling and a `Source_List` whose numbers are a subset of the table's
    - **Validates: Requirements 2.2, 2.7, 6.3, 6.4, 7.4**
    - _Requirements: 2.2, 2.7, 6.3, 6.4, 7.4_

  - [ ]* 9.4 Write property test for the verification biconditional in `tests/test_citations.py`
    - **Property 6: Verification is exactly "at least one marker and no dangling citation"**
    - At least 100 examples, plus three seeded `@example` cases the generator rarely reaches — text with no bracket at all, text whose markers all dangle, and text whose markers are all valid but repeated; the marker-free seed is mandatory because random prose almost never contains `[` and Requirement 7.3 is precisely that case
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.5 Write property test for the source list in `tests/test_citations.py`
    - **Property 7: The `Source_List` holds exactly the cited resolvable numbers, ascending**
    - At least 100 examples; markers generated in shuffled and descending order so a missing `sorted()` fails; asserts no uncited number appears, no cited number appears twice, no unresolvable number appears, and the sequence is strictly ascending
    - **Validates: Requirements 6.2, 6.4, 6.5, 6.6**
    - _Requirements: 6.2, 6.4, 6.5, 6.6_

- [ ] 10. Wed — Presentation, the source list rendering, and refusals

  - [ ] 10.1 Create `src/askmydocs/generation/presenter.py`
    - `RefusalReason` enum with the glossary's hyphenated values, frozen `PresentedAnswer` with `source_list` and a `verified` property, frozen `Refusal` carrying the top score, the threshold, the collection count, and the ingest command; `ReportLine = tuple[Literal["info", "warning"], str]`
    - `present`, `refuse` mapping `RetrievalOutcome` to `RefusalReason` in one small function, `render_answer`, and `render_refusal` — returning **lines, not output**, so every console string still passes through the `Reporter` and its redaction, and the presenter is testable by comparing tuples
    - `render_answer` emits, in order: the dangling-citation or no-citation warning **before** the answer text, the dropped-chunk notice naming the count and the `Context_Budget` when `dropped_chunk_count > 0`, the answer text, a blank line, the `Sources:` heading, one line per entry ascending naming number, source path, ordinal index, offset range, and score, then the verification label and token totals
    - `render_refusal` emits no `Sources:` heading and no source line under any circumstance, the fixed refusal sentence, and then the reason-specific lines: top score plus `Relevance_Threshold` plus the threshold hint for `no-relevant-context`, the stored-chunk count plus the literal `python scripts/05_ingest.py` for `empty-collection`
    - Import nothing from `generation/chat_client.py`
    - _Requirements: 3.5, 6.1, 6.2, 6.6, 7.2, 7.3, 8.4, 8.5, 8.6_

  - [ ]* 10.2 Write presenter tests in `tests/test_presenter.py`
    - The line order with the warning ahead of the answer; the dropped-chunk notice wording; the ascending `Source_List` line format; the `Sources: none cited` line for a marker-free answer; both refusal variants including the exact ingest command and the absence of any source line; the channel of each line
    - _Requirements: 3.5, 6.1, 6.2, 6.6, 7.2, 7.3, 8.4, 8.5, 8.6_

- [ ] 11. Checkpoint — citations resolve, refusals render, no source list leaks into a refusal
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Thu — The Answer_Log

  - [ ] 12.1 Create `src/askmydocs/generation/answer_log.py`
    - `ANSWER_LOG_SCHEMA_VERSION = 1`; frozen `RetrievalSummaryHit` and `AnswerLogRecord` covering every field of the design's schema table, with `to_payload` as the single serialization site and the three constructors `for_refusal`, `for_answer`, and `for_malformed_output` so each zero-fill rule is written once
    - Record the `Prompt_Hash` in place of the prompt text, and hold no chunk text in `retrieval.hits[]`
    - `AnswerLogWriter.append` mirroring the Week 2 `RetrievalLogWriter` structure exactly: `json.dumps` with `ensure_ascii=False` and compact separators, `redact` over the **whole serialized line** before the write so no later-added field can escape, `mkdir(parents=True, exist_ok=True)` inside the lock on every append, one `O_APPEND` descriptor at mode `0o600`, and one `os.write` of one fully-formed line; raise `AnswerLogError` naming the resolved absolute path and the reason
    - Named `answer_log.py`, not `logging.py`, so no stdlib shadowing caveat is needed
    - _Requirements: 7.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 12.2 Write answer-log example tests in `tests/test_answer_log.py`
    - The refusal record's zero token counts, empty answer text, empty citation list, `null` `prompt_hash`, and `null` verification; the malformed-output record with `malformed_model_output: true` and a present `prompt_hash`; the absent parent directory created on the spot; the file mode; an unwritable path raising `AnswerLogError` naming the absolute path
    - _Requirements: 10.5, 10.6, 12.6_

  - [ ]* 12.3 Write property test for the log record round trip in `tests/test_answer_log.py`
    - **Property 10: An `Answer_Log_Record` round-trips through JSON and never carries the prompt text**
    - At least 100 examples; questions and answers containing `"`, `\`, newlines, and non-ASCII characters; asserts field-by-field equality after a parse in both directions, the presence of the `Prompt_Hash`, the absence of the `System_Prompt` and `User_Prompt` strings anywhere in the raw line, and that the recorded verification label and dangling list equal the presented answer's
    - **Validates: Requirements 7.5, 10.2, 10.3**
    - _Requirements: 7.5, 10.2, 10.3_

  - [ ]* 12.4 Write property test for one line per attempt in `tests/test_answer_log.py`
    - **Property 11: Exactly one log line is appended per answer attempt**
    - At least 100 examples over lists of 1–20 records whose text contains `\n` and `\r\n`; asserts the line count equals the attempt count and that every line parses as one JSON object independently — the embedded-newline case is the whole point, and it only holds because `json.dumps` escapes it
    - **Validates: Requirements 9.5, 10.1**
    - _Requirements: 9.5, 10.1_

- [ ] 13. Thu — `answer_question`, the one shared path

  - [ ] 13.1 Create `src/askmydocs/generation/pipeline.py`
    - Frozen `AnswerAttempt` whose `__post_init__` asserts the Requirement 8.1–8.3 biconditional as a constructor precondition: exactly one of `presented` and `refusal` is set, and `prompt is None` whenever `refusal` is set
    - `answer_question(question, *, retriever, generator, configuration, log_writer, source, top_k=None, clock=...)` validating the question, retrieving, branching on `RetrievalResult.outcome` **before any prompt exists** so no refusal path can reach the `AnswerGenerator`, building, generating, validating citations, presenting, and appending exactly one log record as the last step on every branch including the malformed-output branch
    - Raise `PromptStateError` when `generator is None` on a `RELEVANT_CONTEXT` outcome; leave the Week 2 `Retrieval_Log` behaviour untouched
    - _Requirements: 8.1, 8.2, 8.3, 9.1, 9.5, 10.1, 12.6_

  - [ ]* 13.2 Write property test for the refusal biconditional in `tests/test_pipeline.py`
    - **Property 5: Refusal occurs exactly when the outcome is `NO_RELEVANT_CONTEXT` or `EMPTY_COLLECTION`**
    - At least 100 examples over outcomes crossed with hit sequences, deliberately including **inconsistent** inputs — a `NO_RELEVANT_CONTEXT` result carrying hits, an `EMPTY_COLLECTION` result carrying hits — so an implementation that peeks at `len(hits)` instead of the outcome fails
    - Asserts the injected `FakeChatModel` recorded **zero calls** on every refusal, that the rendered lines hold no `Source_List` entry, and that the written record holds the matching `Refusal_Reason`, empty answer text, empty citation list, and three zero token counts
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.6, 10.5**
    - _Requirements: 8.1, 8.2, 8.3, 8.6, 10.5_

  - [ ]* 13.3 Write the two grounding integration tests in `tests/test_pipeline.py`
    - Positive: ingest the fixture corpus into a temporary Chroma collection with the `FakeEmbedder`, assert the outcome is `RELEVANT_CONTEXT` and the top hit's source path is the expected fixture file, then script a `"... [1]."` reply and assert a `Verified_Answer` whose `Source_List` entry lies under the fixture notes folder
    - Negative: run the Negative_Grounding_Question, assert the outcome is `NO_RELEVANT_CONTEXT`, the `Refusal` is produced, and the chat client recorded **zero calls** — the stronger and fully deterministic half
    - _Requirements: 5.1, 5.2, 9.1_

  - [ ]* 13.4 Write property test for key redaction in `tests/test_redaction_week3.py`
    - **Property 13: The API key never appears in the prompt, the log, or any message**
    - At least 100 examples over generated key-shaped strings planted in a monkeypatched environment, crossed with the five attempt paths — successful, refused, retry-exhausted, malformed-output, and budget-exceeded
    - Sweeps **every window of 8 characters or longer**, matching the requirement's wording rather than testing a whole-key `in`, because a partial leak through slicing or truncation is the realistic failure mode; asserts absence from the `Assembled_Prompt`, from every `Answer_Log` line, from the `Configuration_Stamp`, and from every raised exception string and reported console line, and asserts each log line still parses as JSON
    - **Validates: Requirements 4.3, 10.4, 11.7, 12.7, 13.7**
    - _Requirements: 4.3, 10.4, 11.7, 12.7, 13.7_

- [ ] 14. Thu — The Ask_Script

  - [ ] 14.1 Implement `scripts/09_ask.py`
    - Thin `main(argv) -> int` taking a positional question and `--top-k`: load the `Configuration` and the `Reporter`, build the store, embedder, `Retriever`, `AnswerLogWriter`, and `AnswerGenerator` on every run, call `answer_question(..., source="ask")`, render the returned lines through `Reporter.info` / `Reporter.warning` in order, and map typed exceptions to the shared exit-status table
    - `--top-k` overrides the configured `Top_K` for the invocation and is passed straight to the `Retriever`; a missing question prints a usage message naming the required argument and a blank question prints an error naming the empty question, both returning 12 because they are one condition
    - Return 0 on a presented answer and on a presented `Refusal`; print the dropped-chunk notice; on `EMPTY_COLLECTION` with no supported note file, present and log the refusal first, then report the folder path and the supported extensions and return 3
    - Call `sys.exit` from no library module and route every string through the `Reporter`
    - _Requirements: 3.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [ ]* 14.2 Write Ask_Script tests in `tests/test_scripts_week3.py`
    - One case per Week 3 row of the exit-status table reachable from `09_ask.py` — 0, 2, 3, 12 (both forms), 13, 14, 15, 16, 17 — asserting the exact status; the `--top-k` override reaching the `Retriever`; the refusal returning 0; a planted key redacted from every printed line
    - _Requirements: 9.2, 9.3, 9.4, 9.7, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.9_

- [ ] 15. Thu — The evaluation run

  - [ ] 15.1 Implement generate mode in `src/askmydocs/evaluation/answer_eval.py`
    - Load the question set through the unchanged Week 2 `evaluation/question_set.py`, reusing its de-duplication and `Question_Identifier`; raise `QuestionCountError` naming the count and the minimum of 10 **before the first question runs**, so no API spend happens on a run that cannot produce a valid report
    - Call `answer_question(..., source="evaluate")` once per distinct question in file order — the same function the `Ask_Script` calls, which is the whole mechanism behind Requirement 11.1 — so every evaluation question also lands in the `Answer_Log` and the Week 2 `Retrieval_Log`
    - Write `reports/evaluation-run.json` as the machine-owned artifact holding the `Configuration_Stamp`, the timestamp, and per question the identifier, text, outcome, every retrieved `Similarity_Score`, answer text, cited entries, `Prompt_Hash`, `Dropped_Chunk_Count`, and `Token_Usage`; write `reports/evaluation-ratings.csv` with `question_id,question,quality_rating` and the rating column empty
    - Abort the whole run if one question fails, rather than leaving a partial run file whose mean would have an unknown denominator
    - _Requirements: 11.1, 11.2, 11.6_

  - [ ] 15.2 Implement score mode, the `Configuration_Stamp`, and the mean in `src/askmydocs/evaluation/answer_eval.py`
    - `parse_quality_rating` accepting `"3"` and `" 3 "`, treating empty and whitespace as unrated, and raising `QualityRatingError` naming the question and the rejected value for `"0"`, `"6"`, `"-1"`, `"+3"`, `"3.0"`, `"3.5"`, `"three"`, and `"٣"` — the same ASCII-only discipline the citation pattern uses
    - `mean_quality_rating` over the non-empty ratings rounded to two decimal places; `NoRatingsError` when every rating is empty; `EvaluationArtifactError` when a `question_id` is present in one file and absent from the other
    - `build_configuration_stamp` recording every Week 1, Week 2, and Week 3 setting name and resolved value with paths absolute, writing the redaction marker when a key was set and the literal `unset` when none was, and never reading the key value
    - `RefusalPathMissingError` when no recorded outcome was `no_relevant_context` or `empty_collection`; derive the Negative_Grounding_Question as the first refusing question in file order and the Positive_Grounding_Question as the first `Verified_Answer` with a `Source_List` entry under the notes folder, so the identification cannot disagree with what happened
    - Render the `Evaluation_Report` markdown holding per question the answer, citations, scores, and rating, plus the mean to two decimals and the stamp as a table
    - _Requirements: 5.4, 11.3, 11.4, 11.5, 11.7, 11.8, 11.9_

  - [ ] 15.3 Implement `scripts/10_evaluate.py`
    - `generate` and `score` subcommands as a thin `main(argv) -> int` mirroring Week 2's `08_relevance_review.py`, mapping each typed exception to its status from the shared table (18 through 22 plus every status `09_ask.py` can return), and routing every string through the `Reporter`
    - _Requirements: 9.6, 11.1, 11.3, 11.4, 11.5, 11.6, 11.8, 12.8, 12.9_

  - [ ]* 15.4 Write evaluation example tests in `tests/test_answer_eval.py`
    - The 9/10 question-count boundary; every rejected rating form; `NoRatingsError`; the unrated question shown as `—` and excluded from the mean; `RefusalPathMissingError` with the run file left in place; the mismatched-identifier artifact error; the stamp holding the redaction marker with a key set and `unset` without one, and recording `ASKMYDOCS_PROVIDER` and `ASKMYDOCS_CHAT_PROVIDER` side by side; the derived grounding-question identifiers
    - _Requirements: 5.4, 11.5, 11.7, 11.8, 11.9_

  - [ ]* 15.5 Write property test for generate-mode coverage in `tests/test_answer_eval.py`
    - **Property 12: Generate mode covers every distinct non-empty question, in file order**
    - At least 100 examples over question-set texts of 0–20 lines with injected duplicates, blank lines, and whitespace-only lines; asserts one result per distinct non-empty line in first-appearance order with an empty rating, and the count error naming the count and the minimum exactly when fewer than 10 remain; the 9/10 boundary is seeded with explicit `@example` cases
    - **Validates: Requirements 11.1, 11.2, 11.6**
    - _Requirements: 11.1, 11.2, 11.6_

  - [ ]* 15.6 Write property test for the mean quality rating in `tests/test_answer_eval.py`
    - **Property 14: `Mean_Quality_Rating` is the mean of the non-empty ratings, to two decimal places**
    - At least 100 examples; valid ratings interleaved with empty entries for the mean half, and a bad-form strategy for the rejection half; asserts the mean equals `round(fsum(valid) / len(valid), 2)` so the test does not re-implement the rounding rule differently from the code, that it lies in 1.00–5.00, that empty entries are ignored, and that every bad form raises an error naming both the question and the rejected value
    - **Validates: Requirements 11.3, 11.4, 11.9**
    - _Requirements: 11.3, 11.4, 11.9_

  - [ ]* 15.7 Write Evaluation_Script tests in `tests/test_scripts_week3.py`
    - One case per evaluation row of the exit-status table — 18, 19, 20, 21, 22 — asserting the exact status, plus an assertion that the whole Week 3 status set is pairwise distinct per condition
    - _Requirements: 11.4, 11.5, 11.6, 11.8, 12.9_

- [ ] 16. Checkpoint — one path, two callers, one report
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Fri–Sat — Layering and frozen-module verification

  - [ ] 17.1 Write `tests/test_layering_week3.py`
    - **Not optional.** Requirements 2.8 and 16.5 make the import-graph shape a deliverable, not a confidence measure
    - AST-walk every module under `generation/` and fail on any `import` or `from` naming `askmydocs.stores` or `chromadb`, counting the `importlib.import_module` form too; assert `openai` is imported by `chat_client.py` and by no other Week 3 module; assert `prompting.py`, `citations.py`, and `presenter.py` import nothing from `generation/chat_client.py`, so the three pure modules stay runnable with no client; assert no module under `src/` calls `sys.exit`
    - Report every offending module and symbol, not just the first
    - _Requirements: 2.8, 12.8, 16.5_

  - [ ] 17.2 Write `tests/test_frozen_modules_week3.py`
    - **Not optional.** Requirements 4.8 and 16.5 make the frozen set a deliverable
    - Extend the Week 2 baseline with the Week 2 retrieval and store modules and content-check the whole frozen set: `chunking.py`, `similarity.py`, `models.py`, `loading/*`, everything under `embeddings/` including `retry.py`, and the Week 2 frozen modules; fail naming every differing path
    - State in a comment that `config.py` is deliberately absent because Week 3 extends it additively, and that `stores/factory.py` is absent because Week 3 changes it not at all
    - _Requirements: 4.8, 16.5_

- [ ] 18. Fri–Sat — The README

  - [ ] 18.1 Write the Week 3 setup and dataset sections of `README.md`
    - One paragraph on Ask My Docs and the committed `sample-notes/` dataset; the editable install, every environment variable with its default including the six Week 3 variables, the ingest command, and the `Ask_Script` command with and without `--top-k`
    - State plainly that **`OPENAI_API_KEY` is now required by a default install**, because the `Chat_Provider` defaults to `openai` while the `Embedding_Provider` still defaults to the local model — "it worked last week" is exactly the expectation this change breaks; state that every combination of the two providers is permitted
    - Instructions for adding the reader's own documents to `sample-notes/` and re-running ingest; the chosen `Chunk_Size` and `Top_K` with the reasoning for each, referencing the Week 2 Top-K report; that `Context_Budget` bounds prompt **size in code points**, not token count, and the implied minimum of roughly `800 + Chunk_Size + len(question)`
    - Hold no API key value and no substring of a key 8 characters or longer
    - _Requirements: 1.8, 13.1, 13.2, 13.3, 13.5, 13.7_

  - [ ] 18.2 Write the What_I_Learned_Section of `README.md`
    - In the learner's own words: what an embedding is, why `Chunk_Size` and `Chunk_Overlap` matter, and how the retrieval-augmented generation pipeline works end to end from a note file through chunking, embedding, storage, retrieval, the thresholded outcome, prompt construction, generation, citation validation, and the printed `Source_List`
    - _Requirements: 13.4_

  - [ ] 18.3 Render the exit-status table in `README.md`
    - Every status the `Ask_Script` and the `Evaluation_Script` return, each paired with the condition that produces it, verbatim in content from the design's table and in the same order: 0 through 5, the Week 2 range 6–11, and 12 through 22
    - Note that the design table is the single source of truth — a new condition is numbered there first and the README follows
    - _Requirements: 12.9, 13.8_

  - [ ] 18.4 Wire the `Example_Screenshot` reference into `README.md`
    - Commit the `docs/example-answer.png` path and reference it from the README alongside the example question and the expected presented answer with its cited `Source_List`; `docs/` sits outside `reports/` so the `.gitignore` glob for generated evaluation artifacts cannot exclude it
    - Scope is the reference and the committed path only — **capturing the image is a manual step** the learner performs against the running tool
    - _Requirements: 13.6, 16.1_

  - [ ] 18.5 Write the exit-status documentation test in `tests/test_docs_week3.py`
    - **Not optional.** Requirements 12.9 and 13.8 put the obligation on the README, so the check that the two cannot drift is a deliverable
    - Assert every status in the shared script exit-status mapping appears in the README's table paired with its condition, and that the README lists no status the mapping does not contain
    - _Requirements: 12.9, 13.8_

  - [ ]* 18.6 Extend the README presence checks in `tests/test_docs_week3.py`
    - Assert the install, ingest, and ask command strings are present; that all six Week 3 variables appear with their defaults; that the `OPENAI_API_KEY` requirement is stated; that the What_I_Learned_Section, the `Chunk_Size` and `Top_K` reasoning, and the `docs/example-answer.png` reference are present; that no 8-character-or-longer key-shaped substring appears
    - _Requirements: 1.8, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

- [ ] 19. Fri–Sat — Written learning artifacts and the demo shot list

  - [ ] 19.1 Write `learning-notes/rag-vs-fine-tuning.md`
    - Compare retrieval-augmented generation against fine-tuning on at least data freshness, cost, traceability of answers, and the kind of task each suits; state that this project uses retrieval-augmented generation and why, tying traceability to the `Citation_Table` and the printed `Source_List`
    - _Requirements: 14.1, 14.4_

  - [ ] 19.2 Write `learning-notes/demo-script.md`
    - An ordered shot list for a 3 to 4 minute demo covering the ingest step and 2 to 3 live questions, with an explanation of each pipeline stage; name for every shot the command shown and the point the learner makes, including one shot on the refusal path and one on the `Source_List`
    - The shot list **is** the deliverable — recording the video and sending the mentor the links happen outside the repository and are not tasks here; they are items on the `Submission_Checklist`
    - _Requirements: 14.2, 14.3, 14.4_

  - [ ] 19.3 Add the Week 3 explanatory comments to the generation modules
    - In `generation/prompting.py`: the prompt construction rules, instruction by instruction, and the context budget reduction rule with the reason prefix trimming keeps `Citation_Number` stable
    - In `generation/citations.py`: the citation validation rule, why de-duplication precedes the table lookup, and why the marker pattern uses `[0-9]` rather than `\d`
    - Additive comments only; no frozen module is touched
    - _Requirements: 16.4_

- [ ] 20. Sun — Submission and self-check checklists

  - [ ] 20.1 Write `learning-notes/submission-checklist.md`
    - One checkable item each for a public GitHub repository named `ask-my-docs`, the committed `sample-notes/` folder, the recorded 3 to 4 minute demo video, and the repository link plus video link sent to the mentor; record the deadline as Sunday 4 October 2026 23:59
    - Add the pre-commit staged-file check for `.env`, personal note files, `.chroma/`, `logs/`, the answer log, and the relevance review CSVs
    - _Requirements: 15.1, 15.3, 15.4, 16.2_

  - [ ] 20.2 Write `learning-notes/self-check.md`
    - One checkable item each for the six self-check questions: explaining an embedding without using the word "vector", why `Chunk_Size` affects the quality of retrieved context, how a vector database finds similar chunks, the full pipeline in under a minute, showing answers that cite their source, and showing graceful handling of a question with no good answer in the notes
    - _Requirements: 15.2, 15.4_

  - [ ]* 20.3 Extend `tests/test_docs_week3.py` with the learning-artifact checks
    - Assert each of the four learning-notes files exists with its required headings; assert the RAG comparison names all four comparison axes and states the chosen approach; assert the demo script names a command and a point per shot; assert the submission checklist records the deadline string and all four items, and the self-check lists all six questions; structure only, since correctness of the explanations is not machine-checkable
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 15.2, 15.3, 15.4_

- [ ] 21. Final checkpoint — whole suite green, offline, and within budget
  - Ensure all tests pass, ask the user if questions arise.
  - Run the whole suite with no API key present and confirm no network request is issued, that no test constructs an `OpenAIChatClient`, and that Week 1 plus Week 2 plus Week 3 stays under the 300-second ceiling with Week 3 adding under 60 seconds.
  - _Requirements: 4.8, 12.7, 16.5_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Every implementation task and every written repository artifact is required; most test sub-tasks are optional. Three are **not**: task 17.1 (layering), task 17.2 (frozen modules), and task 18.5 (the exit-status documentation check), because Requirements 12.9, 13.8, and 16.5 make them deliverables rather than confidence measures.
- `config.py` changes additively only, in tasks 2.1 and 2.2: one new dataclass, one new defaulted field appended last, one new validation phase appended after the Week 2 phase. Every Week 1 and Week 2 setting name, value, and validation stays unchanged.
- `stores/factory.py` needs **no change**. Week 3 adds no store; the requirement permits an edit and no task makes one.
- Every frozen Week 1 and Week 2 module stays byte-identical. `embeddings/retry.py` is imported for `RetryPolicy`, `classify_failure`, and `FailureKind` and gains nothing; task 17.2 content-checks the whole set.
- The default install now needs `OPENAI_API_KEY`, because the `Chat_Provider` defaults to `openai` while the `Embedding_Provider` still defaults to the local model. The absence is not a configuration error — it is caught at wiring time by `build_chat_client` before any request is issued — and `.env.example` (1.2) and the README (18.1) both say so.
- Task 4.4 pins `len(SYSTEM_PROMPT) == 766` and the literal's SHA-256 on purpose. Editing the prompt, even by one space, is a behaviour change: every existing `Prompt_Hash` stops matching and evaluation runs before and after the edit stop being comparable. A deliberate edit must update the pin and regenerate the report.
- Task 5.2's `FakeChatModel` is the reason no test needs a network. Its scripted library covers a well-cited answer, a cited subset, a dangling marker, the `[0]` case, mixed valid-and-dangling, **no markers at all**, the model's own "I don't know" fallback, the near-miss forms, and **malformed output** in both its `None` and blank-content forms.
- Each of the design's fourteen properties has exactly one property-based test. Conjuncts live as assertions inside that one test, so the design-property-to-test mapping stays one to one. All run at least 100 examples; Property 1 runs 200.
- Requirements 5.1 and 5.2 are covered by integration tests rather than properties: with a scripted model a "the answer cites the notes" property would only assert that the fake cited what the test told it to. The negative half needs no model at all and is fully deterministic.
- Recording the demo video and sending the mentor the repository and video links are not tasks — they happen outside the repository. The shot list (19.2) and the checklist items (20.1) are the deliverables. Capturing `docs/example-answer.png` is likewise manual; task 18.4 wires the reference and commits the path.
- Checkpoints sit after generation is verified against fakes, after citations and the presenter, after the pipeline and the evaluation run, and at the end.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "2.1", "3.1", "7.1", "7.2"] },
    { "id": 1, "tasks": ["1.4", "2.2", "4.1"] },
    { "id": 2, "tasks": ["2.3", "3.2", "4.2"] },
    { "id": 3, "tasks": ["4.3", "5.1"] },
    { "id": 4, "tasks": ["4.4", "5.2"] },
    { "id": 5, "tasks": ["4.5", "6.1"] },
    { "id": 6, "tasks": ["4.6", "6.2", "9.1"] },
    { "id": 7, "tasks": ["4.7", "6.3", "9.2", "10.1"] },
    { "id": 8, "tasks": ["4.8", "9.3", "10.2", "12.1"] },
    { "id": 9, "tasks": ["9.4", "12.2", "13.1"] },
    { "id": 10, "tasks": ["9.5", "12.3", "13.2", "14.1"] },
    { "id": 11, "tasks": ["12.4", "13.3", "14.2", "15.1"] },
    { "id": 12, "tasks": ["13.4", "15.2", "17.1", "17.2"] },
    { "id": 13, "tasks": ["15.3", "15.4", "19.1", "19.2"] },
    { "id": 14, "tasks": ["15.5", "15.7", "18.1", "20.1", "20.2"] },
    { "id": 15, "tasks": ["15.6", "18.2", "19.3"] },
    { "id": 16, "tasks": ["18.3"] },
    { "id": 17, "tasks": ["18.4"] },
    { "id": 18, "tasks": ["18.5"] },
    { "id": 19, "tasks": ["18.6"] },
    { "id": 20, "tasks": ["20.3"] }
  ]
}
```
