---
name: tune-retrieval
description: Diagnose and fix poor RAG answers by reading the retrieval log first, then changing Top_K, the relevance threshold, chunk size, and the system prompt in cost order. Use when answers are wrong, vague, or cite the wrong chunk.
---

# Tune retrieval when answers are poor

Use this when an answer is wrong, vague, or cites the wrong chunk. Change one variable at a time and
record the configuration with each run, otherwise the runs are not comparable.

## Diagnose before you tune

Read the retrieval before you blame the model. `logs/retrievals.jsonl` holds the question, the
returned chunk ids, the source paths, and the `Similarity_Score` of every hit.

| Symptom in the retrieval log | Likely cause | What to change |
|---|---|---|
| Top score is below the `Relevance_Threshold` | the notes genuinely lack the answer, or the question is phrased unlike the notes | confirm the refusal is correct; only then reword the question |
| Retrieved chunks are on-topic but cut mid-thought | `Chunk_Size` too small, or `Chunk_Overlap` too small | raise `Chunk_Overlap` first, then `Chunk_Size`; re-ingest |
| Retrieved chunks each mix several unrelated topics | `Chunk_Size` too large, so one chunk dilutes the embedding | lower `Chunk_Size`; re-ingest |
| Right chunk present but ranked below noise | `Top_K` too small to include it | raise `Top_K` |
| Many retrieved chunks, most irrelevant | `Top_K` too large, so noise crowds the context | lower `Top_K` |
| Retrieval is correct, answer ignores it | prompt problem, not retrieval | revise the `System_Prompt` |

## Order of operations

1. **`Top_K` first.** It is free to change and needs no re-ingest. Try 3, 5, 10 and compare
   precision@K from the relevance review.
2. **`Relevance_Threshold` second.** Also free. Raise it if noise is being answered from; lower it if
   good answers are being refused. Check the boundary is doing what you think: a score exactly equal
   to the threshold counts as relevant.
3. **`Chunk_Size` and `Chunk_Overlap` third.** These require a full re-ingest, which re-embeds every
   changed file and may cost money. Change them only after the two free levers are exhausted.
4. **The `System_Prompt` last.** Editing it changes every `Prompt_Hash`, so evaluation runs before and
   after the edit are no longer comparable. Note the edit in the evaluation report.

## After any change

Re-run the evaluation over the same question set, keep the `Configuration_Stamp`, and compare the
mean quality rating against the previous run. A change you cannot measure is a change you cannot
defend in the demo.

## What not to do

Do not add a re-ranker, a query rewriter, or a framework. The project is graded on understanding the
pipeline you built, and every one of those hides a stage you are meant to be able to explain.

