# Retrieval quality reference

Numbers and defaults the skills refer to. These are the Ask My Docs defaults; adjust for your own corpus.

## Defaults

| Setting | Default | Notes |
|---|---|---|
| Chunk size | 500 characters | Unicode code points, not bytes |
| Chunk overlap | 50 characters | Must be strictly less than chunk size |
| Top K | 5 | Try 3, 5, 10 when tuning |
| Relevance threshold | 0.30 | Cosine similarity; boundary is inclusive |
| Distance metric | cosine | Set explicitly at collection creation, immutable afterwards |
| Exact search limit | 1000 stored chunks | Above this, HNSW is approximate and a recall floor applies |

## Cost of each lever

| Lever | Re-ingest needed | Re-embeds | Invalidates past runs |
|---|---|---|---|
| Top K | no | no | no |
| Relevance threshold | no | no | no |
| Chunk size or overlap | yes | every file | yes, chunk ids change |
| System prompt | no | no | yes, every prompt hash changes |

## Reading a cosine score

A score near 1.0 means the question and the chunk point in nearly the same direction in embedding
space. Near 0 means unrelated. Negative means opposed, which is rare for natural language and usually
signals a degenerate vector rather than genuine opposition. The metric ignores vector magnitude, so a
long chunk is not favoured over a short one by length alone.

## Precision at K

For one question, the count of retrieved chunks a human labelled relevant, divided by the number
retrieved. Label every row before computing it: an unlabelled row is not a negative, it is missing
data, and silently treating it as a negative understates precision.
