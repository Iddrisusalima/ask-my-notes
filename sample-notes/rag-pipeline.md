# The RAG pipeline, end to end

Two phases. Ingest happens once per change to the notes. Answering happens per
question.

## Ingest

Discover the note files. Extract text per format: page by page for PDF, straight
read for markdown. Split each document into overlapping chunks. Embed every
chunk in batches. Store the chunk text, its metadata, and its vector under a
stable id built from the source path and the chunk ordinal.

Re-running must be cheap, so I keep a manifest recording a SHA-256 hash per
source file. Unchanged files are skipped entirely. Changed files have their old
chunks deleted before the new ones are written, otherwise a file that shrinks
leaves orphan chunks behind that still answer queries.

## Answering

Embed the question with the same model used for the chunks. Different models
produce incompatible spaces, so mixing them silently returns nonsense. Ask the
store for the nearest chunks. Compare the best score against a relevance
threshold. Below it, refuse rather than answer from noise.

Above it, assemble the surviving chunks into a numbered context block, send it
with an instruction to use only that context, then check that every citation the
model produced points at a chunk that was actually supplied.

## Why refusal is the interesting part

An assistant that always answers is useless, because you cannot tell the
grounded answers from the invented ones. The refusal path is what makes the
rest trustworthy.
