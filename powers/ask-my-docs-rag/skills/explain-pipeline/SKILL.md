---
name: explain-pipeline
description: Rehearse the end-to-end RAG explanation and the six project self-check answers, including embeddings without the word vector, chunk size effects, vector search, and RAG versus fine-tuning.
---

# Explain the pipeline end to end

The project is assessed on explanation as much as on working code. Six questions must be answerable
out loud, without notes. Use this as a rehearsal script.

## The pipeline in one minute

Ingest: discover the note files, extract their text, split each document into overlapping fixed-size
chunks, turn every chunk into an embedding, store the chunk text plus its embedding in a local vector
database, keyed by a stable chunk id.

Answer: embed the question with the same model, ask the database for the nearest stored chunks, check
the best score against a relevance threshold, put the surviving chunks into a numbered context block,
send that block plus the question to a chat model under an instruction to use only that context, then
validate that every citation the model produced points at a chunk that was actually supplied.

## The six self-check questions

**What is an embedding, without leaning on the word "vector"?** A list of numbers a model assigns to a
piece of text so that texts meaning similar things get similar lists. The numbers are not features
anyone chose; they are positions in a space the model learned, where distance stands in for difference
in meaning.

**Why does chunk size affect the quality of retrieved context?** One embedding has to represent one
chunk. A chunk covering several topics averages them, so it matches many questions weakly and none
strongly. A chunk too small loses the surrounding sentences that made it meaningful, so the retrieved
text is on-topic but unusable. Overlap exists so a fact straddling a boundary survives in one piece.

**How does a vector database find similar chunks?** It compares the question's embedding against the
stored ones under a distance metric - here cosine, which measures direction rather than magnitude.
Scanning every stored vector is exact but linear, so the database builds a proximity graph index and
walks it, trading a small amount of exactness for speed. At this corpus size the search is exact.

**Can you explain the full pipeline in under a minute?** Use the two paragraphs above.

**Does the tool cite where each answer came from?** Every sentence carries a bracketed marker, the
markers map to a numbered source list, and each source names the file, the chunk index, the character
range, and the similarity score. A marker that does not map to a supplied chunk marks the whole answer
unverified.

**Does it handle a question with no good answer?** Yes, in two distinct ways. If the best similarity
score falls below the threshold, the tool refuses before spending a token on generation. If the
collection is empty, it says so and names the ingest command. Those are different outcomes and are
reported differently.

## RAG versus fine-tuning, briefly

Retrieval keeps the knowledge outside the model, so adding a note is a re-ingest rather than a
retraining, and every answer can point at its source. Fine-tuning changes the model's behaviour and
suits teaching it a format or a style, but it cannot cite and it goes stale the moment the notes
change. This project needs fresh, traceable facts, so retrieval is the right tool.

