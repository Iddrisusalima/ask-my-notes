# Ask My Docs

Ask My Docs is a from-scratch Retrieval-Augmented Generation tool over the learner's own PDF and
markdown notes. It chunks documents, embeds the chunks, stores them in a local vector database,
retrieves the most relevant chunks for a question, and answers using only that retrieved context,
with citations back to the source files.

## Why it is built this way

This is a learning project, Project 2 of a mentored program. Every RAG stage is hand-written so the
learner can explain it: what an embedding is, why chunk size and overlap matter, how similarity
ranking works, and where an answer's context came from. Reaching for a framework that hides a stage
defeats the purpose of the project.

## Three-week plan, one spec per week

- **Week 1 — `week1-embeddings-chunking`**: embeddings, document loading, chunking, and an
  in-memory store behind a store interface.
- **Week 2 — `week2-vector-db-retrieval`**: a persistent Chroma store, incremental ingest,
  retrieval with a relevance threshold, top-K tuning, and retrieval logging.
- **Week 3 — `week3-generation-citations`**: prompt construction, answer generation, citations,
  evaluation, README, demo, and submission.

## Deadline and submission

Sunday 4 October 2026, 23:59. Submission is a public GitHub repository named `ask-my-docs` plus a
link to a 3-4 minute demo video, sent to the mentor.

## Definition of done

- The tool answers questions from the learner's notes.
- Every answer cites its sources.
- The tool refuses gracefully when the notes hold no answer, rather than inventing one.
- The README explains embeddings, chunking, and the end-to-end pipeline in the learner's own words.
