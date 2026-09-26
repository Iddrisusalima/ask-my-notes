# ask-my-docs-rag

A Kiro power for building and debugging a hand-written Retrieval-Augmented Generation pipeline over
your own PDF and markdown notes.

## What it bundles

**Skills**

- `tune-retrieval` - diagnose poor answers from the retrieval log, then change Top_K, the relevance
  threshold, chunk size, and the system prompt in cost order.
- `verify-grounding` - prove an answer came from retrieved notes rather than model recall, using a
  positive and a negative grounding question.
- `explain-pipeline` - the end-to-end explanation plus the six self-check answers, including what an
  embedding is without leaning on the word "vector".

**MCP server**

- `fetch` - retrieves web pages, for reading vector database and embedding provider documentation
  while you work. Requires `uv` on the PATH.

**References**

- `references/retrieval-reference.md` - default chunk size, overlap, Top_K, threshold, the cost of
  each tuning lever, and how to read a cosine score.

## Installing

Powers are installed by the Kiro client. Point it at this directory, or at the public repository that
contains it.

## A note on the local copies

While this power is not installed, the same three skills also live in `.kiro/skills/` in the Ask My
Docs workspace so they load without installation. Once the power is installed, delete those copies -
two copies of the same skill will drift apart.
