# Vector databases

A vector database stores embeddings alongside the text and metadata they came
from, and answers the question "which stored vectors are nearest this one".

## How the search actually works

Comparing the query against every stored vector is exact and linear. That is
fine for a few thousand chunks and hopeless for a few million. So the database
builds an index. Chroma uses HNSW, a layered proximity graph: the upper layers
are sparse and let the search jump across the space quickly, the lower layers
are dense and refine the result. The search walks down the layers, keeping a
small candidate set at each level.

The trade is exactness for speed. HNSW is approximate: it can miss a true
nearest neighbour. Below roughly a thousand chunks the distinction stops
mattering because the search effectively scans everything.

## Chroma versus Pinecone

Chroma runs in-process, writes to a local directory, needs no account, and costs
nothing. Pinecone is hosted, scales far past what one machine holds, and charges
for it. For a notes corpus of a few hundred chunks, the hosted option buys
nothing and adds a network hop, an API key, and a bill. I chose Chroma.

## Distance metric

I configure cosine distance explicitly rather than relying on the default, which
is L2. Cosine ignores vector magnitude and compares direction only, which is
what I want: a long chunk should not outrank a short one on length alone.
