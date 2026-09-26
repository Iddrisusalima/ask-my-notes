# Embeddings - working notes

An embedding is the list of numbers a model assigns to a piece of text so that
texts meaning similar things end up with similar lists. Nobody chose what the
numbers mean. They are coordinates in a space the model learned during training,
where distance stands in for difference in meaning.

The local model I settled on, all-MiniLM-L6-v2, returns 384 numbers per text.
The hosted alternative, text-embedding-3-small, returns 1536. More numbers is
not automatically better: the smaller model was faster on my laptop and good
enough for notes retrieval, and it needs no API key.

## What surprised me

Two sentences can share almost no words and still score above 0.7, because the
model is matching meaning rather than vocabulary. The reverse also happens: two
sentences sharing most of their words can score low when one negates the other.

## My benchmark run, 24 September

I compared three sentences about morning coffee against three about tax filing
deadlines. Within the coffee group the mean cosine similarity was 0.6412. Across
the two groups it was 0.1187. That gap is the whole basis of semantic search.
