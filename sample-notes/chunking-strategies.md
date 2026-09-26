# Chunking strategies

Splitting matters for two independent reasons. A model has a context limit, so a
long document cannot be passed whole. And retrieval precision degrades when a
chunk covers several topics, because one embedding has to represent all of them
and ends up representing none of them strongly.

## Fixed size versus semantic

Fixed size splits every N characters. It is crude, trivially reversible, and
completely predictable, which is why I started there. Semantic chunking splits on
meaning boundaries such as headings or paragraphs. It produces better chunks and
much worse guarantees: you can no longer say what the boundaries will be.

## Overlap

Overlap exists so a fact straddling a boundary survives intact in at least one
chunk. Without it, a sentence split across two chunks is retrievable from
neither, because half a sentence embeds to something close to nonsense.

## Settings I landed on

After the experiment I settled on a chunk size of 500 characters with 50
characters of overlap, a 10 percent overlap ratio. At size 200 the chunks were
too fragmentary to answer anything. At 1000 a single chunk pulled in two
unrelated headings and the retrieval got noisier.
