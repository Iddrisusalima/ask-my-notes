# RAG versus fine-tuning

Both change what a model can do for you. They are not alternatives so much as
answers to different questions.

## Retrieval

Knowledge stays outside the model, in a store you control. Adding a document is
an ingest, not a training run. Every answer can point at its source, so a reader
can verify it. Facts are as fresh as the last ingest.

Cost is per query: you pay to embed the question and to generate over the
retrieved context.

## Fine-tuning

Behaviour is baked into the weights. It is the right tool for teaching a model a
format, a tone, or a task shape it does not already know. It cannot cite, because
the knowledge is diffused through the parameters with no index back to a source.
It goes stale the moment the underlying facts change, and refreshing means
retraining.

Cost is up front and large, then cheap per query.

## The decision for this project

I need answers about notes that change weekly, and I need to show where each
answer came from. That is retrieval, unambiguously. Fine-tuning would have been
the wrong tool even if it were free.
