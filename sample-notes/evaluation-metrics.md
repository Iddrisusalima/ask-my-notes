# Evaluating retrieval quality

You cannot tune what you do not measure, and "the answers feel better" is not a
measurement.

## Precision at K

For one question, the count of retrieved chunks a human judged relevant divided
by the number retrieved. Labelling is manual and tedious and there is no way
around it: relevance is a judgement about intent, not a property of the text.

An unlabelled row is missing data, not a negative. Treating blanks as
irrelevant understates precision and makes every comparison meaningless.

## Choosing K

I ran the same ten questions at K equal to 3, 5, and 10. At 3 the right chunk
was sometimes just outside the window. At 10 precision dropped sharply because
the tail was filler that crowded the context budget. At 5 the mean precision was
0.72, the best of the three, so 5 became my default.

## The refinement loop

Change one variable, re-run the same question set, compare. Stamp every run with
the full configuration, otherwise two runs are not comparable and the comparison
is theatre. Top K and the threshold are free to change. Chunk size forces a full
re-ingest. The prompt invalidates every earlier run.
