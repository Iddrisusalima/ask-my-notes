---
name: verify-grounding
description: Prove an answer came from retrieved notes rather than model recall, using a positive and a negative grounding question. Use when checking grounding, citation trust, or preparing the demo.
---

# Prove an answer is grounded, not recalled

A retrieval-augmented answer must come from the retrieved context. A model answering from its own
training data looks identical on the surface, so grounding has to be demonstrated rather than assumed.

## The two-question test

**Positive grounding question.** Pick a fact that appears in your notes and nowhere in general
knowledge — a date you chose, a name you invented, a number specific to your own work. Ask it. A
grounded pipeline answers correctly and cites the chunk holding that fact. If it answers correctly but
cites nothing, the answer is unverified, not grounded.

**Negative grounding question.** Ask something plainly absent from the notes but well known to any
model, for example the boiling point of water. A grounded pipeline refuses. If it answers, the model
is drawing on training data and the `System_Prompt` constraint is not holding.

Keep both questions in the evaluation question set so every run re-checks them.

## Reading the result

| Outcome | What it means |
|---|---|
| Verified answer, citation points at the right chunk | grounded; this is the target |
| Correct answer, no citation marker | unverified; the model may be recalling. Tighten the citation instruction |
| Correct answer, citation number not in the context | dangling citation; the validator must catch it and the answer must not be presented as verified |
| Refusal on the positive question | retrieval failure, not a grounding failure. Tune retrieval instead |
| Confident answer on the negative question | grounding failure. The prompt is being overridden |

## Why a dangling citation matters more than it looks

A fabricated citation number is the one failure mode that makes a wrong answer *look* verifiable. A
reader who trusts the source list stops checking. That is why validation compares every marker in the
answer against the citation table and labels the whole answer unverified when any marker has no match,
rather than silently dropping the bad marker.

## In the demo

Show both questions live, in that order. The refusal is the more persuasive of the two: it is the
moment that proves the tool is reading your notes rather than reciting the internet.

