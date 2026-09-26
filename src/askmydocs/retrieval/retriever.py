"""Retrieval: embed the question, rank, threshold, log.

The threshold decision produces one of three outcomes, as a closed enum rather
than a boolean. Week 3 keys its refusal logic off this, and a boolean could not
distinguish "the notes hold nothing relevant" from "there is nothing indexed at
all" -- two situations that need different advice to the user.

The boundary is inclusive: a top score exactly equal to the threshold counts as
relevant. An exclusive boundary would make the default 0.30 mean "strictly
above 0.30", which is not what the number looks like it means.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from askmydocs.errors import AskMyDocsError
from askmydocs.models import SearchHit


class RetrievalOutcome(Enum):
    """What the retriever concluded."""

    RELEVANT_CONTEXT = "relevant-context"
    NO_RELEVANT_CONTEXT = "no-relevant-context"
    EMPTY_COLLECTION = "empty-collection"


class EmptyQuestionError(AskMyDocsError):
    """The question held no non-whitespace character."""


class QuestionTooLongError(AskMyDocsError):
    """The question exceeded the configured maximum input length."""


@dataclass(frozen=True)
class ScoredHit:
    """One retrieved chunk, flagged against the threshold."""

    hit: SearchHit
    below_threshold: bool

    @property
    def chunk(self):
        return self.hit.chunk

    @property
    def score(self) -> float:
        return self.hit.score


@dataclass(frozen=True)
class RetrievalResult:
    """Everything one question produced."""

    question: str
    hits: tuple[ScoredHit, ...]
    outcome: RetrievalOutcome
    top_score: float | None
    threshold: float
    top_k: int
    collection_count: int

    @property
    def should_answer(self) -> bool:
        """True only when there is context worth sending to a model."""
        return self.outcome is RetrievalOutcome.RELEVANT_CONTEXT


class Retriever:
    """Turns a question into a thresholded, ordered result."""

    def __init__(self, store, embedder, threshold: float, max_input_length: int) -> None:
        self._store = store
        self._embedder = embedder
        self._threshold = threshold
        self._max_input_length = max_input_length

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        """Embed, query, threshold.

        Raises:
            EmptyQuestionError: the question is blank. No embedder call is made,
                so a blank question costs nothing.
            QuestionTooLongError: the question exceeds the input cap.
        """
        if not question or not question.strip():
            raise EmptyQuestionError(
                "The question is empty or contains only whitespace."
            )
        if len(question) > self._max_input_length:
            raise QuestionTooLongError(
                f"The question is {len(question)} code points, above the configured "
                f"maximum of {self._max_input_length}."
            )

        count = self._store.count()
        if count == 0:
            return RetrievalResult(
                question=question,
                hits=(),
                outcome=RetrievalOutcome.EMPTY_COLLECTION,
                top_score=None,
                threshold=self._threshold,
                top_k=top_k,
                collection_count=0,
            )

        vector = self._embedder.embed_text(question)
        raw = self._store.query(vector, top_k)
        top_score = raw[0].score if raw else None

        # Sub-threshold hits are returned, not suppressed, each marked. Hiding
        # them would leave a user unable to see how close the near-misses were.
        hits = tuple(
            ScoredHit(hit=hit, below_threshold=hit.score < self._threshold)
            for hit in raw
        )
        relevant = top_score is not None and top_score >= self._threshold
        outcome = (
            RetrievalOutcome.RELEVANT_CONTEXT
            if relevant
            else RetrievalOutcome.NO_RELEVANT_CONTEXT
        )

        return RetrievalResult(
            question=question,
            hits=hits,
            outcome=outcome,
            top_score=top_score,
            threshold=self._threshold,
            top_k=top_k,
            collection_count=count,
        )
