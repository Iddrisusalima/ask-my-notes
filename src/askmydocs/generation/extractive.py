"""Extractive answering: compose the answer from the retrieved text itself.

Abstractive generation - handing the context to a chat model and asking it to
paraphrase - needs a paid API or a local model too large to be practical here.
Extraction needs neither, and it buys a property abstraction cannot offer:
the answer is literally sentences from the notes, so it cannot hallucinate.
Every sentence carries the citation marker of the chunk it came from, so the
citation table is correct by construction rather than by the model's goodwill.

The ranking reuses the project's own machinery. Chunks were ranked against the
question by cosine similarity over embeddings; here the same comparison runs one
level finer, over the sentences inside the supplied chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from askmydocs.generation.generator import GeneratedAnswer, TokenUsage
from askmydocs.similarity import cosine_similarity

#: Split on sentence-ending punctuation followed by whitespace. Deliberately
#: simple: a heavier splitter would be another dependency, and note prose is
#: well behaved.
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

#: Markdown heading lines, dropped before sentence splitting.
HEADING_PATTERN = re.compile(r"^#{1,6}\s")

#: Fragments shorter than this are list stubs or truncation debris, not answers.
MIN_SENTENCE_CHARS = 40

DEFAULT_MAX_SENTENCES = 3


@dataclass(frozen=True)
class SentenceCandidate:
    """One sentence from one supplied chunk, with its citation number."""

    text: str
    citation_number: int
    score: float


def split_sentences(text: str) -> list[str]:
    """Sentences worth considering as an answer.

    Markdown heading lines are dropped first. A heading carries no terminating
    punctuation, so leaving it in would glue it to the sentence that follows and
    the answer would open with "# Chunking strategies Splitting matters...".
    """
    prose_lines = [
        line
        for line in text.splitlines()
        if not HEADING_PATTERN.match(line.strip())
    ]
    flattened = " ".join(" ".join(prose_lines).split())
    parts = SENTENCE_PATTERN.split(flattened)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_CHARS]


def rank_sentences(question: str, prompt, result, embedder) -> list[SentenceCandidate]:
    """Score every sentence of the supplied chunks against the question.

    One embedder call for the question, one batched call for the sentences, so
    the cost is two requests regardless of how many chunks were supplied.
    """
    numbered: list[tuple[int, str]] = []
    for entry, scored in zip(prompt.citations, result.hits):
        for sentence in split_sentences(scored.chunk.text):
            numbered.append((entry.number, sentence))

    if not numbered:
        return []

    question_vector = embedder.embed_text(question)
    sentence_vectors = embedder.embed_texts([s for _, s in numbered])

    candidates = [
        SentenceCandidate(
            text=sentence,
            citation_number=number,
            score=cosine_similarity(question_vector, vector),
        )
        for (number, sentence), vector in zip(numbered, sentence_vectors)
    ]
    candidates.sort(key=lambda c: -c.score)
    return candidates


def build_extractive_answer(
    question: str,
    prompt,
    result,
    embedder,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> GeneratedAnswer:
    """Compose an answer from the highest-scoring retrieved sentences.

    Returns the same ``GeneratedAnswer`` shape the chat path returns, so the
    citation validator, the source list, and the presenter are untouched.
    """
    candidates = rank_sentences(question, prompt, result, embedder)
    if not candidates:
        return GeneratedAnswer(
            text="I don't know based on the provided notes.",
            usage=TokenUsage(),
            model="extractive",
        )

    chosen: list[SentenceCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        chosen.append(candidate)
        if len(chosen) >= max_sentences:
            break

    sentences = [
        f"{c.text.rstrip('.')} [{c.citation_number}]." for c in chosen
    ]
    return GeneratedAnswer(
        text=" ".join(sentences),
        usage=TokenUsage(),
        model="extractive",
    )
