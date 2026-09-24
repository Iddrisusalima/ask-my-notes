"""Property tests for the in-memory store.

Feature: week1-embeddings-chunking, Properties 14-19.
Each test states the acceptance criteria it validates.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from askmydocs.errors import BatchLengthMismatchError, VectorLengthError
from askmydocs.similarity import cosine_similarity
from askmydocs.stores.memory import InMemoryStore
from tests.strategies import make_chunk, vector

DIM = 4


@st.composite
def batches(draw, max_batches: int = 5):
    """A list of batch sizes, zero-sized batches allowed (Requirement 10.6)."""
    return draw(st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=max_batches))


def fill(store: InMemoryStore, sizes, dim: int = DIM) -> int:
    """Add the given batch sizes, returning the total added."""
    total = 0
    for size in sizes:
        chunks = [make_chunk(total + i, f"chunk-{total + i}") for i in range(size)]
        vectors = [[float((total + i) % 7 + 1)] + [0.0] * (dim - 1) for i in range(size)]
        store.add(chunks, vectors)
        total += size
    return total


# Property 14: Stored item count is additive over add operations.
# Validates: Requirements 10.6, 10.5
@given(batches())
def test_property_14_count_additivity(sizes) -> None:
    store = InMemoryStore()
    running = 0
    for size in sizes:
        chunks = [make_chunk(running + i, f"c{running + i}") for i in range(size)]
        vectors = [[1.0, 0.0, 0.0, 0.0] for _ in range(size)]
        store.add(chunks, vectors)
        running += size
        assert store.count() == running
    assert store.count() == sum(sizes)


# Property 15: Chunk-to-vector association is preserved.
# Validates: Requirements 10.4, 10.2
@given(st.integers(min_value=1, max_value=20))
def test_property_15_association_preserved(n) -> None:
    store = InMemoryStore()
    chunks = [make_chunk(i, f"text-{i}") for i in range(n)]
    vectors = [[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)]
    store.add(chunks, vectors)
    for i, record in enumerate(store._records):
        assert record.chunk.text == f"text-{i}"
        assert record.embedding == (float(i + 1), 0.0, 0.0, 0.0)
        assert record.insertion_index == i


# Property 16: Add validation is atomic.
# Validates: Requirements 10.7, 10.8
@given(batches(max_batches=3), st.sampled_from(["count", "length"]))
def test_property_16_add_is_atomic(sizes, violation) -> None:
    store = InMemoryStore()
    fill(store, sizes)
    before_count = store.count()
    before_records = list(store._records)

    if violation == "count":
        bad_chunks, bad_vectors = [make_chunk(999)], []
        expected = BatchLengthMismatchError
    else:
        bad_chunks = [make_chunk(999)]
        bad_vectors = [[1.0] * (DIM + 3)]
        expected = VectorLengthError

    if before_count == 0 and violation == "length":
        return  # no dimensionality established yet, so no mismatch is possible
    try:
        store.add(bad_chunks, bad_vectors)
        raise AssertionError("expected the invalid batch to be rejected")
    except expected:
        pass
    assert store.count() == before_count
    assert store._records == before_records


# Property 17: Query returns the top-K prefix in descending order, deterministically.
# Validates: Requirements 10.9, 10.10
@given(st.integers(min_value=0, max_value=30), st.integers(min_value=1, max_value=40), vector(DIM, DIM))
def test_property_17_topk_matches_reference_ranking(n, k, query) -> None:
    store = InMemoryStore()
    fill(store, [n])
    hits = store.query(query, k) if n else store.query(query, k)
    assert len(hits) == min(k, n)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Independent reference ranking over the same records.
    reference = sorted(
        store._records,
        key=lambda r: (-cosine_similarity(query, r.embedding), r.insertion_index),
    )[:k]
    assert [h.insertion_index for h in hits] == [r.insertion_index for r in reference]
    assert [h.insertion_index for h in store.query(query, k)] == [h.insertion_index for h in hits]


# Property 18: A stored vector retrieves its own chunk; ties by insertion index.
# Validates: Requirements 10.12, 10.15
@given(st.integers(min_value=2, max_value=5))
def test_property_18_self_retrieval_and_tie_order(copies) -> None:
    store = InMemoryStore()
    # Distinct vectors: querying with record i's vector must return chunk i.
    chunks = [make_chunk(i, f"d{i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    store.add(chunks, vectors)
    for i, v in enumerate(vectors):
        assert store.query(v, 1)[0].chunk.text == f"d{i}"

    # Ties: one vector repeated must come back in ascending insertion index.
    tied = InMemoryStore()
    tied.add(
        [make_chunk(i, f"t{i}") for i in range(copies)],
        [[1.0, 0.0, 0.0, 0.0] for _ in range(copies)],
    )
    hits = tied.query([1.0, 0.0, 0.0, 0.0], copies)
    assert [h.insertion_index for h in hits] == list(range(copies))


# Property 19: An empty store returns no results without raising.
# Validates: Requirements 10.11
@given(vector(DIM, DIM), st.integers(min_value=1, max_value=100))
def test_property_19_empty_store_returns_nothing(query, k) -> None:
    assert InMemoryStore().query(query, k) == []
