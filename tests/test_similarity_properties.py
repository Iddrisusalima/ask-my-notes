"""Property tests for cosine similarity.

Feature: week1-embeddings-chunking, Properties 1-4.
Each test states the acceptance criteria it validates.
"""

from __future__ import annotations

import math

from hypothesis import example, given
from hypothesis import strategies as st

from askmydocs.similarity import cosine_similarity
from tests.strategies import scale_factor, vector, vector_pair

TOLERANCE = 1e-9


# Property 1: Cosine similarity is bounded.
# Validates: Requirements 4.2
@given(vector_pair())
@example(([1.0, 0.0], [0.0, 1.0]))
@example(([1.0, 0.0], [-1.0, 0.0]))
@example(([1.0], [1.0]))
def test_property_1_similarity_is_bounded(pair) -> None:
    left, right = pair
    result = cosine_similarity(left, right)
    assert -1.0 - TOLERANCE <= result <= 1.0 + TOLERANCE
    assert math.isfinite(result)


# Property 2: Cosine similarity is symmetric.
# Validates: Requirements 4.3
@given(vector_pair())
def test_property_2_similarity_is_symmetric(pair) -> None:
    left, right = pair
    assert abs(cosine_similarity(left, right) - cosine_similarity(right, left)) <= TOLERANCE


# Property 3: Self-similarity is one.
# Validates: Requirements 4.4
@given(vector())
def test_property_3_self_similarity_is_one(v) -> None:
    assert abs(cosine_similarity(v, v) - 1.0) <= TOLERANCE


# Property 4: Positive scaling does not change similarity.
# Validates: Requirements 4.5
@given(vector(), scale_factor)
@example([1.0, 2.0, 3.0], 1e-6)
@example([1.0, 2.0, 3.0], 1.0)
@example([1.0, 2.0, 3.0], 1e6)
def test_property_4_positive_scaling_preserves_similarity(v, k) -> None:
    scaled = [x * k for x in v]
    # A repaired vector can still scale to a degenerate norm at extreme k;
    # rescale rather than skip, so the example still counts.
    if not any(abs(x) > 1e-9 for x in scaled):
        scaled = [x * 1e9 for x in scaled]
    assert abs(cosine_similarity(v, scaled) - 1.0) <= TOLERANCE
