"""Cosine similarity, written by hand.

This is the numeric core the whole project rests on: the store ranks by it, the
relevance threshold is denominated in it, and Week 2 checks the vector
database's own distances against it.

Validation order is fixed and tested, because the requirements pin it
(Requirement 4.6 puts the length check first):

1. lengths differ            -> DimensionMismatchError  (Req 4.6)
2. vectors are empty         -> EmptyVectorError        (Req 4.8)
3. a non-finite element      -> NonFiniteValueError     (Req 4.9)
4. a degenerate norm         -> ZeroNormError           (Req 4.7)

Steps 1 and 2 are in that order deliberately: two empty vectors have *equal*
length, so they fall past the length check into the emptiness check, while one
empty and one non-empty is a length mismatch. Step 3 precedes step 4 because a
NaN element would otherwise produce a NaN norm and a misleading error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from askmydocs.errors import (
    DimensionMismatchError,
    EmptyVectorError,
    NonFiniteValueError,
    ZeroNormError,
)

#: Norms at or below this are treated as degenerate. Requirements 4.1, 4.7.
#: A plain ``== 0`` test is not enough: float64 underflow can produce a norm
#: that is positive but so small the division is meaningless.
ZERO_NORM_THRESHOLD = 1e-12


def euclidean_norm(vector: Sequence[float]) -> float:
    """Euclidean norm over float64."""
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two comparable vectors.

    Returns the dot product over the product of the norms, clamped into
    [-1.0, 1.0] to absorb float64 rounding at the extremes (Requirement 4.2).

    Raises:
        DimensionMismatchError: lengths differ (Req 4.6).
        EmptyVectorError: the vectors are empty (Req 4.8).
        NonFiniteValueError: an element is NaN or infinite (Req 4.9).
        ZeroNormError: a norm is at or below 1e-12 (Req 4.7).
    """
    if len(left) != len(right):
        raise DimensionMismatchError(
            f"Vectors have different lengths: left has {len(left)}, "
            f"right has {len(right)}."
        )
    if len(left) == 0:
        raise EmptyVectorError("Both vectors are empty; cosine similarity is undefined.")

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)

    if not np.all(np.isfinite(a)):
        raise NonFiniteValueError("The left vector holds a non-finite element.")
    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError("The right vector holds a non-finite element.")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= ZERO_NORM_THRESHOLD:
        raise ZeroNormError(
            f"The left vector has a degenerate norm of {norm_a!r}, "
            f"at or below {ZERO_NORM_THRESHOLD}."
        )
    if norm_b <= ZERO_NORM_THRESHOLD:
        raise ZeroNormError(
            f"The right vector has a degenerate norm of {norm_b!r}, "
            f"at or below {ZERO_NORM_THRESHOLD}."
        )

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, similarity))
