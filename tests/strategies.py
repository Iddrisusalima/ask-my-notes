"""Shared Hypothesis strategies.

One rule shapes this module: degenerate draws are **repaired, not filtered**.
Using ``assume()`` to throw away a zero-norm vector would silently reduce the
number of examples actually exercised, so the 100-example floor the specs
require would stop being honest. Instead a degenerate vector is rescaled into a
usable one, so every draw counts.
"""

from __future__ import annotations

from hypothesis import strategies as st

from askmydocs.models import Chunk

#: Element magnitude cap. Chosen so that scaling by 1e6 in the scale-invariance
#: property cannot overflow float64.
MAX_ELEMENT = 1e3

#: Mixed alphabet: ASCII, an accented letter, a combining sequence, an
#: astral-plane emoji, and whitespace. Exercises code-point counting.
MIXED_ALPHABET = ["a", "b", " ", "\n", "\t", "\u00e9", "e\u0301", "\U0001F600", "\u4e2d"]

finite_element = st.floats(
    min_value=-MAX_ELEMENT,
    max_value=MAX_ELEMENT,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def vector(draw, min_size: int = 1, max_size: int = 64) -> list[float]:
    """A finite vector whose norm is comfortably above the degenerate threshold.

    A drawn all-zero or near-zero vector is repaired by setting its first
    element to 1.0 rather than discarded.
    """
    values = draw(st.lists(finite_element, min_size=min_size, max_size=max_size))
    if not any(abs(v) > 1e-6 for v in values):
        values[0] = 1.0
    return values


@st.composite
def vector_pair(draw, min_size: int = 1, max_size: int = 64):
    """Two vectors of equal length, both non-degenerate."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(vector(size, size)), draw(vector(size, size))


#: Positive scale factors spanning six orders of magnitude either side of 1.
scale_factor = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def chunk_config(draw, max_size: int = 200) -> tuple[int, int]:
    """A valid (chunk_size, chunk_overlap) pair, overlap strictly below size."""
    size = draw(st.integers(min_value=1, max_value=max_size))
    overlap = draw(st.integers(min_value=0, max_value=size - 1))
    return size, overlap


document_text = st.text(min_size=0, max_size=600)

unicode_text = st.lists(
    st.sampled_from(MIXED_ALPHABET), min_size=0, max_size=80
).map("".join)


@st.composite
def multi_chunk_case(draw, max_size: int = 60):
    """A (text, size, overlap) triple guaranteed to produce 2 or more chunks.

    The text length is derived from the drawn chunk size, so the multi-chunk
    branch is reached on essentially every example instead of only when a random
    text happens to be long enough.
    """
    size, overlap = draw(chunk_config(max_size=max_size))
    stride = size - overlap
    extra = draw(st.integers(min_value=1, max_value=3 * stride))
    text = draw(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=size + extra,
            max_size=size + extra,
        )
    )
    return text, size, overlap


def make_chunk(index: int, text: str = "x", source_path: str = "notes/a.md") -> Chunk:
    """A Chunk with self-consistent offsets, for store tests."""
    return Chunk(
        text=text,
        source_path=source_path,
        index=index,
        start_offset=0,
        end_offset=len(text),
    )
