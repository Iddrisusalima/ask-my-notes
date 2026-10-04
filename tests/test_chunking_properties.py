"""Property tests for the chunker.

Feature: embeddings-chunking, Properties 5-13.
Each test states the acceptance criteria it validates.
"""

from __future__ import annotations

from hypothesis import example, given
from hypothesis import strategies as st

from askmydocs.chunking import Chunker
from tests.strategies import chunk_config, document_text, multi_chunk_case, unicode_text

SOURCE = "notes/a.md"


# Property 5: Chunk length bounds.
# Validates: Requirements 8.2, 8.3
@given(document_text, chunk_config())
def test_property_5_chunk_length_bounds(text, config) -> None:
    size, overlap = config
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    for chunk in chunks:
        assert chunk.length <= size
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            assert chunk.length == size
        # The last chunk carries new text, so it must exceed the overlap.
        assert overlap < chunks[-1].length <= size


# Property 6: Consecutive chunks advance by exactly the stride.
# Validates: Requirements 8.4, 8.1
@given(document_text, chunk_config())
def test_property_6_constant_stride(text, config) -> None:
    size, overlap = config
    chunker = Chunker(size, overlap)
    chunks = chunker.chunk_text(text, SOURCE)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start_offset - earlier.start_offset == chunker.stride
        assert later.start_offset > earlier.start_offset


# Property 7: Chunks reconstruct the document text exactly.
# Validates: Requirements 8.5
@given(document_text, chunk_config())
@example("a" * 10, (10, 3))
@example("a" * 11, (10, 3))
@example("a" * 17, (10, 3))
def test_property_7_reconstruction(text, config) -> None:
    size, overlap = config
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    if not chunks:
        assert text == ""
        return
    rebuilt = chunks[0].text + "".join(c.text[overlap:] for c in chunks[1:])
    assert rebuilt == text


# Property 8: Overlap invariant between consecutive chunks.
# Validates: Requirements 8.6
@given(multi_chunk_case())
def test_property_8_overlap_invariant(case) -> None:
    text, size, overlap = case
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    if overlap == 0:
        return
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.text[-overlap:] == later.text[:overlap]


# Property 9: Chunk boundaries are deterministic.
# Validates: Requirements 8.7
@given(document_text, chunk_config())
def test_property_9_deterministic_boundaries(text, config) -> None:
    size, overlap = config
    first = Chunker(size, overlap).chunk_text(text, SOURCE)
    second = Chunker(size, overlap).chunk_text(text, SOURCE)
    assert first == second


# Property 10: Offsets index the document text.
# Validates: Requirements 8.11, 8.10
@given(document_text, chunk_config())
def test_property_10_offsets_index_the_text(text, config) -> None:
    size, overlap = config
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    for chunk in chunks:
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text
        assert chunk.end_offset - chunk.start_offset == len(chunk.text)
        assert 0 <= chunk.start_offset <= chunk.end_offset <= len(text)
        assert chunk.source_path == SOURCE


# Property 11: Ordinal indices are consecutive from zero.
# Validates: Requirements 8.12
@given(document_text, chunk_config())
def test_property_11_consecutive_ordinals(text, config) -> None:
    size, overlap = config
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# Property 12: Chunk count matches the closed-form model.
# Validates: Requirements 8.13, 8.8, 8.9, 8.1
@given(document_text, chunk_config())
@example("", (10, 3))
@example("a", (10, 3))
@example("a" * 10, (10, 3))
@example("ab", (10, 3))
def test_property_12_chunk_count_model(text, config) -> None:
    size, overlap = config
    chunker = Chunker(size, overlap)
    chunks = chunker.chunk_text(text, SOURCE)
    assert len(chunks) == chunker.expected_chunk_count(len(text))
    if len(text) == 0:
        assert chunks == []
    elif len(text) <= size:
        assert len(chunks) == 1 and chunks[0].text == text
    else:
        assert len(chunks) >= 2
    # Requirement 8.13: only the final chunk may end at the text length.
    for chunk in chunks[:-1]:
        assert chunk.end_offset < len(text)
    if chunks:
        assert chunks[-1].end_offset == len(text)


# Property 13: All counting is in Unicode code points.
# Validates: Requirements 8.14
@given(unicode_text, chunk_config(max_size=40))
def test_property_13_code_point_counting(text, config) -> None:
    size, overlap = config
    chunks = Chunker(size, overlap).chunk_text(text, SOURCE)
    for chunk in chunks:
        assert chunk.length <= size
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text
    if chunks:
        rebuilt = chunks[0].text + "".join(c.text[overlap:] for c in chunks[1:])
        assert rebuilt == text
    # Guard: on non-ASCII input the byte length differs from the code point
    # length, so a byte-counting implementation cannot pass this silently.
    if any(ord(ch) > 127 for ch in text):
        assert len(text.encode("utf-8")) != len(text)
