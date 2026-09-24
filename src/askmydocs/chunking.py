"""Fixed-size chunking with overlap.

A strict character-count chunker: no word or sentence boundary snapping. That
strictness is what makes the reconstruction property exact (Requirement 8.5) --
concatenating the first chunk with every later chunk minus its leading overlap
returns the source text character for character.

All counting is in Unicode code points, never bytes and never grapheme
clusters, which plain ``str`` slicing gives for free (Requirement 8.14).

The stop condition is the subtle part. Emission halts as soon as a chunk ends at
the text length (Requirement 8.13). Without that, a stride advance past the end
would emit a short trailing chunk wholly contained in the previous chunk's
overlap region -- a chunk carrying no new text, which would also break the
guarantee that the last chunk is longer than the overlap (Requirement 8.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from askmydocs.errors import ChunkerConfigError
from askmydocs.models import Chunk, Document


@dataclass(frozen=True)
class Chunker:
    """Splits text into overlapping fixed-size chunks.

    Attributes:
        chunk_size: Maximum chunk length in code points. At least 1.
        chunk_overlap: Code points repeated from the previous chunk. In
            ``[0, chunk_size - 1]``, so the stride is always positive.
    """

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        """Reject invalid settings, mirroring Requirements 1.8 and 1.9.

        Duplicated here rather than trusted from Configuration so a Chunker
        built directly in a test or in the experiment script cannot hold
        settings that would make the stride zero or negative.
        """
        if not isinstance(self.chunk_size, int) or isinstance(self.chunk_size, bool):
            raise ChunkerConfigError(
                f"chunk_size must be an integer, got {type(self.chunk_size).__name__}."
            )
        if not isinstance(self.chunk_overlap, int) or isinstance(self.chunk_overlap, bool):
            raise ChunkerConfigError(
                f"chunk_overlap must be an integer, got "
                f"{type(self.chunk_overlap).__name__}."
            )
        if self.chunk_size < 1:
            raise ChunkerConfigError(
                f"chunk_size is {self.chunk_size}; the permitted range is 1 to 10000."
            )
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ChunkerConfigError(
                f"chunk_overlap is {self.chunk_overlap} and chunk_size is "
                f"{self.chunk_size}; the permitted range for chunk_overlap is 0 to "
                f"{self.chunk_size - 1} inclusive, so that the stride stays positive."
            )

    @property
    def stride(self) -> int:
        """Code points advanced between consecutive chunk starts (Req 8.4).

        Always at least 1, guaranteed by ``__post_init__``.
        """
        return self.chunk_size - self.chunk_overlap

    def expected_chunk_count(self, text_length: int) -> int:
        """Closed-form chunk count, used as a reference model in tests.

        Zero for empty text, one when the text fits in a single chunk, and
        otherwise one chunk plus one per stride needed to cover the remainder.
        """
        if text_length <= 0:
            return 0
        if text_length <= self.chunk_size:
            return 1
        remainder = text_length - self.chunk_size
        return 1 + -(-remainder // self.stride)  # ceiling division

    def chunk_text(self, text: str, source_path: str) -> list[Chunk]:
        """Split ``text`` into chunks in source order.

        Returns an empty list for empty text (Requirement 8.9) and exactly one
        chunk holding the full text when it fits, including when the text is
        shorter than the overlap (Requirement 8.8).
        """
        length = len(text)
        if length == 0:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0
        while True:
            end = min(start + self.chunk_size, length)
            chunks.append(
                Chunk(
                    text=text[start:end],
                    source_path=source_path,
                    index=index,
                    start_offset=start,
                    end_offset=end,
                )
            )
            if end == length:
                # Requirement 8.13: the text is fully covered. Advancing again
                # would emit a chunk carrying no new text.
                break
            start += self.stride
            index += 1
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a Document, carrying its source path onto every chunk."""
        return self.chunk_text(document.text, document.source_path)
