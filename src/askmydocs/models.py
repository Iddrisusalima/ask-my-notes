"""Data models passed between the pipeline stages.

Every model is a frozen dataclass, so a value handed to one stage cannot be
mutated by another. Chunk offsets are Unicode code point counts, start
inclusive and end exclusive, and they index ``Document.text`` directly
(Requirements 8.11, 8.14, 7.10).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """The text of one loaded source file, plus where it came from.

    Requirement 7.3.

    Attributes:
        text: Full document text, newline-normalized, otherwise untransformed.
        source_path: Forward-slash path relative to the notes folder. Unique
            among the loaded documents.
        file_type: Lower-cased extension without the leading period.
    """

    text: str
    source_path: str
    file_type: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of one Document's text.

    Requirements 8.10, 8.11, 8.12.

    Attributes:
        text: The slice itself.
        source_path: The source document's relative path.
        index: Zero-based ordinal within the source document.
        start_offset: Inclusive start, in code points.
        end_offset: Exclusive end, in code points.
    """

    text: str
    source_path: str
    index: int
    start_offset: int
    end_offset: int

    @property
    def length(self) -> int:
        """Code point length. Equals ``end_offset - start_offset`` (Req 8.11)."""
        return self.end_offset - self.start_offset

    @property
    def chunk_id(self) -> str:
        """Stable identifier, unique across the corpus.

        Unique because Requirement 7.3 makes ``source_path`` unique per document
        and 8.12 makes ``index`` unique within a document. Week 2 hands these
        straight to Chroma, which requires caller-supplied ids.
        """
        return f"{self.source_path}#{self.index}"

    def to_metadata(self) -> dict[str, str | int]:
        """Flat metadata mapping, holding exactly the fields Req 8.10 mandates.

        Flat and scalar-valued because that is the shape a vector database
        metadata column accepts.
        """
        return {
            "source_path": self.source_path,
            "index": self.index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass(frozen=True)
class StoredRecord:
    """One chunk held in a store, with its embedding and arrival order.

    Requirements 10.2, 10.4.

    ``embedding`` is a tuple rather than a list so a caller cannot mutate stored
    state after the fact and silently break the chunk-to-vector association.
    """

    chunk: Chunk
    embedding: tuple[float, ...]
    insertion_index: int


@dataclass(frozen=True)
class SearchHit:
    """One result of a store query.

    Requirements 10.9, 10.15. ``insertion_index`` is carried so ties in
    ``score`` break deterministically by arrival order.
    """

    chunk: Chunk
    score: float
    insertion_index: int
