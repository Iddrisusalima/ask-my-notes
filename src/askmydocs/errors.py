"""Exception hierarchy for Ask My Docs.

Library code raises these; only script ``main`` functions translate them into
exit statuses. Nothing in this package calls ``sys.exit``.

Every leaf type carries enough detail in its message for a learner to fix the
cause without reading a traceback: the offending value, the permitted range,
and the name of the setting or file involved.
"""

from __future__ import annotations


class AskMyDocsError(Exception):
    """Root of every error this project raises deliberately."""


# --- Configuration ----------------------------------------------------------


class ConfigurationError(AskMyDocsError):
    """A setting is absent, unparsable, or outside its permitted range.

    Requirements 1.3, 1.5, 1.8, 1.9, 1.11, 1.12.
    """


# --- Notes discovery and loading -------------------------------------------


class NotesFolderError(AskMyDocsError):
    """The notes folder is missing or is not a directory. Requirement 6.5."""


class DocumentLoadError(AskMyDocsError):
    """A single source file could not be turned into a Document.

    Raised per file and handled per file: loading continues with the remaining
    files. Requirement 7.7.
    """


class EncryptedPdfError(DocumentLoadError):
    """The PDF is encrypted or password-protected. Requirement 7.11."""


class FileTooLargeError(DocumentLoadError):
    """The file exceeds the 25 MB limit. Requirement 7.12."""


class EmptyDocumentError(DocumentLoadError):
    """The file yielded no non-whitespace text. Requirement 7.8."""


# --- Chunking ---------------------------------------------------------------


class ChunkerConfigError(AskMyDocsError):
    """Chunk size or overlap is invalid.

    Mirrors the Configuration checks so a Chunker built directly, in a test or
    in the experiment script, cannot hold invalid settings.
    Requirements 1.8, 1.9.
    """


# --- Similarity -------------------------------------------------------------


class SimilarityError(AskMyDocsError):
    """A cosine similarity input is not comparable."""


class DimensionMismatchError(SimilarityError):
    """The two vectors have different lengths. Requirement 4.6."""


class EmptyVectorError(SimilarityError):
    """A vector has no elements. Requirement 4.8."""


class NonFiniteValueError(SimilarityError):
    """A vector holds a NaN or an infinity. Requirement 4.9."""


class ZeroNormError(SimilarityError):
    """A vector's Euclidean norm is at or below 1e-12. Requirement 4.7."""


# --- Embedding --------------------------------------------------------------


class EmbeddingError(AskMyDocsError):
    """An embedding could not be produced."""


class EmptyInputError(EmbeddingError):
    """The text is empty or whitespace-only. Requirement 2.6."""


class InputTooLongError(EmbeddingError):
    """The text exceeds the configured maximum input length. Requirement 2.8."""


class EmbeddingFailedError(EmbeddingError):
    """The provider failed and the retry attempts are exhausted.

    Requirements 2.7, 3.7.
    """


# --- Storage ----------------------------------------------------------------


class StoreError(AskMyDocsError):
    """A store operation was rejected. The store is left unchanged."""


class BatchLengthMismatchError(StoreError):
    """The chunk count and the vector count differ. Requirement 10.7."""


class VectorLengthError(StoreError):
    """A vector's length disagrees with the established dimensionality.

    Requirement 10.8.
    """


class DegenerateVectorError(StoreError):
    """A query vector has a norm at or below 1e-12. Requirement 10.14."""


class InvalidKError(StoreError):
    """K is not an integer, or is less than 1. Requirement 10.13."""


# --- Cost guardrails --------------------------------------------------------


class GuardrailError(AskMyDocsError):
    """A run would exceed a configured cost guardrail.

    Raised before the first provider request, so tripping it costs nothing.
    Requirement 11.9.
    """
