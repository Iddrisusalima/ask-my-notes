"""Retry policy and failure classification.

``sleep`` is injected so tests exercise ten retries in microseconds instead of
ninety seconds. That keeps the backoff bounds a data assertion rather than a
timing one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Final, TypeVar

from askmydocs.errors import EmbeddingFailedError

T = TypeVar("T")

#: Requirement 2.9 bounds each wait to at least 1 and at most 30 seconds.
MIN_DELAY: Final = 1.0
MAX_DELAY: Final = 30.0


class FailureKind(Enum):
    """Whether a provider failure is worth retrying."""

    TRANSIENT = auto()  # timeout, connection reset, rate limit -> retry (Req 2.9)
    TERMINAL = auto()  # bad credentials, rejected input -> no retry (Req 2.10)


#: Substrings that identify a retryable failure. Matched on the exception type
#: name and message, so no provider SDK needs importing here.
_TRANSIENT_MARKERS: Final = (
    "timeout",
    "ratelimit",
    "rate limit",
    "connection",
    "temporarily",
    "unavailable",
    "internalserver",
    "toomanyrequests",
)

#: Substrings that identify a failure retrying cannot fix.
_TERMINAL_MARKERS: Final = (
    "authentication",
    "permissiondenied",
    "badrequest",
    "invalidrequest",
    "notfound",
    "unauthorized",
    "apikey",
    "api key",
)


def classify_failure(error: Exception) -> FailureKind:
    """Classify a provider failure.

    Terminal markers are checked first: an authentication error mentioning a
    connection must not be retried just because it says "connection".
    Unrecognized failures are treated as terminal, so an unknown error surfaces
    immediately instead of being retried pointlessly.
    """
    haystack = f"{type(error).__name__} {error}".lower()
    if any(marker in haystack for marker in _TERMINAL_MARKERS):
        return FailureKind.TERMINAL
    if any(marker in haystack for marker in _TRANSIENT_MARKERS):
        return FailureKind.TRANSIENT
    return FailureKind.TERMINAL


class RetryPolicy:
    """Retries transient failures with clamped exponential backoff."""

    def __init__(
        self,
        max_retry_attempts: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retry_attempts = max_retry_attempts
        self._sleep = sleep

    def delay_for_attempt(self, attempt: int) -> float:
        """Backoff for retry ``attempt``, clamped into [1, 30] seconds.

        Gives 1, 2, 4, 8, 16, 30, 30 ... which satisfies Requirement 2.9 across
        the whole permitted range of 0 to 10 attempts.
        """
        return min(MAX_DELAY, max(MIN_DELAY, float(2**attempt)))

    def run(self, operation: Callable[[], T], description: str) -> T:
        """Run ``operation``, retrying transient failures.

        Raises:
            EmbeddingFailedError: after the attempts are exhausted, naming the
                operation, the attempt count, and the reason (Requirement 2.7).
            Exception: a terminal failure is re-raised unchanged, with no retry
                (Requirement 2.10).
        """
        attempts = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                if classify_failure(exc) is FailureKind.TERMINAL:
                    raise
                if attempts >= self.max_retry_attempts:
                    raise EmbeddingFailedError(
                        f"{description} failed after {attempts + 1} attempt(s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                self._sleep(self.delay_for_attempt(attempts))
                attempts += 1
