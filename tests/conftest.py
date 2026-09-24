"""Shared test configuration.

Two guarantees are established here, once, for the whole session:

* Hypothesis profiles (Requirement 13.7). Every property test runs at least 100
  examples. The ``pure`` profile runs 200 and is the default; the
  ``filesystem`` profile runs 100 with a deadline wide enough for real file
  input and output.
* No API key and no network (Requirements 13.5, 13.6). This is enforced rather
  than asserted: the key variables are removed from the environment and socket
  creation raises, so an accidental request fails loudly instead of silently
  reaching a provider.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import timedelta
from typing import Any, NoReturn
from unittest import mock

import pytest
from hypothesis import HealthCheck, settings

#: Environment variables that could point the OpenAI client at a real account.
KEY_VARIABLES = ("OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_BASE_URL")

#: Raised when a test attempts to open a socket.
NETWORK_BLOCKED_MESSAGE = "Test suite attempted a network connection"

# Pure, in-process properties: plenty of examples, a tight deadline.
settings.register_profile(
    "pure",
    max_examples=200,
    deadline=timedelta(milliseconds=500),
)

# Properties that touch the filesystem: fewer examples, a deadline that
# tolerates directory creation, and the too-slow health check suppressed
# because real file input and output is legitimately slow.
settings.register_profile(
    "filesystem",
    max_examples=100,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)

# `pure` is the default. Override for one run with HYPOTHESIS_PROFILE=filesystem.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "pure"))


@pytest.fixture(autouse=True, scope="session")
def _no_network_no_key() -> Iterator[None]:
    """Remove every API key variable and make socket creation raise.

    Patching ``socket`` rather than the OpenAI client catches an accidental
    request through any path: a stray HTTP call, a model download from a model
    hub, a telemetry ping.
    """
    saved = {name: os.environ.pop(name, None) for name in KEY_VARIABLES}

    def _blocked(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(NETWORK_BLOCKED_MESSAGE)

    patches = [
        mock.patch("socket.socket", _blocked),
        mock.patch("socket.create_connection", _blocked),
    ]
    for patch in patches:
        patch.start()
    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.stop()
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
