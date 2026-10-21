"""One clean environment for every test that runs the job script or the session.

On a pod the environment already holds the session's own names (``FS11_SESSION=fs32-pair``,
``FS11_ALLOW_ANY_GPU``, ``FS11_DEADLINE_S``, ``FS31_CARDS``, the Runpod ids), and a test that copies
``os.environ`` then runs the pod's session instead of its own. FS32's two-A40 run failed 13 job tests
that way. :func:`clean_environ` drops every name the job script or the session reads;
``conftest.py`` applies :func:`scrub` to every test here, so in-process reads are clean too.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

DROP_PREFIXES = ("FS11_", "FS31_", "RUNPOD_")
DROP_NAMES = ("DRY_RUN",)


def is_session_name(name: str) -> bool:
    return name.startswith(DROP_PREFIXES) or name in DROP_NAMES


def clean_environ(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """``os.environ`` without any session name, then ``extra`` on top."""
    env = {k: v for k, v in os.environ.items() if not is_session_name(k)}
    env.update(extra or {})
    return env


def scrub(monkeypatch: Any) -> None:
    for k in [k for k in os.environ if is_session_name(k)]:
        monkeypatch.delenv(k)
