import pytest

from tests.verification.runs.session_env import scrub


@pytest.fixture(autouse=True)
def _no_session_names_from_the_host(monkeypatch):
    """A pod's FS11_*, FS31_*, DRY_RUN and Runpod names must not leak into a test (session_env)."""
    scrub(monkeypatch)
