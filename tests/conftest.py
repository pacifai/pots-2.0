from tests import exit_guard


def pytest_unconfigure(config):
    """Last hook of the run: see :mod:`tests.exit_guard`."""
    exit_guard.arm(getattr(config, "_exit_status", 0))


def pytest_sessionfinish(session, exitstatus):
    session.config._exit_status = int(exitstatus)
