from tests import exit_guard


def pytest_configure(config):
    """First hook of the run: ``SIGUSR1`` dumps every stack, see :mod:`tests.exit_guard`."""
    exit_guard.register_stacks()


def pytest_unconfigure(config):
    """Last hook of the run: see :mod:`tests.exit_guard`."""
    exit_guard.arm()
