"""Conftest for M/M/c queueing test suite.

Shared fixtures and markers used across the test modules.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: cross-module validation tests")

