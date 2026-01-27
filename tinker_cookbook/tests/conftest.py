"""Pytest configuration for tinker_cookbook tests."""

import pytest

# Register pytest-asyncio marker if pytest-asyncio is available
try:
    import pytest_asyncio
    pytest_asyncio.plugin  # Check if plugin is available
except ImportError:
    # pytest-asyncio not installed, tests will need to use asyncio.run() wrapper
    pass
