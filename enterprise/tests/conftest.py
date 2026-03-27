"""Enterprise test configuration.

Enterprise tests CAN import from both clawloop and enterprise.
Community tests (tests/) must NEVER import from enterprise.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark all enterprise tests."""
    for item in items:
        item.add_marker(pytest.mark.enterprise)
