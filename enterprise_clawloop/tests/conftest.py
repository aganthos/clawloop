"""Enterprise test configuration.

Enterprise tests CAN import from both clawloop and enterprise_clawloop.
Community tests (tests/) must NEVER import from enterprise_clawloop.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark all enterprise tests."""
    for item in items:
        item.add_marker(pytest.mark.enterprise)
