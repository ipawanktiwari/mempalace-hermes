"""pytest configuration for mempalace-hermes tests.

Mocks Hermes module imports before the plugin module is loaded,
so tests can run without a full Hermes installation.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock Hermes module dependencies BEFORE the plugin is imported anywhere
# ---------------------------------------------------------------------------

# agent.memory_provider provides the MemoryProvider base class
hermes_memory_provider = MagicMock()
hermes_memory_provider.MemoryProvider = object  # simple base class

sys.modules["agent"] = MagicMock()
sys.modules["agent.memory_provider"] = hermes_memory_provider

# tools.registry provides tool_error helper
sys.modules["tools"] = MagicMock()
sys.modules["tools.registry"] = MagicMock()
sys.modules["tools.registry"].tool_error = lambda msg: f"ERROR: {msg}"

# hermes_cli.config provides load_config (used by _load_config)
sys.modules["hermes_cli"] = MagicMock()
sys.modules["hermes_cli.config"] = MagicMock()


# ---------------------------------------------------------------------------
# Now safe to import the plugin module
# ---------------------------------------------------------------------------
import os
import sys

_plugin_dir = os.path.join(os.path.dirname(__file__), "..", "plugin")
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)


import pytest


@pytest.fixture
def provider():
    """Return a MemPalaceProvider instance with known config (no real CLI)."""
    from __init__ import MemPalaceProvider

    p = MemPalaceProvider(
        config={
            "binary": "",  # empty → no CLI needed
            "results": 5,
            "min_score": 0.3,
            "timeout": 10,
            "deduplicate": True,
            "max_prefetch_chars": 4000,
            "wing": "sessions",
        }
    )
    # Override _available so is_available doesn't try to exec a binary
    p._available = True
    return p
