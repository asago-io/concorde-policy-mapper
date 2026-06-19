import sys
from pathlib import Path

# Add tests directory and project root to path for experiments imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.llm import LLMConfig


@pytest.fixture
def mock_config():
    return LLMConfig(base_url="http://localhost:8000/v1", model="test-model", max_context=0)


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client
