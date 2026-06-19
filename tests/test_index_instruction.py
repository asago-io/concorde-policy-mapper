"""Tests for RiskIndex.set_query_instruction()."""

from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.index import RiskIndex, _RemoteBiEncoder


def test_set_query_instruction_updates_remote_bi_encoder():
    index = RiskIndex.__new__(RiskIndex)
    mock_remote = MagicMock(spec=_RemoteBiEncoder)
    mock_remote._query_instruction = "old instruction"
    index._remote_bi_encoder = mock_remote
    index._bi_encoder = None

    index.set_query_instruction("new instruction")

    assert mock_remote._query_instruction == "new instruction"


def test_set_query_instruction_raises_without_remote():
    index = RiskIndex.__new__(RiskIndex)
    index._remote_bi_encoder = None
    index._bi_encoder = MagicMock()

    with pytest.raises(ValueError, match=r"(?i)remote"):
        index.set_query_instruction("new instruction")
