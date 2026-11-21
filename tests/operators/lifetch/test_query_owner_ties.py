"""Strict Q4 replay tie convention, including nonidentity Index layouts."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from query_owner_report import check_cutoff_ids


def test_equal_score_uses_earliest_valid_physical_positions():
    layout = np.array([[5, 2, 1, 4, 3, 0]])
    # Token 2 wins outright, 0 is masked and tokens 1, 3, 4, 5 tie.
    data = SimpleNamespace(logical_score_keys=np.array([[[99, 6, 7, 6, 6, 6]]]),
                           logical_score_valid=np.array([[[False, True, True, True, True, True]]]))
    check_cutoff_ids(data, layout, np.array([[[1, 2, 5]]]))
    with pytest.raises(AssertionError, match=r'Q=1.*missing=\[5\].*extra=\[3\]'):
        check_cutoff_ids(data, layout, np.array([[[1, 2, 3]]]))
