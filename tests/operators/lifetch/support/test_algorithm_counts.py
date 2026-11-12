import numpy as np
import pytest

from diagnostics import algorithm_counts


def fixture():
    mapping = np.full((1, 4096), -1, dtype=np.int32)
    # Physical order differs from admission order across two windows.
    mapping[0, [0, 3000, 3001]] = [2050, 2048, 2049]
    selected = np.broadcast_to(np.arange(2048), (1, 4, 2048)).copy()
    selected[0, 1, 0] = 3000
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 9], rows[0, 247] = 3, 1
    rows[0, 16:18] = [1, 2]
    posts = np.zeros((1, 4, 8), dtype=np.int32)
    posts[0, 1, 0] = 1
    return mapping, selected, rows, posts


def test_window_admission_order_query_union_and_exact_unused():
    result = algorithm_counts(*fixture(), 2048, 2, hot_valid=np.ones((1, 2048)), token_bytes=2624)
    row = result['requests'][0]
    assert row['selected_unique'] == 2049 and row['initial_cold'] == 1
    assert [item['unused'] for item in row['windows']] == [0, 1]
    assert [item['cumulative_unused'] for item in row['windows']] == [0, 1]
    assert [item['new_unique'] for item in row['queries']] == [2048, 1, 0, 0]
    assert [item['final_posted'] for item in row['queries']] == [0, 1, 0, 0]
    assert row['KV_read_bytes'] == 4 * 2624
    assert row['useful'] is None and row['redundant'] is None


@pytest.mark.parametrize('corrupt', ['ordinal', 'window', 'final'])
def test_inconsistent_identity_or_counts_are_not_silently_summarized(corrupt):
    mapping, selected, rows, posts = fixture()
    if corrupt == 'ordinal':
        mapping[0, 3000] = 2050
    elif corrupt == 'window':
        rows[0, 16] += 1
    else:
        posts[0, 1, 0] += 1
    with pytest.raises(ValueError, match='match'):
        algorithm_counts(mapping, selected, rows, posts, 2048, 2)


def test_missing_initial_state_is_not_assumed_all_hot():
    result = algorithm_counts(*fixture(), 2048, 2)
    assert result['totals']['initial_cold'] is None
    assert result['totals']['cold_miss_rate'] is None
    assert result['requests'][0]['queries'][0]['initial_local'] is None


def test_abc_is_local_and_not_counted_as_initial_cold():
    result = algorithm_counts(*fixture(), 2048, 2, hot_valid=np.ones((1, 2048)),
                              tail_bounds=np.array([[3000, 3002]]))
    assert result['requests'][0]['initial_hot_valid'] == 2048
    assert result['requests'][0]['initial_cold'] == 0

    assert result['requests'][0]['queries'][1]['initial_local'] == 1
