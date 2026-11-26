from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from pr31_capacity_table import markdown, same_long_input, summary_row


def test_table_uses_two_pr31_wide_inputs_for_li3_and_length_deltas():
    row = summary_row(100, {'graph': 98}, {'graph': 99.25, 'LI3': 20}, {'graph': 160, 'LI3': 95})
    assert row == (100, 98, 99.25, 160, 1.25, 60.75, 75)
    other = summary_row(90, {'graph': 85}, {'graph': 80.75, 'LI3': 18}, {'graph': 180, 'LI3': 90})
    rendered = markdown({10: row, 4: other}, Path('a'), Path('b'))
    assert '| **五组等权** | 95.00 | 91.50 | 90.00 | 170.00 | -1.50 | +80.00 | +73.50 |' in rendered


def test_long_inputs_can_have_different_legal_tied_winners_between_blocks():
    first = dict(source_kind='rank_preserving_short_capture', source_oracle_digest='original',
                 source_page_map_sha256='mapped', long_operand_sha256={'query': 'q'},
                 long_state_sha256={'layout': 'l'}, short_unique={'cold': 100},
                 native_queries=64, new_winners=0, long_unique={'cold': 101})
    assert same_long_input(first, dict(first, long_unique={'cold': 99}))
    assert not same_long_input(first, dict(first, source_page_map_sha256='other'))
