# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from main_five_run import parse_args


@pytest.mark.parametrize(('mode', 'expected'), [
    ([], 20),
    (['--history-cap', '131328', '--production-graph'], 20),
    (['--ablation-policy', 'single'], 10),
    (['--ablation-policy', 'paced'], 10),
])
def test_parent_and_device_worker_agree_on_replay_count(mode, expected):
    args = ['--main', '.', '--trace', '.', '--owner-archive', '.', '--kvaf-root', '.',
            '--output', '.', '--kvaf-server', '127.0.0.1:1', '--group', '7', *mode]
    assert parse_args(args).iterations == expected
    assert parse_args([*args, '--worker']).iterations == expected
