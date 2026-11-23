import os
import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).with_name('replay_cutoff_tail_npu4.sh')
PEER = Path('tests/operators/lifetch/peer_fused.sh')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()


@pytest.mark.parametrize('state', ['clean', 'dirty', 'wrong-head', 'parent-repo', 'broken-gitfile'])
def test_fill_runs_only_pinned_reader_and_preserves_existing_tree(tmp_path, state):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'Replay test')
    git(repo, 'config', 'user.email', 'replay@example.invalid')
    git(repo, 'config', 'commit.gpgsign', 'false')
    (repo / PEER).parent.mkdir(parents=True)
    (repo / PEER).write_text('''set -eu
[[ $PEER_PHYSICAL_NPU == 4 && $ASCEND_RT_VISIBLE_DEVICES == 4 && $PEER_LOCAL_ONLY == 1 ]]
root=$(cd "$(dirname "$0")/../../.." && pwd -P)
[[ $(git -C "$root" rev-parse HEAD) == "$EXPECTED_READER" ]]
printf '%s\\t' "$root" "$@" >> "$PEER_G2_WORK/calls"
printf '\\n' >> "$PEER_G2_WORK/calls"
''')
    git(repo, 'add', '--', PEER)
    git(repo, 'commit', '-qm', 'fixture peer')
    native = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'commit', '-q', '--allow-empty', '-m', 'fixture reader')
    reader = git(repo, 'rev-parse', 'HEAD')

    work = tmp_path / 'work'
    previous = work / 'peer-baseline/fused.fmn5ww'
    current = work / 'peer-baseline/fused.N7n8ZP'
    for archive in [previous, current]:
        archive.mkdir(parents=True)
        (archive / 'logical-ties.txt').write_text(native + '\n')
    runner = work / 'fused-scan-sources' / f'reader-{reader}'
    if state == 'parent-repo':
        git(repo, 'worktree', 'add', '-q', '--detach', runner.parent, reader)
        runner.mkdir()  # Git finds the correct HEAD one directory above this empty directory.
    elif state == 'broken-gitfile':
        runner.mkdir(parents=True)
        (runner / '.git').write_text('gitdir: missing\n')
    else:
        git(repo, 'worktree', 'add', '-q', '--detach', runner,
            native if state == 'wrong-head' else reader)
    if state == 'dirty':
        (runner / PEER).write_text('exit 99\n')
        git(runner, 'add', PEER)
        (runner / PEER).write_text('exit 98\n')
        (runner / 'local-results').write_text('preserve me\n')

    env = dict(os.environ, PEER_G2_WORK=str(work), EXPECTED_READER=reader)
    result = subprocess.run(['bash', LAUNCHER, repo, reader, 'fill-g10-g2', native],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [line.rstrip('\t').split('\t') for line in (work / 'calls').read_text().splitlines()]
    assert [args for _, *args in calls] == [
        ['--serial-g10-only', str(current)], ['--owner-topk', native, '--reference', str(previous), '2']]
    assert calls[0][0] == calls[1][0]
    assert (Path(calls[0][0]) == runner.resolve()) == (state == 'clean')

    if state == 'dirty':
        assert git(runner, 'show', f':{PEER}') == 'exit 99'
        assert (runner / PEER).read_text() == 'exit 98\n'
        assert (runner / 'local-results').read_text() == 'preserve me\n'
    elif state == 'wrong-head':
        assert git(runner, 'rev-parse', 'HEAD') == native
    elif state == 'parent-repo':
        assert not list(runner.iterdir())
    elif state == 'broken-gitfile':
        assert (runner / '.git').read_text() == 'gitdir: missing\n'
