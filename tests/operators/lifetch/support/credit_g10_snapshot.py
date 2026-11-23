"""Export just the input consumed by the archived B24 G10 mainline run."""
import io
import json
from pathlib import Path
import sys
import tarfile


FIELDS = ('pre.metadata','pre.logical_ids','pre.hot_valid','pre.addresses','pre.lru_slots',
          'pre.query_ends','pre.tail_bounds','pre.keys','pre.key_scales','pre.query',
          'pre.query_scale','pre.weights','post.score_keys','post.logical_output')


def write_json(tar, name, value):
    content = json.dumps(value).encode()
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def export(work, out):
    root = work/'peer-baseline/main-five.hF6qQ8'
    group = root/'g10/mainline/results/mainline/cases/diagnose'
    expected = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
    if (root/'physical_npu.txt').read_text().strip() != '4' or (root/'main.txt').read_text().strip() != expected:
        raise ValueError('Wrong B24 G10 reference')
    proof = json.loads((group/'fused/mainline_checks.json').read_text())
    if proof['production_revision'] != expected or proof['runtime']['visible'] != '4':
        raise ValueError('Wrong measured B24 input')
    source = list(map(Path, json.loads((group/'replay_input.json').read_text())['source_paths']))
    if len(source) != 8:
        raise ValueError('Expected eight G10 ranks')
    with tarfile.open(fileobj=out, mode='w|', dereference=True) as tar:
        write_json(tar, 'capture_operand_hashes.json', proof['capture_operand_hashes'])
        if len(set(source)) == 8:
            for dp, path in enumerate(source):
                rank = path.parents[2]
                if path.relative_to(rank).as_posix() != 'round0001/group010/manifest.json' or rank.name != f'dp{dp}':
                    raise ValueError('Unexpected prepared G10 source path')
                for p in (rank/'series.json', path, path.parent/'scores.npz'):
                    tar.add(p, arcname='g10-input/'+rank.name+'/'+p.relative_to(rank).as_posix(), recursive=False)
            return
        if len(set(source)) != 1:
            raise ValueError('Mixed G10 input types')
        path = source[0]
        m = json.loads(path.read_text())
        if m['content'] != 'real_li_synthetic_kv' or m['status'] != 'complete' or m['metadata']['group'] != 10:
            raise ValueError('Wrong merged G10 source')
        original = m['tensors']
        m['tensors'] = {}
        for field in FIELDS:
            spec = dict(original[field])
            src = (path.parent/spec['file']).resolve()
            if not src.is_relative_to(path.parent.resolve()) or not src.is_file():
                raise ValueError(f'Invalid G10 tensor {field}')
            spec['file'] = f'tensors/{field}.npy'
            m['tensors'][field] = spec
            tar.add(src, arcname='g10-input/'+spec['file'], recursive=False)
        write_json(tar, 'g10-input/manifest.json', m)


if __name__ == '__main__':
    export(Path(sys.argv[1]), sys.stdout.buffer)
