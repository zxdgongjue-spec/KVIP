"""Freeze current TopK-prefetch B; optionally add PR15 recent Index seeding."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--abc-tail-tokens', type=int, default=384)
    args = parser.parse_args()
    if not 0 <= args.abc_tail_tokens <= 2048:
        parser.error('--abc-tail-tokens must be in [0,2048]')
    here = Path(__file__).resolve().parent
    # Both sides start from the exact current B, never the old bitmap baseline.
    subprocess.run([sys.executable, str(here / 'prepare.py'), str(args.output),
                    '--topk-prefetch'], check=True)
    dest = args.output.resolve()
    kernel = dest / 'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    frozen = {str(p.relative_to(kernel)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(kernel.rglob('*')) if p.is_file()}
    if not args.baseline:
        subprocess.run(['patch', '-p1', '--batch', '--forward', '--fuzz=0',
                        '-i', str(here / 'recent_seed.patch')], cwd=dest, check=True)
        shutil.copyfile(here / 'recent_seed.h', kernel / 'indexer/recent_seed.h')
        (kernel / 'indexer/recent_config.h').write_text(
            f'#pragma once\n#define KVIP_RECENT_TOKENS {args.abc_tail_tokens}\n')
    manifest = dict(variant='A_current_B' if args.baseline else 'B_current_B_plus_recent',
                    abc_tail_tokens=args.abc_tail_tokens, history=131072,
                    recent_begin=131072-args.abc_tail_tokens, recent_end=131072,
                    semantics='Configured current ABC tail length; Index-only seed, identical KV residency in A/B',
                    pr15_commit='4385ac75435c3e872b5e77a4b66e98f8fa5838a8',
                    frozen_current_b_sha256=frozen)
    (dest / 'recent_experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Recent Index experiment: {manifest['variant']}; configured ABC tail="
          f"[{manifest['recent_begin']},{manifest['recent_end']}); causal clipping per Q")


if __name__ == '__main__':
    main()
