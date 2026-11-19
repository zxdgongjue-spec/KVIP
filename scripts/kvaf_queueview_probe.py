"""Print the identity each fused queue view receives at bind time.

Diagnostics only: adds one `print` after `KvafQueueView` registers itself with
its lease, so a startup log shows whether every group really got its own
channel entity and output buffer. Nothing is written to the file unless the
anchor is found exactly once.

    python3 scripts/kvaf_queueview_probe.py [path/to/adapters/kvaf.py]
"""
import ast
import pathlib
import sys

DEFAULT = '/home/z00605689/vLLM-DSA/vLLM-DSA/third_party/KVIP/python/kvip/adapters/kvaf.py'
P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
MARK = 'QUEUEVIEW'

source = P.read_text()
if MARK in source:
    print('already patched')
    raise SystemExit(0)

lines = source.splitlines(keepends=True)
hits = [index for index, line in enumerate(lines) if line.strip() == 'lease.plans.append(self)']
if len(hits) != 1:
    raise SystemExit('anchor count %d, expected 1' % len(hits))

index = hits[0]
indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
probe = (
    indent
    + 'print("QUEUEVIEW tok=%s entity=0x%x out=0x%x addr=0x%x n=%s w=%s" % ('
    + 'self.context["token_bytes"], self.context["device_channel_entity"], '
    + 'output.data_ptr(), addresses.data_ptr(), addresses.numel(), '
    + 'self.context["worker_count"]), flush=True)'
    + '\n'
)
patched = ''.join(lines[:index] + [probe] + lines[index:])
ast.parse(patched)
P.with_suffix('.py.bak-1plus3').write_text(source)
P.write_text(patched)
print('patched ok')
