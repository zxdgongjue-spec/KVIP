"""Keep queue/route warp dedicated; distribute output across other warps."""
import argparse
from pathlib import Path
from prepare_status_model import prepare_status_model
from private_topk import replace_once


def prepare(destination):
    prepare_status_model(destination)
    path=destination/'parallel_post.h'
    assert 'if(tid>=128)for(uint32_t i=tid-128;i<2048;i+=384)' in path.read_text()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    prepare(p.parse_args().output.resolve())
