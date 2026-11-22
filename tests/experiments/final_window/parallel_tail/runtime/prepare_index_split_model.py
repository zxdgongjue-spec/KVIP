"""Overlap disjoint key and metadata swaps with dedicated worker groups."""
import argparse
from pathlib import Path
from prepare_status_model import prepare_status_model
from private_topk import replace_once


from index_layout import adapt_index_split


def prepare(destination):
    prepare_status_model(destination)
    assert "if(t<448)" in (destination / "helpers.h").read_text()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    prepare(p.parse_args().output.resolve())
