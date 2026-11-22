"""Exercise 2048 Index changes (512 per core), with old/new worker layouts."""
import argparse
from pathlib import Path
import re
from prepare_status_model import prepare_status_model
from index_layout import restore_index_baseline
from prepare_index_split_model import prepare as prepare_split


def prepare(destination, split):
    (prepare_split if split else prepare_status_model)(destination)
    if not split:
        restore_index_baseline(destination)
    path=destination/'driver.cpp';s=path.read_text()
    # 1902 prior moves +146 final moves fills exactly the direct LRU capacity.
    # Move prior tokens farther from the end so the whole range stays in history.
    for old,new in ((84,1902),(230,2048),(596,2414)):
        s=re.sub(rf'\b{old}\b',str(new),s)
    path.write_text(s)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--split',action='store_true');args=p.parse_args()
    prepare(args.output.resolve(),args.split)
