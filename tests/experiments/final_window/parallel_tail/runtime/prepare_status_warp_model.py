"""Generate the integrated separate-warp status publication model."""
import argparse
from pathlib import Path
from prepare_status_model import prepare_status_model


def prepare(destination):
    prepare_status_model(destination)
    assert 'if(t==64 || t==65)' in (destination / 'terminal_finish.h').read_text()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    prepare(parser.parse_args().output.resolve())
