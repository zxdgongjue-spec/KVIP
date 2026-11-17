#!/usr/bin/env python3
"""Plot source scheduling counts, not A5 timings or ISA counts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grid = read_csv(args.results / 'primary_grid.csv')
    shortlist = read_csv(args.results / 'shortlist.csv')
    controls = read_csv(args.results / 'controls.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    for ax, windows in zip(axes, (2, 8)):
        rows = [r for r in grid if (r['windows'], r['prepare'], r['service']) ==
                (str(windows), '2', '1')]
        ax.scatter([int(r['score_critical_128key_iterations']) for r in rows],
                   [int(r['prepared_q_weight_qscale_bytes_each_direction']) / 2**20 for r in rows],
                   s=10, alpha=.25, color='#3568a6', label='All Ph/Pc')
        for r in shortlist:
            if int(r['windows']) != windows:
                continue
            x = int(r['score_critical_128key_iterations'])
            y = int(r['prepared_q_weight_qscale_bytes_each_direction']) / 2**20
            ax.scatter([x], [y], color='#c84b31', s=36)
            ax.annotate(f"{r['requested_hot_partitions']}/{r['requested_cold_partitions']}",
                        (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set(xscale='log', xlabel='Sum of longest task loops per wave (128 keys)',
               ylabel='Prepared input payload, each direction (MiB)',
               title=f'W{windows}: direct / phase / inline')
        ax.grid(alpha=.2)
    variants = [('Old staged', '04ddfaa_separate_wave', '0', '0', '0'),
                ('Direct', 'current', '2', '0', '0'),
                ('+ Phase', 'current', '2', '2', '0'),
                ('+ Inline', 'current', '2', '2', '1'),
                ('Invocation\n+ inline', 'current', '2', '1', '1')]
    for offset, windows, color in ((-.18, 2, '#3568a6'), (.18, 8, '#c84b31')):
        heights = []
        for _, label, post, prepare, service in variants:
            rows = [r for r in controls if
                    (r['windows'], r['label'], r['post'], r['prepare'], r['service']) ==
                    (str(windows), label, post, prepare, service)]
            if len(rows) != 1:
                raise ValueError('Expected exactly one matching source control')
            heights.append(int(rows[0]['dispatch_count']))
        bars = axes[2].bar([i + offset for i in range(len(variants))], heights,
                           width=.36, label=f'W{windows}', color=color)
        axes[2].bar_label(bars, fontsize=9)
    axes[2].set(xticks=range(len(variants)), xticklabels=[r[0] for r in variants],
                ylabel='Host dispatch calls per forward', title='Same C32/V64 budget')
    axes[2].tick_params(axis='x', labelsize=9, rotation=20)
    axes[2].legend()
    axes[2].set_ylim(0, max(int(r['dispatch_count']) for r in controls) * 1.12)
    fig.suptitle('B1 / Q4 / H128K / resident16K — source counts only; no device speed ranking')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
