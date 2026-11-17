#!/usr/bin/env python3
"""Export source-count tradeoffs, not device timing charts."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    p=argparse.ArgumentParser();p.add_argument('grid',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=list(csv.DictReader(a.grid.open()))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for i,w in enumerate((2,8)):
        data=[r for r in rows if r['windows']==str(w) and (r['post'],r['prepare'],r['service'])==('2','2','1')
              and int(r['requested_hot_partitions'])>0 and int(r['requested_cold_partitions'])>0]
        for j,(field,title) in enumerate((('score_critical_128key_iterations','Critical 128-key iterations (Q4)'),
                                        ('prepared_q_weight_qscale_bytes_each_direction','Prepared input per direction (MiB)'))):
            grid=np.zeros((28,28))
            for r in data:
                value=float(r[field])/(1024**2 if j else 1)
                grid[int(r['requested_cold_partitions'])-1,int(r['requested_hot_partitions'])-1]=value
            ax=axes[i,j];norm=LogNorm(vmin=grid.min(),vmax=grid.max()) if j==0 else None
            plot=ax.imshow(grid,origin='lower',extent=(.5,28.5,.5,28.5),aspect='equal',cmap='viridis',norm=norm)
            fig.colorbar(plot,ax=ax,shrink=.85);ax.set_title(f'W{w}: {title}',fontsize=10)
            ax.set_xlabel('P_hot');ax.set_ylabel('P_cold');ax.set_xticks((1,4,8,16,26,28));ax.set_yticks((1,8,16,26,28))
            for name,ph,pc in [('A',4,8),('B',8,16),('C',26,28 if w==2 else 26)]:
                ax.scatter(ph,pc,s=50,facecolors='none',edgecolors='white',linewidths=1.7)
                ax.annotate(name,(ph,pc),xytext=(5,-10),textcoords='offset points',color='white',weight='bold')
    fig.suptitle('B1 / Q4 / H128K / R16K / C28 / V56\nSource counts: lower copy volume trades against more score iterations',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170)
    fig.savefig(a.output.with_suffix('.svg'))
    print(a.output)

if __name__=='__main__':main()
