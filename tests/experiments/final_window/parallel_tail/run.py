"""Build and verify the experimental four-Q tail with a simulated CQ delay."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--kvaf-include',required=True,type=Path)
p.add_argument('--baseline-us',type=float,default=80.0)
p.add_argument('--case',type=int,choices=(1,2,3,4,5,8,9,10,11,12,15,16),default=1)
a=p.parse_args();out=a.output.resolve();here=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(here/'prepare.py'),str(out)],check=True)
(out/'kvaf-include').symlink_to(a.kvaf_include.resolve(),target_is_directory=True)
for command,log in [(['cmake','-S',str(out),'-B',str(out/'build')],'configure.log'),(['cmake','--build',str(out/'build'),'-j4'],'build.log'),(['msprof','op','simulator',f'--application={out}/build/real_topk_sim {a.case}',f'--output={out}/profile','--soc-version=Ascend950PR_9599'],'run.log')]:
    print(f'Running {command[0]}; log={out/log}',flush=True)
    with (out/log).open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(here/'report.py'),str(out),'--baseline-us',str(a.baseline_us)],check=True)
