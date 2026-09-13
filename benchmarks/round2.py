"""Compare additional optimizations to PR #3's first implementation.
python -m benchmarks.round2 --baseline b4474d341630dbad4415d87b4042fdcec017d75f
Full CPU reference timings only, no GPU inference. No image files generated.
"""
import argparse, json, subprocess, sys, types, tracemalloc, platform
from time import perf_counter
from statistics import median
import numpy as np
from prototype import preview as current

args=argparse.ArgumentParser()
args.add_argument('--baseline',default='b4474d341630dbad4415d87b4042fdcec017d75f')
ref=args.parse_args().baseline
old=types.ModuleType('baseline');sys.modules['baseline']=old
exec(subprocess.check_output(['git','show',ref+':prototype/preview.py'],text=True),old.__dict__)
h,w=384,512
y,x=np.mgrid[:h,:w].astype(np.float32)
smooth=np.stack([.5+.45*np.sin(x/31),.5+.45*np.cos(y/27),.5+.45*np.sin((x+y)/41)],-1)
dense=np.random.default_rng(4).random((h,w,3),dtype=np.float32)
sparse=np.zeros_like(dense);sparse[144:240,192:320]=[.9,.6,.2]
cases=[('smooth',smooth,dict()),('dense',dense,dict()),
       ('sparse',sparse,dict(radius=8)),('sparse128',sparse,dict(radius=8,dispersion_steps=128))]
rows=[]
for name,src,kwargs in cases:
 for fast in (False,True):
  kwargs=dict(kwargs,fast_sampling=fast)
  modules=[old,current];params=[m.Params(**kwargs) for m in modules]
  timings=[[],[]];outputs=[None,None];peaks=[]
  for i in (0,1): modules[i].render(src,params[i])
  for trial in range(3):
   for i in ([0,1] if trial%2==0 else [1,0]):
    t=perf_counter();outputs[i]=modules[i].render(src,params[i]);timings[i].append(perf_counter()-t)
  err=np.abs(outputs[0]-outputs[1])
  for i in (0,1):
   tracemalloc.start();modules[i].render(src,params[i]);peaks.append(tracemalloc.get_traced_memory()[1]);tracemalloc.stop()
  rows.append(dict(pattern=name,fast=fast,baseline_s=median(timings[0]),new_s=median(timings[1]),
    speedup=median(timings[0])/median(timings[1]),baseline_peak_MB=peaks[0]/1e6,new_peak_MB=peaks[1]/1e6,
    max_error=float(err.max()),mae=float(err.mean())))
print(json.dumps(dict(scope='full CPU NumPy reference, 512x384, 3 alternating timed trials; separate tracemalloc run',
    baseline='b4474d341630dbad4415d87b4042fdcec017d75f (same tree as local benchmark baseline)',
    python=platform.python_version(),numpy=np.__version__,results=rows),indent=2))
