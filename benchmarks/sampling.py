"""Reproducible full CPU-reference benchmark; NOT a YMM4/GPU benchmark.
Run from repo root: python -m benchmarks.sampling > benchmarks/sampling-results.json
Uses synthetic numeric test patterns; no photograph generation or image output.
"""
import json
import platform
from dataclasses import replace
from statistics import median
from time import perf_counter
import numpy as np
from prototype.preview import Params, render

h, w = 384, 512
y, x = np.mgrid[:h, :w].astype(np.float32)
soft = np.stack([.5 + .45*np.sin(x/31), .5 + .45*np.cos(y/27), .5 + .45*np.sin((x+y)/41)], axis=-1)
checker = np.repeat((((x//4 + y//4) % 2)[..., None]), 3, axis=-1)
results = []
for name, src in [('smooth', soft), ('checker', checker)]:
    for steps, strength in [(16, 46), (128, 10), (128, 46), (128, 400)]:
        p = Params(dispersion_steps=steps, strength=strength, radius=8, threshold=0)
        timings = [[], []]
        outputs = [None, None]
        # Warm both paths, then alternate order to reduce ordering bias.
        for fast in (False, True):
            render(src, replace(p, fast_sampling=fast))
        for trial in range(3):
            for fast in ([False, True] if trial % 2 == 0 else [True, False]):
                t = perf_counter()
                outputs[int(fast)] = render(src, replace(p, fast_sampling=fast))
                timings[int(fast)].append(perf_counter() - t)
        err = np.abs(outputs[0] - outputs[1])
        mse = np.mean(err**2)
        results.append(dict(pattern=name, steps=steps, strength=strength,
            standard_seconds=median(timings[0]), fast_seconds=median(timings[1]),
            speedup=median(timings[0])/median(timings[1]),
            mae=float(err.mean()), max_error=float(err.max()),
            psnr_db=float(-10*np.log10(mse)) if mse > 0 else None))
print(json.dumps(dict(scope='full NumPy reference render; no GPU timing',
    resolution=[w,h], trials=3, python=platform.python_version(), numpy=np.__version__,
    results=results), indent=2))
