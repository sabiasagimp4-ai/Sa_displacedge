# Sa_displacedge benchmark comparison

Input: attached `image(2).jpeg`, 1536×1152 RGB. Each value is one full
`prototype/preview.py` render on the same Linux reference environment. The
prototype is a CPU reference, not a YMM4/Direct2D GPU measurement.

| Case | Main stressor | Before (s) | After (s) | Change |
|---|---|---:|---:|---:|
| 01 default | 16 dispersion taps, detail 4 | 12.992 | 11.482 | -11.6% |
| 02 calm edge | turbulence 0, 3 taps | 4.115 | 4.469 | +8.6% |
| 03 wild | detail 6, 48 taps | 24.619 | 25.603 | +4.0% |
| 04 large soft | radius 192, strength 140 | 12.059 | 10.698 | -11.3% |
| 05 fine turbulent | detail 6, 32 taps | 16.101 | 17.288 | +7.4% |
| 06 zero dispersion | dispersion 0, steps 128 | 53.919 | 4.790 | **-91.1%** |
| 07 classic RGB | 3 taps | 7.195 | 7.025 | -2.4% |
| 08 max stress | strength 200, detail 6, 128 taps | 66.553 | 61.408 | -7.7% |

The before/after figures use identical parameters for all rows. Case 08 uses
the old stress values (`seed=999`, `phase=1000`) so the timing comparison is
fair. With the newly expanded slider maxima (`seed=10000`, `phase=10000`),
the same case rendered in 71.756 s.

The largest exact optimization is case 06: when dispersion is zero, every
tap samples the same coordinate, so replacing 128 fetches with one fetch
does not change the normalized result. The generated 8-case visual matrix is
in `baseline_outputs/contact_sheet_optimized.png`.
