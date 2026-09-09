"""Sa_displacedge prototype preview.

Portable NumPy reference of the "Mirage Edge" liquid-refraction effect, used
to art-direct parameters before porting the algorithm to the YMM4 HLSL
shaders. Mirrors the workflow in Sa_aohue/prototype: render the same
formulas the GPU will run so the look can be tuned without a
Windows/Direct2D toolchain, then keep this script as the numeric reference
for the shader port.

Pipeline (matches ymm/Shaders/EdgeGradient.hlsl + FlowDisplace.hlsl):
  1. OKLab L extraction, Scharr gradient (gx, gy, magnitude) at DetectionScale.
  2. Triple box blur (radius/sqrt(3) x3, same trick as Sa_aohue's Radius blur)
     approximates a Gaussian spread of the *raw* gradient field, so nearby
     opposing edges partially cancel instead of doubling a binary mask.
  3. Threshold + contrast gamma on the blurred magnitude gives a soft mask.
  4. A divergence-free curl-noise field (the analytic gradient of rotated-
     octave value-noise fBm), evaluated at time*flow_speed + phase_offset,
     is blended with the local edge normal by "turbulence" to get a flow
     direction; scaled by strength * mask gives the displacement vector.
  5. The source is swept dispersion_steps times per pixel along that vector
     (scale ranging over 1 +/- dispersion), each tap weighted by an
     approximate spectral response, for a prism-like fringe; steps=3
     reduces to a classic R/G/B split, more steps give a smoother spectrum.
  6. An animated rim glint, colourised through a cosine palette (thin-film
     iridescence), is added along the flow crest.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour helpers (shared domain with Sa_aohue: sRGB -> OKLab L)
# ---------------------------------------------------------------------------

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab_l(rgb_linear: np.ndarray) -> np.ndarray:
    l = 0.4122214708 * rgb_linear[..., 0] + 0.5363325363 * rgb_linear[..., 1] + 0.0514459929 * rgb_linear[..., 2]
    m = 0.2119034982 * rgb_linear[..., 0] + 0.6806995451 * rgb_linear[..., 1] + 0.1073969566 * rgb_linear[..., 2]
    s = 0.0883024619 * rgb_linear[..., 0] + 0.2817188376 * rgb_linear[..., 1] + 0.6299787005 * rgb_linear[..., 2]
    l_, m_, s_ = np.cbrt(np.maximum(l, 0)), np.cbrt(np.maximum(m, 0)), np.cbrt(np.maximum(s, 0))
    return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_


# ---------------------------------------------------------------------------
# Edge gradient (Scharr) at an adjustable detection scale
# ---------------------------------------------------------------------------

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 16.0
_SCHARR_Y = _SCHARR_X.T


def _sample_scalar(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample for a 2-D scalar field with clamp-to-edge."""
    h, w = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    top = img[y0c, x0c] * (1.0 - wx) + img[y0c, x1c] * wx
    bot = img[y1c, x0c] * (1.0 - wx) + img[y1c, x1c] * wx
    return top * (1.0 - wy) + bot * wy


def scharr_gradient(luma: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    # Keep the prototype's fractional spacing in the same domain as the HLSL
    # shader. The old implementation rounded this value to an integer, which
    # made the lower half of the UI range visually identical.
    step = max(0.25, float(step))
    # These taps are uniformly translated grids, not arbitrary coordinates.
    # Interpolate along each axis using 1-D indices instead of allocating
    # eight full-frame coordinate/index/weight grids.
    h, w = luma.shape
    gx = np.zeros_like(luma, dtype=np.result_type(luma, np.float32))
    gy = np.zeros_like(luma, dtype=np.result_type(luma, np.float32))
    horizontal = []
    for dx in (-step, 0.0, step):
        # Form coordinates in float32, like the previous mgrid reference.
        # Fractions may vary slightly per pixel due to float32 rounding.
        x = np.arange(w, dtype=np.float32) + dx
        lo = np.floor(x).astype(np.int32)
        f = x - lo
        horizontal.append(luma[:, np.clip(lo, 0, w-1)] * (1-f) +
                          luma[:, np.clip(lo+1, 0, w-1)] * f)
    for j in range(3):
        y = np.arange(h, dtype=np.float32) + (j-1)*step
        lo = np.floor(y).astype(np.int32)
        f = (y-lo)[:, None]
        y0, y1 = np.clip(lo, 0, h-1), np.clip(lo+1, 0, h-1)
        for i in range(3):
            if _SCHARR_X[j, i] != 0 or _SCHARR_Y[j, i] != 0:
                tap = horizontal[i][y0] * (1-f) + horizontal[i][y1] * f
                gx += _SCHARR_X[j, i] * tap
                gy += _SCHARR_Y[j, i] * tap
    return gx, gy


# ---------------------------------------------------------------------------
# Sa_aohue-style triple box blur (approximates a Gaussian, O(w*h) per pass)
# ---------------------------------------------------------------------------

def _box_blur_axis(arr: np.ndarray, radius: float, axis: int) -> np.ndarray:
    r = int(round(radius))
    if r <= 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    padded = np.pad(arr, pad, mode="edge")
    csum = np.cumsum(padded, axis=axis)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=csum.dtype), csum], axis=axis)
    n = arr.shape[axis]
    # Views replace np.take's full-size copies of both cumulative sums.
    hi_slice = [slice(None)] * arr.ndim
    lo_slice = [slice(None)] * arr.ndim
    hi_slice[axis] = slice(2*r+1, 2*r+1+n)
    lo_slice[axis] = slice(0, n)
    hi, lo = csum[tuple(hi_slice)], csum[tuple(lo_slice)]
    return (hi - lo) / (2 * r + 1)


def gaussian_like_blur(arr: np.ndarray, sigma: float, passes: int = 3) -> np.ndarray:
    if sigma <= 0:
        return arr
    r = sigma / math.sqrt(3.0)
    out = arr
    for _ in range(passes):
        out = _box_blur_axis(out, r, axis=0)
        out = _box_blur_axis(out, r, axis=1)
    return out


# ---------------------------------------------------------------------------
# Curl noise: divergence-free flow from the analytic gradient of a
# rotated-octave value-noise fBm
# ---------------------------------------------------------------------------

def _hash2(p: np.ndarray) -> np.ndarray:
    x = np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7) * 43758.5453123
    return x - np.floor(x)


def _value_noise(p: np.ndarray) -> np.ndarray:
    i = np.floor(p)
    f = p - i
    a = _hash2(i)
    b = _hash2(i + np.array([1.0, 0.0]))
    c = _hash2(i + np.array([0.0, 1.0]))
    d = _hash2(i + np.array([1.0, 1.0]))
    u = f * f * (3.0 - 2.0 * f)
    ux, uy = u[..., 0], u[..., 1]
    return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy


_OCTAVE_ROT = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=np.float32)


def fbm(p: np.ndarray, octaves: int) -> np.ndarray:
    value = np.zeros(p.shape[:-1], dtype=np.result_type(p, np.float32))
    amp, freq = 0.5, 1.0
    pp = p
    for _ in range(max(1, min(8, int(octaves)))):
        value = value + amp * _value_noise(pp * freq)
        pp = pp @ _OCTAVE_ROT.T
        freq *= 2.03
        amp *= 0.5
    return value


def _value_noise_gradient(p: np.ndarray) -> np.ndarray:
    """Analytic gradient of smooth value noise with respect to p=(x,y)."""
    i = np.floor(p)
    f = p - i
    a = _hash2(i)
    b = _hash2(i + np.array([1.0, 0.0], dtype=p.dtype))
    c = _hash2(i + np.array([0.0, 1.0], dtype=p.dtype))
    d = _hash2(i + np.array([1.0, 1.0], dtype=p.dtype))
    u = f * f * (3.0 - 2.0 * f)
    du = 6.0 * f * (1.0 - f)
    x0 = a + (b - a) * u[..., 0]
    x1 = c + (d - c) * u[..., 0]
    dvalue_dx = (1.0 - u[..., 1]) * (b - a) + u[..., 1] * (d - c)
    dvalue_dy = x1 - x0
    return np.stack([dvalue_dx * du[..., 0], dvalue_dy * du[..., 1]], axis=-1)


def fbm_gradient(p: np.ndarray, octaves: int) -> np.ndarray:
    """Gradient of the rotated-octave fBm with respect to its input p."""
    grad = np.zeros_like(p, dtype=np.result_type(p, np.float32))
    pp = p
    axis_x = np.array([1.0, 0.0], dtype=grad.dtype)
    axis_y = np.array([0.0, 1.0], dtype=grad.dtype)
    amp, freq = 0.5, 1.0
    for _ in range(max(1, min(8, int(octaves)))):
        grad_q = _value_noise_gradient(pp * freq) * freq
        grad[..., 0] += amp * np.sum(grad_q * axis_x, axis=-1)
        grad[..., 1] += amp * np.sum(grad_q * axis_y, axis=-1)
        pp = pp @ _OCTAVE_ROT.T
        axis_x = axis_x @ _OCTAVE_ROT.T
        axis_y = axis_y @ _OCTAVE_ROT.T
        freq *= 2.03
        amp *= 0.5
    return grad


def curl_noise(p: np.ndarray, octaves: int, eps: float = 0.5) -> np.ndarray:
    # v = (d psi/dy, -d psi/dx). Computing the gradient analytically reduces
    # four full fBm evaluations per pixel to one. The optional eps argument is
    # retained for API compatibility with older callers; it is no longer used.
    del eps
    gradient = fbm_gradient(p, octaves)
    return np.stack([gradient[..., 1], -gradient[..., 0]], axis=-1)


# ---------------------------------------------------------------------------
# Bilinear sampling with clamp-to-edge, used for the dispersive re-sample
# ---------------------------------------------------------------------------

def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx = np.clip(x - x0, 0.0, 1.0)[..., None]
    wy = np.clip(y - y0, 0.0, 1.0)[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    top = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx
    bot = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx
    return top * (1 - wy) + bot * wy


def iridescent_palette(t: np.ndarray) -> np.ndarray:
    freq = np.array([1.0, 1.0, 1.0])
    phase = np.array([0.0, 0.33, 0.67])
    return 0.5 + 0.5 * np.cos(2 * math.pi * (freq * t[..., None] + phase))


def _gauss(x: float, center: float, sigma: float) -> float:
    d = (x - center) / sigma
    return math.exp(-0.5 * d * d)


def spectrum_weight(t: float) -> np.ndarray:
    """Approximate per-channel spectral response for a dispersion sweep
    position t in [0, 1] (0 = innermost sample, 1 = outermost). Three
    overlapping Gaussian bumps keep the middle of the sweep close to white
    while the extremes read as colour, like real lens chromatic fringing
    rather than a hard three-tap RGB split."""
    return np.array([_gauss(t, 0.85, 0.35), _gauss(t, 0.5, 0.35), _gauss(t, 0.15, 0.35)])


@lru_cache(maxsize=252)
def spectral_table(steps: int, fast: bool = False) -> np.ndarray:
    """Normalized RGB weights + signed positions; mirrors SpectralTable.cs.

    Fast mode linearly redistributes the original weights to at most eight
    nodes. It preserves each channel's zeroth and first moments, but can
    alias sharp/high-frequency content between those nodes.
    """
    steps = max(3, min(128, int(steps)))
    count = min(steps, 8) if fast else steps
    result = np.zeros((count, 4), dtype=np.float64)
    total = np.zeros(3)
    for i in range(steps):
        t = i / (steps - 1)
        w = spectrum_weight(t)
        total += w
        node = t * (count - 1)
        lo = min(int(node), count - 1) if fast else i
        hi = min(lo + 1, count - 1) if fast else i
        f = node - lo if fast else 0.0
        result[lo, :3] += w * (1 - f)
        result[hi, :3] += w * f
    result[:, :3] /= total
    result[:, 3] = np.linspace(-1, 1, count)
    result.setflags(write=False)
    return result


# ---------------------------------------------------------------------------
# Parameters (mirrors the animatable UI exposed by SaDisplacedgeEffect.cs)
# ---------------------------------------------------------------------------

@dataclass
class Params:
    detection_scale: float = 1.5      # px, Sobel/Scharr tap spacing
    threshold: float = 40.0           # 0-255, edge cutoff before blur-softened field
    radius: float = 48.0              # px, spread of the blurred gradient field
    contrast: float = 1.0             # mask gamma
    strength: float = 46.0            # px, max displacement
    turbulence: float = 0.6           # 0-1, blend edge-normal flow -> curl swirl
    turbulence_detail: int = 4        # fBm octaves
    noise_scale: float = 220.0        # px per noise unit (higher = larger swirls)
    flow_speed: float = 1.0           # animation rate multiplier
    dispersion: float = 0.35          # 0-1, sweep spread around scale 1.0
    dispersion_steps: int = 16        # >=3, taps swept across the spread; 3 == classic R/G/B split
    fast_sampling: bool = False       # approximate maximum-eight-tap mode
    iridescence: float = 0.55         # 0-1, glint colour strength
    light_angle_deg: float = 55.0
    seed: float = 0.0
    time: float = 0.6                 # seconds; still-frame phase for the preview
    phase_offset: float = 0.0         # same units as time*flow_speed; shifts the swirl's start point


def render(src: np.ndarray, p: Params, output_mode: str = "composite") -> np.ndarray:
    """src: HxWx3 float32 in [0,1] straight sRGB. Returns HxWx3 float32 sRGB."""
    if output_mode == "composite" and p.strength == 0 and p.iridescence == 0:
        return src.copy()
    h, w = src.shape[:2]
    linear = srgb_to_linear(src)
    luma = oklab_l(linear)
    del linear

    gx, gy = scharr_gradient(luma, p.detection_scale)
    mag = np.hypot(gx, gy)

    gx_b = gaussian_like_blur(gx, p.radius)
    gy_b = gaussian_like_blur(gy, p.radius)
    mag_b = gaussian_like_blur(mag, p.radius)
    del gx, gy, mag

    cutoff = (p.threshold / 255.0) * 0.12
    softness = 0.05
    mask = np.clip((mag_b - cutoff) / max(1e-6, softness), 0.0, 1.0)
    mask = mask ** (1.0 / max(0.01, p.contrast))

    if output_mode == "mask":
        gray = mask
        return np.stack([gray, gray, gray], axis=-1)

    active = mask > 0
    if not active.any():
        if output_mode == "flow":
            result = np.zeros_like(src)
            result[..., :2] = .5
            return result
        return src.copy()

    inv_mag = np.zeros_like(mag_b)
    np.divide(1.0, mag_b, out=inv_mag, where=mag_b > 1e-6)
    edge_normal = np.stack([gx_b * inv_mag, gy_b * inv_mag], axis=-1)

    # phase_offset is added in the same units as time*flow_speed, *before*
    # the 0.35/1.3 rate scaling below, so it shifts the swirl's starting
    # point by a fixed amount regardless of flow_speed -- unlike seed (a
    # spatial coordinate offset, a different pattern entirely), this only
    # moves *where in its cycle* the same pattern currently is.
    anim_time = p.time * p.flow_speed + p.phase_offset
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coord = np.stack([xx, yy], axis=-1) / max(1.0, p.noise_scale)
    coord = coord + p.seed * 17.0
    coord[..., 0] += anim_time * 0.35
    turb = np.clip(p.turbulence, 0.0, 1.0)
    irid = np.clip(p.iridescence, 0.0, 1.0)
    curl = np.zeros_like(coord)
    if turb > 1e-5 or (output_mode == "composite" and irid > 1e-5):
        if active.all():
            curl = curl_noise(coord, p.turbulence_detail)
        elif active.any():
            curl[active] = curl_noise(coord[active], p.turbulence_detail)
    curl_len = np.hypot(curl[..., 0], curl[..., 1])
    curl_norm = curl / np.maximum(1e-6, curl_len)[..., None]
    curl_norm[curl_len <= 1e-5] = np.array([1.0, 0.0], dtype=curl_norm.dtype)

    flow = edge_normal * (1 - turb) + curl_norm * turb
    flow_len = np.maximum(1e-6, np.hypot(flow[..., 0], flow[..., 1]))
    flow = flow / flow_len[..., None]

    disp = flow * (p.strength * mask)[..., None]

    if output_mode == "flow":
        vis = np.stack([
            0.5 + 0.5 * flow[..., 0] * mask,
            0.5 + 0.5 * flow[..., 1] * mask,
            mask,
        ], axis=-1)
        return np.clip(vis, 0, 1)

    if p.strength == 0.0 and p.iridescence == 0.0:
        return src.copy()

    # Sweep dispersion_steps taps from scale (1-dispersion) to (1+dispersion)
    # along the displacement vector, each weighted by an approximate
    # spectral response, and blend them into a smoothly graded prism
    # fringe. steps=3 reduces to the classic R/G/B split; more steps trade
    # compute for a smoother, more continuous spectrum.
    table = spectral_table(int(round(p.dispersion_steps)), p.fast_sampling)
    dense = active.all()
    if dense:
        sample_x, sample_y, displacement = xx, yy, disp
        color_sum = np.zeros_like(src)
    else:
        sample_x, sample_y, displacement = xx[active], yy[active], disp[active]
        color_sum = np.zeros((np.count_nonzero(active), 3), dtype=src.dtype)
    for row in table:
        scale = 1.0 + np.clip(p.dispersion, 0.0, 1.0) * row[3]
        sx = sample_x + displacement[..., 0] * scale
        sy = sample_y + displacement[..., 1] * scale
        tap = bilinear_sample(src, sx, sy)
        color_sum += tap * row[:3]
    if dense:
        out = color_sum
    else:
        out = src.copy()
        out[active] = color_sum

    light = np.array([math.cos(math.radians(p.light_angle_deg)), math.sin(math.radians(p.light_angle_deg))], dtype=src.dtype)
    ndotl = edge_normal[..., 0] * light[0] + edge_normal[..., 1] * light[1]
    spec = np.clip(ndotl * 0.5 + 0.5, 0, 1) ** 8

    phase = mag_b * 10.0 + anim_time * 1.3 + curl[..., 0] * 0.6
    glint = iridescent_palette(phase) * (spec * mask * p.iridescence)[..., None]

    out = np.clip(out + glint, 0.0, 1.0)
    return out


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(arr, 0.0, 1.0)
    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mode", choices=["composite", "mask", "flow"], default="composite")
    ap.add_argument("--strength", type=float, default=Params.strength)
    ap.add_argument("--turbulence", type=float, default=Params.turbulence)
    ap.add_argument("--radius", type=float, default=Params.radius)
    ap.add_argument("--dispersion", type=float, default=Params.dispersion)
    ap.add_argument("--dispersion-steps", type=int, default=Params.dispersion_steps)
    ap.add_argument("--fast-sampling", action="store_true")
    ap.add_argument("--iridescence", type=float, default=Params.iridescence)
    ap.add_argument("--seed", type=float, default=Params.seed)
    ap.add_argument("--time", type=float, default=Params.time)
    ap.add_argument("--phase-offset", type=float, default=Params.phase_offset)
    args = ap.parse_args()

    src = load_image(args.input)
    params = Params(
        strength=args.strength, turbulence=args.turbulence, radius=args.radius,
        dispersion=args.dispersion, dispersion_steps=args.dispersion_steps,
        iridescence=args.iridescence, seed=args.seed, time=args.time,
        phase_offset=args.phase_offset, fast_sampling=args.fast_sampling,
    )
    out = render(src, params, output_mode=args.mode)
    save_image(out, args.output)
    print(f"wrote {args.output} ({out.shape[1]}x{out.shape[0]}, mode={args.mode})")


if __name__ == "__main__":
    main()
