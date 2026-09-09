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
  4. A divergence-free curl-noise field (rotated-octave value-noise fBm),
     evaluated at time*flow_speed + phase_offset, is blended with the local
     edge normal by "turbulence" to get a flow direction; scaled by
     strength * mask gives the displacement vector.
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

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64) / 16.0
_SCHARR_Y = _SCHARR_X.T


def scharr_gradient(luma: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(round(step)))
    padded = np.pad(luma, step, mode="edge")
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    h, w = luma.shape
    for j in range(3):
        for i in range(3):
            tap = padded[j * step:j * step + h, i * step:i * step + w]
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
    hi = np.take(csum, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
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
# Curl noise: divergence-free flow from a rotated-octave value-noise fBm
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


_OCTAVE_ROT = np.array([[0.8, 0.6], [-0.6, 0.8]])


def fbm(p: np.ndarray, octaves: int) -> np.ndarray:
    value = np.zeros(p.shape[:-1])
    amp, freq = 0.5, 1.0
    pp = p
    for _ in range(max(1, octaves)):
        value = value + amp * _value_noise(pp * freq)
        pp = pp @ _OCTAVE_ROT.T
        freq *= 2.03
        amp *= 0.5
    return value


def curl_noise(p: np.ndarray, octaves: int, eps: float = 0.5) -> np.ndarray:
    # v = (d psi/dy, -d psi/dx) for a scalar potential psi = fbm(p): the
    # *mixed* partials cancel in the divergence (d2psi/dydx == d2psi/dxdy),
    # which is what makes this field swirl instead of pooling or draining.
    # Pairing same-axis derivatives instead (an easy mistake) gives the
    # Laplacian difference, not a divergence-free field.
    dpsi_dy = (fbm(p + np.array([0.0, eps]), octaves) - fbm(p - np.array([0.0, eps]), octaves)) / (2 * eps)
    dpsi_dx = (fbm(p + np.array([eps, 0.0]), octaves) - fbm(p - np.array([eps, 0.0]), octaves)) / (2 * eps)
    return np.stack([dpsi_dy, -dpsi_dx], axis=-1)


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
    noise_scale: float = 220.0        # px per noise unit (lower = larger swirls)
    flow_speed: float = 1.0           # animation rate multiplier
    dispersion: float = 0.35          # 0-1, sweep spread around scale 1.0
    dispersion_steps: int = 16        # >=3, taps swept across the spread; 3 == classic R/G/B split
    iridescence: float = 0.55         # 0-1, glint colour strength
    light_angle_deg: float = 55.0
    seed: float = 0.0
    time: float = 0.6                 # seconds; still-frame phase for the preview
    phase_offset: float = 0.0         # same units as time*flow_speed; shifts the swirl's start point


def render(src: np.ndarray, p: Params, output_mode: str = "composite") -> np.ndarray:
    """src: HxWx3 float32 in [0,1] straight sRGB. Returns HxWx3 float32 sRGB."""
    h, w = src.shape[:2]
    linear = srgb_to_linear(src)
    luma = oklab_l(linear)

    gx, gy = scharr_gradient(luma, p.detection_scale)
    mag = np.hypot(gx, gy)

    gx_b = gaussian_like_blur(gx, p.radius)
    gy_b = gaussian_like_blur(gy, p.radius)
    mag_b = gaussian_like_blur(mag, p.radius)

    cutoff = (p.threshold / 255.0) * 0.12
    softness = 0.05
    mask = np.clip((mag_b - cutoff) / max(1e-6, softness), 0.0, 1.0)
    mask = mask ** (1.0 / max(0.1, p.contrast))

    inv_mag = np.where(mag_b > 1e-6, 1.0 / mag_b, 0.0)
    edge_normal = np.stack([gx_b * inv_mag, gy_b * inv_mag], axis=-1)

    # phase_offset is added in the same units as time*flow_speed, *before*
    # the 0.35/1.3 rate scaling below, so it shifts the swirl's starting
    # point by a fixed amount regardless of flow_speed -- unlike seed (a
    # spatial coordinate offset, a different pattern entirely), this only
    # moves *where in its cycle* the same pattern currently is.
    anim_time = p.time * p.flow_speed + p.phase_offset
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coord = np.stack([xx, yy], axis=-1) / p.noise_scale
    coord = coord + p.seed * 17.0
    coord[..., 0] += anim_time * 0.35
    curl = curl_noise(coord, p.turbulence_detail)
    curl_norm = curl / np.maximum(1e-6, np.hypot(curl[..., 0], curl[..., 1]))[..., None]

    flow = edge_normal * (1 - p.turbulence) + curl_norm * p.turbulence
    flow_len = np.maximum(1e-6, np.hypot(flow[..., 0], flow[..., 1]))
    flow = flow / flow_len[..., None]

    disp = flow * (p.strength * mask)[..., None]

    if output_mode == "mask":
        gray = mask
        return np.stack([gray, gray, gray], axis=-1)
    if output_mode == "flow":
        vis = np.stack([
            0.5 + 0.5 * flow[..., 0] * mask,
            0.5 + 0.5 * flow[..., 1] * mask,
            mask,
        ], axis=-1)
        return np.clip(vis, 0, 1)

    # Sweep dispersion_steps taps from scale (1-dispersion) to (1+dispersion)
    # along the displacement vector, each weighted by an approximate
    # spectral response, and blend them into a smoothly graded prism
    # fringe. steps=3 reduces to the classic R/G/B split; more steps trade
    # compute for a smoother, more continuous spectrum.
    steps = max(3, int(round(p.dispersion_steps)))
    color_sum = np.zeros(src.shape[:2] + (3,))
    weight_sum = np.zeros(3)
    for s in range(steps):
        t = s / (steps - 1)
        scale = 1.0 + p.dispersion * (2.0 * t - 1.0)
        sx = xx + disp[..., 0] * scale
        sy = yy + disp[..., 1] * scale
        tap = bilinear_sample(src, sx, sy)
        w = spectrum_weight(t)
        color_sum += tap * w
        weight_sum += w
    out = color_sum / np.maximum(weight_sum, 1e-5)

    light = np.array([math.cos(math.radians(p.light_angle_deg)), math.sin(math.radians(p.light_angle_deg))])
    ndotl = edge_normal[..., 0] * light[0] + edge_normal[..., 1] * light[1]
    spec = np.clip(ndotl * 0.5 + 0.5, 0, 1) ** 8

    phase = mag_b * 10.0 + anim_time * 1.3 + curl[..., 0] * 0.6
    glint = iridescent_palette(phase) * (spec * mask * p.iridescence)[..., None]

    out = np.clip(out + glint, 0.0, 1.0)
    return out


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float64) / 255.0


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
        phase_offset=args.phase_offset,
    )
    out = render(src, params, output_mode=args.mode)
    save_image(out, args.output)
    print(f"wrote {args.output} ({out.shape[1]}x{out.shape[0]}, mode={args.mode})")


if __name__ == "__main__":
    main()
