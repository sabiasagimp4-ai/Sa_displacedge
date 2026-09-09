"""Portable reference/contract tests for prototype/preview.py.

These do not execute Direct2D or YMM4; they check the numeric properties the
HLSL port relies on (kernel normalization, weight-carry correctness, and the
divergence-free property of the curl-noise flow field), the same role
Sa_aohue's tests/test_ymm_v02.py plays for its own kernels.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prototype"))
import preview  # noqa: E402


class BoxBlur(unittest.TestCase):
    def test_constant_field_is_unchanged(self):
        arr = np.full((40, 50), 3.25)
        blurred = preview.gaussian_like_blur(arr, sigma=12.0)
        np.testing.assert_allclose(blurred, arr, atol=1e-9)

    def test_zero_sigma_is_identity(self):
        rng = np.random.default_rng(1)
        arr = rng.random((30, 30))
        np.testing.assert_array_equal(preview.gaussian_like_blur(arr, 0.0), arr)

    def test_impulse_sum_is_conserved(self):
        arr = np.zeros((81, 81))
        arr[40, 40] = 1.0
        blurred = preview.gaussian_like_blur(arr, sigma=8.0)
        self.assertAlmostEqual(blurred.sum(), arr.sum(), delta=1e-6)
        self.assertLess(blurred[40, 40], arr[40, 40])
        self.assertGreater(blurred[40, 41], 0)


class ScharrGradient(unittest.TestCase):
    def test_flat_field_has_zero_gradient(self):
        luma = np.full((20, 20), 0.6)
        gx, gy = preview.scharr_gradient(luma, step=1)
        np.testing.assert_allclose(gx, 0, atol=1e-12)
        np.testing.assert_allclose(gy, 0, atol=1e-12)

    def test_horizontal_ramp_has_zero_vertical_gradient(self):
        luma = np.tile(np.linspace(0, 1, 40), (40, 1))
        gx, gy = preview.scharr_gradient(luma, step=1)
        np.testing.assert_allclose(gy[3:-3, 3:-3], 0, atol=1e-9)
        self.assertGreater(np.mean(gx[3:-3, 3:-3]), 0)

    def test_kernel_taps_sum_to_zero(self):
        self.assertAlmostEqual(preview._SCHARR_X.sum(), 0.0)
        self.assertAlmostEqual(preview._SCHARR_Y.sum(), 0.0)


class CurlNoiseIsDivergenceFree(unittest.TestCase):
    def test_discrete_divergence_is_near_zero_at_matching_step(self):
        # curl_noise builds v = (d(fbm)/dy, -d(fbm)/dx) from *central*
        # differences. Central-difference operators along independent axes
        # commute exactly (they are linear combinations of the same grid
        # samples in either order), so probing divergence with the SAME
        # step used internally (eps=0.5) must cancel to float precision.
        # Probing with a *different* outer step re-differentiates a field
        # that already has eps=0.5 baked into it and is not expected to
        # cancel -- that would be testing an unrelated quantity, not this
        # field's divergence.
        rng = np.random.default_rng(7)
        pts = rng.uniform(-50, 50, size=(200, 2))
        eps = 0.5
        octaves = 4

        def curl_at(p):
            return preview.curl_noise(p[None, :], octaves, eps=eps)[0]

        for p in pts:
            vx1 = curl_at(p + np.array([eps, 0]))[0]
            vx0 = curl_at(p - np.array([eps, 0]))[0]
            vy1 = curl_at(p + np.array([0, eps]))[1]
            vy0 = curl_at(p - np.array([0, eps]))[1]
            div = (vx1 - vx0) / (2 * eps) + (vy1 - vy0) / (2 * eps)
            self.assertAlmostEqual(div, 0.0, places=9)


class BilinearSample(unittest.TestCase):
    def test_integer_coordinates_are_exact(self):
        rng = np.random.default_rng(3)
        img = rng.random((16, 16, 3))
        yy, xx = np.mgrid[0:16, 0:16].astype(np.float64)
        sampled = preview.bilinear_sample(img, xx, yy)
        np.testing.assert_allclose(sampled, img, atol=1e-12)

    def test_clamps_outside_bounds(self):
        img = np.zeros((4, 4, 3))
        img[0, 0] = [1, 0, 0]
        sampled = preview.bilinear_sample(img, np.array([-5.0]), np.array([-5.0]))
        np.testing.assert_allclose(sampled[0], [1, 0, 0])


class IridescentPalette(unittest.TestCase):
    def test_output_is_in_unit_range(self):
        t = np.linspace(-10, 10, 500)
        colours = preview.iridescent_palette(t)
        self.assertGreaterEqual(colours.min(), 0.0)
        self.assertLessEqual(colours.max(), 1.0)

    def test_channels_are_phase_shifted(self):
        # t=0 is a degenerate point for phases (0, .33, .67): they are
        # mirror-symmetric about .5, so cos(2pi*.33) == cos(2pi*.67) there.
        # Any non-symmetric t separates all three channels.
        colours = preview.iridescent_palette(np.array([0.15]))[0]
        self.assertFalse(np.allclose(colours[0], colours[1]))
        self.assertFalse(np.allclose(colours[1], colours[2]))


class RenderContract(unittest.TestCase):
    def test_zero_strength_is_close_to_source_rgb(self):
        rng = np.random.default_rng(11)
        src = rng.random((24, 24, 3))
        params = preview.Params(strength=0.0, iridescence=0.0)
        out = preview.render(src, params)
        # No displacement and no glint: only the (disabled) dispersion spread
        # and mask-gated glint remain, both zero here, so output ~= source.
        np.testing.assert_allclose(out, src, atol=1e-6)

    def test_output_modes_are_finite_and_bounded(self):
        rng = np.random.default_rng(12)
        src = rng.random((24, 24, 3))
        params = preview.Params()
        for mode in ("composite", "mask", "flow"):
            out = preview.render(src, params, output_mode=mode)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertGreaterEqual(out.min(), -1e-6)
            self.assertLessEqual(out.max(), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
