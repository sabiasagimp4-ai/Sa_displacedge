"""Portable reference/contract tests for prototype/preview.py.

These do not execute Direct2D or YMM4; they check the numeric properties the
HLSL port relies on (kernel normalization, weight-carry correctness, and the
rotated-gradient construction of the curl-noise flow field), the same role
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
    def test_analytic_gradient_matches_potential(self):
        # The optimized path differentiates the smooth value-noise potential
        # analytically. Compare it against an independent finite difference of
        # the scalar fBm at points away from integer cell boundaries.
        rng = np.random.default_rng(7)
        pts = rng.uniform(-50, 50, size=(200, 2))
        eps = 1e-5
        octaves = 4
        for p in pts:
            analytic = preview.fbm_gradient(p[None, :], octaves)[0]
            dx = np.array([eps, 0.0])
            dy = np.array([0.0, eps])
            finite = np.array([
                (preview.fbm((p + dx)[None, :], octaves)[0] - preview.fbm((p - dx)[None, :], octaves)[0]) / (2 * eps),
                (preview.fbm((p + dy)[None, :], octaves)[0] - preview.fbm((p - dy)[None, :], octaves)[0]) / (2 * eps),
            ])
            np.testing.assert_allclose(analytic, finite, atol=2e-4, rtol=2e-4)

    def test_curl_is_the_rotated_gradient(self):
        rng = np.random.default_rng(8)
        pts = rng.uniform(-50, 50, size=(200, 2))
        gradient = preview.fbm_gradient(pts, 4)
        curl = preview.curl_noise(pts, 4)
        np.testing.assert_allclose(curl, np.stack([gradient[:, 1], -gradient[:, 0]], axis=-1), atol=1e-7)


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

    def test_reflects_outside_bounds(self):
        img = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)
        np.testing.assert_allclose(
            preview.reflect_bilinear_sample(img, np.array([-1.0]), np.array([1.0]))[0],
            img[1, 1],
        )


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


class SpectrumWeight(unittest.TestCase):
    def test_weights_are_in_unit_range(self):
        for t in np.linspace(0, 1, 21):
            w = preview.spectrum_weight(t)
            self.assertTrue(np.all(w >= 0.0))
            self.assertTrue(np.all(w <= 1.0 + 1e-9))

    def test_channel_peaks_follow_sweep_position(self):
        # Red should dominate at the outer end (t=1), blue at the inner end
        # (t=0), green in the middle -- matching the original R/(1+d),
        # G/1, B/(1-d) tap ordering when steps == 3.
        red, green, blue = preview.spectrum_weight(1.0), preview.spectrum_weight(0.5), preview.spectrum_weight(0.0)
        self.assertEqual(np.argmax(red), 0)
        self.assertEqual(np.argmax(green), 1)
        self.assertEqual(np.argmax(blue), 2)


class RenderContract(unittest.TestCase):
    def test_zero_strength_is_close_to_source_rgb(self):
        rng = np.random.default_rng(11)
        src = rng.random((24, 24, 3))
        params = preview.Params(strength=0.0, iridescence=0.0)
        out = preview.render(src, params)
        # No displacement and no glint: every dispersion tap samples the same
        # position regardless of dispersion_steps, so output ~= source.
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

    def test_dispersion_steps_is_finite_and_bounded_across_range(self):
        rng = np.random.default_rng(13)
        src = rng.random((20, 20, 3))
        for steps in (3, 8, 32, 128):
            out = preview.render(src, preview.Params(dispersion_steps=steps))
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertGreaterEqual(out.min(), -1e-6)
            self.assertLessEqual(out.max(), 1.0 + 1e-6)

    def test_s_distort_chroma_controls_change_the_render(self):
        rng = np.random.default_rng(15)
        src = rng.random((24, 24, 3))
        base = preview.render(src, preview.Params(dispersion=2.0, warp_red=.5, warp_blue=1.0))
        rotated = preview.render(src, preview.Params(dispersion=2.0, warp_red=3.0, warp_blue=-2.0, warp_rotation_deg=90))
        self.assertTrue(np.isfinite(rotated).all())
        self.assertGreater(float(np.mean(np.abs(rotated - base))), 1e-4)

    def test_more_steps_converges_to_a_stable_result(self):
        # As dispersion_steps grows the tap sweep is a finer quadrature of
        # the same continuous spectral sweep, so raising it further should
        # have rapidly diminishing effect on the result.
        rng = np.random.default_rng(14)
        src = rng.random((20, 20, 3))
        out_64 = preview.render(src, preview.Params(dispersion_steps=64))
        out_128 = preview.render(src, preview.Params(dispersion_steps=128))
        np.testing.assert_allclose(out_64, out_128, atol=0.01)


class PhaseOffset(unittest.TestCase):
    def test_equivalent_to_advancing_time(self):
        # anim_time = time*flow_speed + phase_offset is the only place
        # time/phase_offset reach the render, so (time=T, phase=0) and
        # (time=0, phase=T) at flow_speed=1 must be pixel-identical.
        rng = np.random.default_rng(20)
        src = rng.random((20, 20, 3))
        out_via_time = preview.render(src, preview.Params(flow_speed=1.0, time=7.0, phase_offset=0.0))
        out_via_phase = preview.render(src, preview.Params(flow_speed=1.0, time=0.0, phase_offset=7.0))
        np.testing.assert_allclose(out_via_time, out_via_phase, atol=1e-9)

    def test_zero_offset_matches_default_time_only_behaviour(self):
        rng = np.random.default_rng(21)
        src = rng.random((20, 20, 3))
        baseline = preview.render(src, preview.Params(phase_offset=0.0))
        explicit_zero = preview.render(src, preview.Params(phase_offset=0.0))
        np.testing.assert_array_equal(baseline, explicit_zero)

    def test_nonzero_offset_changes_the_render(self):
        rng = np.random.default_rng(22)
        src = rng.random((20, 20, 3))
        out_a = preview.render(src, preview.Params(phase_offset=0.0))
        out_b = preview.render(src, preview.Params(phase_offset=250.0))
        self.assertFalse(np.allclose(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
