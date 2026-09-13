import unittest
import numpy as np
from prototype.preview import spectral_table, spectrum_weight, render, Params

class Sampling(unittest.TestCase):
    def test_mass_and_first_moment_all_counts(self):
        for steps in range(3, 129):
            exact, fast = spectral_table(steps), spectral_table(steps, True)
            weights = np.array([spectrum_weight(i / (steps - 1)) for i in range(steps)])
            np.testing.assert_allclose(exact[:, :3], weights / weights.sum(axis=0), atol=1e-14)
            np.testing.assert_allclose(fast[:, :3].sum(axis=0), 1, atol=1e-14)
            np.testing.assert_allclose(fast[:, :3].T @ fast[:, 3], exact[:, :3].T @ exact[:, 3], atol=1e-14)
            self.assertTrue(np.all(fast[:, :3] >= 0))
            self.assertEqual(len(fast), min(steps, 8))

    def test_small_counts_unchanged(self):
        for steps in range(3, 9):
            np.testing.assert_allclose(spectral_table(steps), spectral_table(steps, True), atol=1e-14)

    def test_fast_render_bounded_and_deterministic(self):
        src = np.random.default_rng(8).random((24, 32, 3), dtype=np.float32)
        p = Params(fast_sampling=True, dispersion_steps=128)
        a, b = render(src, p), render(src, p)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a >= 0).all() and (a <= 1).all())
        np.testing.assert_array_equal(a, b)

    def test_high_dispersion_is_supported(self):
        src = np.random.default_rng(9).random((32, 40, 3), dtype=np.float32)
        low = render(src, Params(strength=6, dispersion=.35, dispersion_steps=24))
        high = render(src, Params(strength=6, dispersion=8.0, dispersion_steps=24))
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue((high >= 0).all() and (high <= 1).all())
        self.assertGreater(float(np.mean(np.abs(high - low))), 1e-4)
