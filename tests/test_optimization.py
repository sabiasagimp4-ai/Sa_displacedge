import unittest
from unittest.mock import patch
import numpy as np
from prototype import preview as p

class ExactOptimizations(unittest.TestCase):
    def test_translated_scharr_matches_arbitrary_bilinear(self):
        for shape in ((1, 1), (1, 11), (13, 1), (35, 47)):
            image = np.random.default_rng(2).random(shape, dtype=np.float32)
            y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
            for step in (.25, .7, 1, 1.5, 3.123, 32):
                gx, gy = np.zeros_like(image), np.zeros_like(image)
                for j in range(3):
                    for i in range(3):
                        tap = p._sample_scalar(image, x+(i-1)*step, y+(j-1)*step)
                        gx += p._SCHARR_X[j,i]*tap
                        gy += p._SCHARR_Y[j,i]*tap
                a, b = p.scharr_gradient(image, step)
                np.testing.assert_allclose(a, gx, atol=1e-7)
                np.testing.assert_allclose(b, gy, atol=1e-7)

    def test_zero_mask_skips_noise_and_sampling(self):
        src = np.full((32,40,3), .2, np.float32)
        with patch.object(p, 'curl_noise', side_effect=AssertionError('noise called')), \
             patch.object(p, 'bilinear_sample', side_effect=AssertionError('sampling called')):
            np.testing.assert_array_equal(p.render(src,p.Params()),src)
            flow=p.render(src,p.Params(), 'flow')
            np.testing.assert_array_equal(flow[...,:2],.5)
            np.testing.assert_array_equal(flow[...,2],0)

    def test_sparse_render_evaluates_only_active_noise(self):
        src = np.zeros((48,64,3),np.float32)
        src[16:32,24:40] = [.9,.3,.7]
        params=p.Params(radius=0, turbulence_detail=4)
        original = p.curl_noise
        counts=[]
        def inspect(coord, octaves):
            counts.append(coord.shape)
            return original(coord,octaves)
        with patch.object(p,'curl_noise',side_effect=inspect):
            a=p.render(src,params)
        self.assertEqual(len(counts[0]),2)
        self.assertLess(counts[0][0],48*64)
        self.assertTrue(np.isfinite(a).all())
