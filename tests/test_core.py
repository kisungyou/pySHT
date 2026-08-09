from __future__ import annotations

import unittest

import numpy as np

from pysht import __version__, _core


class NativeCoreTests(unittest.TestCase):
    def test_build_info_matches_package_version(self) -> None:
        info = _core.build_info()

        self.assertEqual(info["version"], __version__)
        self.assertEqual(info["cpp_standard"], 17)
        self.assertIs(info["stable_abi"], True)

    def test_pairwise_squared_distances(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        y = np.array([[1.0, 0.0], [2.0, 2.0]], dtype=np.float64)

        actual = _core.pairwise_squared_distances(x, y)
        expected = np.array([[1.0, 8.0], [4.0, 1.0]])

        np.testing.assert_allclose(actual, expected)

    def test_pairwise_distances_resist_avoidable_square_overflow(self) -> None:
        x = np.array([[1.0e200, 0.0]], dtype=np.float64)
        y = np.array([[-1.0e200, 0.0]], dtype=np.float64)

        actual = _core.pairwise_distances(x, y)

        np.testing.assert_allclose(actual, np.array([[2.0e200]]))

    def test_rejects_invalid_native_inputs(self) -> None:
        x = np.ones((2, 2), dtype=np.float64)
        wrong_features = np.ones((2, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            _core.pairwise_squared_distances(x, wrong_features)

        nonfinite = x.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            _core.pairwise_squared_distances(nonfinite, x)

        with self.assertRaises(TypeError):
            _core.pairwise_squared_distances(x.astype(np.float32), x)


if __name__ == "__main__":
    unittest.main()
