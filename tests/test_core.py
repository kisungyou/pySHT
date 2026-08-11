from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from pysht import __version__, _core


class NativeCoreTests(unittest.TestCase):
    def test_build_info_matches_package_version(self) -> None:
        info = _core.build_info()

        self.assertEqual(info["version"], __version__)
        self.assertEqual(info["cpp_standard"], 17)
        self.assertIs(info["stable_abi"], True)

    def test_static_dependency_notices_are_release_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notices = (root / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2022 Wenzel Jakob", notices)
        self.assertIn("Copyright (c) 2017 Thibaut Goetghebuer-Planchon", notices)

        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("THIRD_PARTY_LICENSES.md", project["project"]["license-files"])
        self.assertIn(
            "THIRD_PARTY_LICENSES.md",
            project["tool"]["scikit-build"]["sdist"]["include"],
        )

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

    def test_pairwise_distances_match_independent_scipy_oracle(self) -> None:
        generator = np.random.default_rng(20260810)

        for exponent in (-300, -200, -100, 0, 100, 200, 300):
            scale = 10.0**exponent
            x = np.ascontiguousarray(generator.normal(size=(5, 4)) * scale)
            y = np.ascontiguousarray(generator.normal(size=(6, 4)) * scale)

            actual = _core.pairwise_distances(x, y) / scale
            expected = cdist(x / scale, y / scale)

            np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-15)

    def test_distance_output_shape_and_lifetime(self) -> None:
        x = np.empty((0, 3), dtype=np.float64)
        y = np.ones((4, 3), dtype=np.float64)

        empty = _core.pairwise_distances(x, y)
        self.assertEqual(empty.shape, (0, 4))
        self.assertTrue(empty.flags.c_contiguous)

        source = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        owned = _core.pairwise_distances(source, source)
        del source

        np.testing.assert_array_equal(owned, [[0.0, 5.0], [5.0, 0.0]])

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

        with self.assertRaises(TypeError):
            _core.pairwise_distances(np.ones(2, dtype=np.float64), x)

        with self.assertRaises(TypeError):
            _core.pairwise_distances(np.asfortranarray(x), x)

        empty_features = np.empty((2, 0), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            _core.pairwise_distances(empty_features, empty_features)

        extreme = np.array([[1.0e200]], dtype=np.float64)
        with self.assertRaisesRegex(OverflowError, "overflowed"):
            _core.pairwise_squared_distances(extreme, -extreme)


if __name__ == "__main__":
    unittest.main()
