"""Minimal smoke test executed against an installed release wheel."""

from __future__ import annotations

import numpy as np

import pysht
from pysht import _core
from pysht.equaldist import biswas_ghosh_2samp
from pysht.mean import ttest_1samp
from pysht.variance import f_2samp


def main() -> None:
    points = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    distances = _core.pairwise_distances(points, points)
    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])

    mean_result = ttest_1samp([1.0, 2.0, 3.0], popmean=0.0)
    assert 0.0 <= mean_result.pvalue <= 1.0

    variance_result = f_2samp([0.0, 1.0, 3.0], [0.0, 2.0, 5.0, 7.0])
    assert variance_result.df == (2.0, 3.0)

    distance_result = biswas_ghosh_2samp(
        [0.0, 0.5],
        [2.0, 3.0],
        calibration="exact",
        n_resamples=6,
    )
    assert distance_result.calibration == "exact permutation"
    assert distance_result.exact is True
    assert "p-value" in str(distance_result)

    build_info = _core.build_info()
    assert build_info["stable_abi"] is True
    print(f"pysht {pysht.__version__}: installed-wheel smoke test passed")


if __name__ == "__main__":
    main()
