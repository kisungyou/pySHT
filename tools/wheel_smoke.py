"""Smoke test the complete public API from an installed release wheel."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import files
from types import ModuleType

import numpy as np

import pysht
from pysht import _core

EXPECTED_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "pysht.mean": (
        "ttest_1samp",
        "ttest_2samp",
        "anova_oneway",
        "hotelling_1samp",
        "hotelling_2samp",
        "dempster_1samp",
        "bs_1samp",
        "sd_1samp",
        "dempster_2samp",
        "yao_2samp",
        "johansen_2samp",
        "nvm_2samp",
        "bs_2samp",
        "cq_2samp",
        "ky_2samp",
        "sd_2samp",
        "li_1samp",
        "li_2samp",
        "li_ksamp",
        "ljw_2samp",
        "thulin_2samp",
        "maximum_pairwise_bayes_factor_2samp",
        "schott_ksamp",
        "zx_ksamp",
        "cph_ksamp",
    ),
    "pysht.variance": (
        "chisquare_1samp",
        "f_2samp",
        "bartlett",
        "levene",
        "brown_forsythe",
    ),
    "pysht.covariance": (
        "czz_identity_1samp",
        "czz_sphericity_1samp",
        "wl_1samp",
        "lc_2samp",
        "clx_2samp",
        "wl_2samp",
        "maximum_pairwise_bayes_factor_2samp",
        "schott_2001_ksamp",
        "schott_2007_ksamp",
    ),
    "pysht.mean_variance": (
        "lrt_1samp",
        "pn_2samp",
        "pl_2samp",
        "muirhead_2samp",
        "exact_lrt_2samp",
        "lrt_2samp",
    ),
    "pysht.mean_covariance": (
        "llzs_1samp",
        "lrt_1samp",
        "hn_2samp",
    ),
    "pysht.equaldist": (
        "bg_2samp",
        "energy_ksamp",
        "mmd_2samp",
    ),
    "pysht.independence": (
        "dhsic",
        "distance_covariance",
        "distance_multivariance",
        "hsic",
    ),
    "pysht.normality": (
        "shapiro_wilk",
        "shapiro_francia",
        "jarque_bera",
        "adjusted_jarque_bera",
        "energy",
        "henze_zirkler",
        "robust_jarque_bera",
    ),
    "pysht.uniformity": ("ehy", "ym_interpoint", "ym_quantile"),
    "pysht.simplex": ("alpha_energy_ksamp", "ehy_uniformity", "uniformity"),
    "pysht.circular": (
        "hermans_rasson",
        "mardia_watson_wheeler_ksamp",
        "rayleigh",
        "watson",
    ),
}


def _load_complete_api() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}
    function_count = 0
    for module_name, expected_names in EXPECTED_FUNCTIONS.items():
        module = import_module(module_name)
        modules[module_name] = module
        assert set(module.__all__) == set(expected_names)
        for function_name in expected_names:
            assert callable(getattr(module, function_name))
        function_count += len(expected_names)

    assert function_count == 72
    assert set(pysht.__all__) == {
        "BayesFactorTestResult",
        "DistanceTestResult",
        "HypothesisTestResult",
        "ResamplingTestResult",
        "StatisticalTestResult",
        "__version__",
    }
    return modules


def main() -> None:
    modules = _load_complete_api()

    installed_files = files("pysht")
    assert installed_files is not None
    normalized_files = {str(path).replace("\\", "/") for path in installed_files}
    assert "pysht/py.typed" in normalized_files
    assert "pysht/_core.pyi" in normalized_files
    assert any(
        path.startswith("pysht/_core.") and path.endswith((".so", ".pyd", ".dylib"))
        for path in normalized_files
    )
    assert any(path.endswith("licenses/LICENSE") for path in normalized_files)
    assert any(
        path.endswith("licenses/THIRD_PARTY_LICENSES.md") for path in normalized_files
    )

    points = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    distances = _core.pairwise_distances(points, points)
    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])

    mean_result = modules["pysht.mean"].ttest_1samp([1.0, 2.0, 3.0], popmean=0.0)
    assert 0.0 <= mean_result.pvalue <= 1.0

    variance_result = modules["pysht.variance"].f_2samp(
        [0.0, 1.0, 3.0], [0.0, 2.0, 5.0, 7.0]
    )
    assert variance_result.df == (2.0, 3.0)

    distance_result = modules["pysht.equaldist"].bg_2samp(
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
    assert build_info["version"] == pysht.__version__
    print(
        f"pysht {pysht.__version__}: installed-wheel smoke test passed for 72 functions"
    )


if __name__ == "__main__":
    main()
