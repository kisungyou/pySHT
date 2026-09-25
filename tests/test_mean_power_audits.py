from __future__ import annotations

from dataclasses import replace

from pysht import mean
from tools.mean_power_audits import (
    FIRST_SCENARIO_SEED,
    _scenario_registry,
    _select_scenarios,
    run_scenario,
)


def test_power_registry_covers_every_public_mean_routine() -> None:
    expected = {
        f"mean.{name}"
        for name in mean.__all__
        if name != "maximum_pairwise_bayes_factor_2samp"
    }
    expected.add("equaldist.bg_2samp")
    expected.update(
        {
            "covariance.czz_identity_1samp",
            "covariance.czz_sphericity_1samp",
        }
    )
    registered = {scenario.method for scenario in _scenario_registry()}

    assert registered == expected
    assert "mean.clx_2samp" not in registered
    assert "mean._clx_2samp" not in registered


def test_filtered_scenario_retains_its_documented_seed() -> None:
    [(index, scenario)] = _select_scenarios(["mean.thulin_2samp"])

    assert FIRST_SCENARIO_SEED + index == 2_026_090_320
    assert scenario.replications == 300


def test_native_scenarios_replay_the_ledger_streams() -> None:
    expected = {
        "mean.cq_2samp": 20260941,
        "mean.li_1samp": 20260942,
        "mean.li_2samp": 20260943,
        "mean.li_ksamp": 20260944,
        "covariance.czz_identity_1samp": 20260945,
        "covariance.czz_sphericity_1samp": 20260946,
    }
    scenarios = {
        scenario.method: scenario
        for scenario in _scenario_registry()
        if scenario.method in expected
    }
    assert {method: scenario.seed for method, scenario in scenarios.items()} == expected
    assert all(scenario.replications == 2_000 for scenario in scenarios.values())


def test_small_power_audit_is_exactly_replayable() -> None:
    scenario = replace(_scenario_registry()[0], replications=3)

    assert run_scenario(scenario, 0) == run_scenario(scenario, 0)
