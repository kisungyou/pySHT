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
    expected = {f"mean.{name}" for name in mean.__all__ if name != "lyl_2samp"}
    expected.add("equaldist.bg_2samp")
    registered = {scenario.method for scenario in _scenario_registry()}

    assert registered == expected
    assert "mean.clx_2samp" not in registered
    assert "mean._clx_2samp" not in registered


def test_filtered_scenario_retains_its_documented_seed() -> None:
    [(index, scenario)] = _select_scenarios(["mean.thulin_2samp"])

    assert FIRST_SCENARIO_SEED + index == 2_026_090_320
    assert scenario.replications == 300


def test_small_power_audit_is_exactly_replayable() -> None:
    scenario = replace(_scenario_registry()[0], replications=3)

    assert run_scenario(scenario, 0) == run_scenario(scenario, 0)
