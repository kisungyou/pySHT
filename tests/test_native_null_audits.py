"""Fast plumbing checks for the opt-in native-method null audits."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from tools.native_null_audits import SCENARIOS, run


def test_native_null_registry_covers_released_research_methods() -> None:
    assert set(SCENARIOS) == {
        "mean.cq_2samp",
        "mean.li_1samp",
        "mean.li_2samp",
        "mean.li_ksamp",
        "mean.li_ksamp.balanced",
        "covariance.czz_identity_1samp",
        "covariance.czz_sphericity_1samp",
    }
    assert SCENARIOS["mean.li_ksamp"].streams == (20260842,)
    assert SCENARIOS["mean.li_ksamp.balanced"].streams == (20260844,)


def test_native_null_stream_is_replayable() -> None:
    scenario = replace(SCENARIOS["mean.li_1samp"], replications=2)
    assert run(scenario, scenario.streams[0]) == run(scenario, scenario.streams[0])


def test_native_null_evidence_can_be_serialized_as_json() -> None:
    scenario = replace(SCENARIOS["mean.li_1samp"], replications=2)
    result = run(scenario, scenario.streams[0])
    payload = json.loads(json.dumps(asdict(result)))
    assert len(payload["rejection_rate_intervals_95"]) == 3
    assert all(type(value) is bool for value in payload["nominal_in_interval_95"])
