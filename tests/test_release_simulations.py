"""Tests for the opt-in release-simulation runner's cheap control layer."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO

import numpy as np

from pysht import normality
from tools import covariance_release_audits
from tools import release_simulations as simulations


class ReleaseSimulationTests(unittest.TestCase):
    def test_release_tolerance_contract(self) -> None:
        self.assertAlmostEqual(
            simulations.release_tolerance(0.05, 20_000),
            4.0 * (0.05 * 0.95 / 20_000) ** 0.5,
        )
        with self.assertRaisesRegex(TypeError, "integer"):
            simulations.release_tolerance(0.05, True)
        with self.assertRaisesRegex(ValueError, "positive"):
            simulations.release_tolerance(0.05, 0)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            simulations.release_tolerance(1.0, 20_000)

    def test_small_run_is_deterministic_and_does_not_claim_table_replay(self) -> None:
        scenario = simulations.SCENARIOS["variance.bartlett.g3-n100"]
        first = simulations.run_scenario(scenario, seed=20260823, replications=12)
        second = simulations.run_scenario(scenario, seed=20260823, replications=12)
        self.assertEqual(first, second)
        self.assertEqual(first.replications, 12)
        self.assertIsNone(first.documented_rates)
        self.assertIsNone(first.reproduces_documented_counts)
        self.assertEqual(sum(first.gate_passes), 3)

    def test_seed_and_replication_validation(self) -> None:
        scenario = simulations.SCENARIOS["normality.shapiro-wilk.n20"]
        with self.assertRaisesRegex(TypeError, "seed"):
            simulations.run_scenario(scenario, seed=True, replications=1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            simulations.run_scenario(scenario, seed=-1, replications=1)
        with self.assertRaisesRegex(TypeError, "replications"):
            simulations.run_scenario(scenario, seed=1, replications=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            simulations.run_scenario(scenario, seed=1, replications=0)

    def test_normality_stream_prelude_is_part_of_scenario(self) -> None:
        scenario = simulations.SCENARIOS["normality.shapiro-wilk.n100"]
        result = simulations.run_scenario(scenario, seed=7, replications=5)

        generator = np.random.default_rng(7)
        generator.standard_normal((5, 20))
        levels = np.asarray(scenario.levels)
        counts = np.zeros(3, dtype=np.int64)
        for _ in range(5):
            pvalue = normality.shapiro_wilk(generator.standard_normal(100)).pvalue
            counts += pvalue < levels
        self.assertEqual(result.rejection_counts, tuple(int(value) for value in counts))

    def test_cli_lists_scenarios_and_emits_machine_readable_metadata(self) -> None:
        listing = StringIO()
        with redirect_stdout(listing):
            status = simulations.main(["--list"])
        self.assertEqual(status, 0)
        self.assertIn("variance.bartlett.g3-n100", listing.getvalue())

        output = StringIO()
        with redirect_stdout(output):
            status = simulations.main(
                [
                    "normality.shapiro-wilk.n20",
                    "--seed",
                    "7",
                    "--replications",
                    "3",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["scenario"]["ledger"], "docs/validation/normality.md")
        self.assertEqual(payload["results"][0]["seed"], 7)
        self.assertEqual(payload["environment"]["pysht"], version("pysht"))

    def test_covariance_audit_stream_is_small_run_replayable(self) -> None:
        scenario = covariance_release_audits.SCENARIOS[
            "covariance.wl-1samp.n100-p30-m25"
        ]
        first = covariance_release_audits.run_scenario(scenario, seed=7, replications=4)
        second = covariance_release_audits.run_scenario(
            scenario, seed=7, replications=4
        )
        self.assertEqual(first, second)
        self.assertEqual(first.replications, 4)
        self.assertEqual(len(first.rejection_counts), 3)


if __name__ == "__main__":
    unittest.main()
