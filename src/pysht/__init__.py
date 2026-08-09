"""pySHT: validated statistical hypothesis tests for Python.

The public test functions live in domain-specific modules.  For example::

    from pysht.equaldist import biswas_ghosh_2samp

Result classes are re-exported here for convenient type checks.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._results import DistanceTestResult, HypothesisTestResult, ResamplingTestResult

try:
    __version__ = version("pysht")
except PackageNotFoundError:  # Source-tree imports before installation.
    __version__ = "0.1.0.dev0"

__all__ = [
    "DistanceTestResult",
    "HypothesisTestResult",
    "ResamplingTestResult",
    "__version__",
]
