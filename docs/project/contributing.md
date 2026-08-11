# Contributing

Contributions are welcome, especially independent formula checks, numerical
stress cases, documentation corrections, and carefully scoped statistical
procedures. pySHT is pre-1.0, so discuss large API or method additions in an
[issue](https://github.com/kisungyou/pySHT/issues) before investing in a full
implementation.

## Development setup

The project requires Python 3.12 or newer. Editable source builds also require
a C++17 compiler and CMake.

```console
git clone https://github.com/kisungyou/pySHT.git
cd pySHT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run the checks

```console
python -m pytest
ruff check .
ruff format --check .
mypy src/pysht tools
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Use a focused test while iterating, then run the full suite before submitting
a change. Documentation warnings are treated as errors.

## Statistical contribution standard

A new or changed test should include all of the following:

- a precise null and alternative, data layout, assumptions, and sample-size
  constraints;
- a formula ledger that records every denominator, degree of freedom, tail,
  correction, and finite-sample convention;
- a production implementation that fails explicitly for undefined inputs;
- an independent oracle that does not call the private helpers or native
  kernel being tested;
- fixed hand-checkable examples or agreement with a trusted implementation;
- metamorphic checks such as row/group order, translation, common scaling,
  rotation, or feature permutation when mathematically applicable;
- degenerate, non-finite, wrong-shape, and extreme-magnitude cases;
- null-calibration or simulation evidence when a reference distribution is
  approximate; and
- public API, result-field, and validation documentation.

Output from SHT for R can be a regression fixture, but it cannot be the sole
oracle. If production and legacy results disagree, resolve the formulas and
calibration from primary sources before changing either implementation.

## Code and API conventions

- Put public tests in the domain module that matches the hypothesis.
- Use descriptive Python names rather than encoding a publication year.
- Return an immutable result class; do not print or return an unstructured
  dictionary.
- Validate public inputs before numerical work. Do not silently drop missing
  values, coerce Boolean data, or substitute a different calibration.
- Preserve float64 accuracy in native and Python paths. Document any
  normalization that changes the numerical representation of a statistic.
- Add the public name to `__all__`, type it strictly, and document keyword
  semantics.
- Keep random state explicit and report resampling diagnostics.

Native C++ changes require tests through the public Python boundary as well as
tests of any directly exposed kernel. Avoid moving a formula into native code
until its reference implementation is clear enough to audit.

## Documentation contributions

Use MyST Markdown in `docs/`, relative links for project pages, NumPy-style
docstrings for public Python objects, and math notation for formulas. State
whether a method is implemented, experimental, approximate, or blocked. Do
not describe roadmap items as available functionality.

## Submitting a change

Keep each pull request focused. In its description, identify:

- the problem and scientific scope;
- primary references and any trusted comparator;
- formulas or behavior that changed;
- validation evidence and commands run; and
- compatibility or numerical consequences.

By contributing, you agree that your contribution is distributed under the
project's [MIT license](https://github.com/kisungyou/pySHT/blob/master/LICENSE).

Maintainers preparing a publication should follow the dedicated
[release runbook](releasing.md).
