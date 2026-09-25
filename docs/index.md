---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
description: Statistical hypothesis testing for Python.
---

{.document-title}
# pySHT: statistical hypothesis testing

```{image} _static/og.png
:alt: pySHT with two probability distributions and a vertical test threshold
:class: pysht-home-figure
:align: center
```

::::{div} home-introduction
pySHT is a Python library for statistical hypothesis testing. It provides
literature-traceable implementations for mean, variance, covariance,
joint-parameter, distributional, goodness-of-fit, and structured-domain
questions, with a consistent result contract across methods.

The package is an extended and independently audited successor to
[SHT for R](https://github.com/kisungyou/SHT). Start with the
[installation guide](getting-started.md), read the [user guide](user-guide/index.md),
or go directly to the [API reference](api/index.md).
::::

::::{div} home-status
**Development version 0.5.0.dev0.** Version 0.1.0 remains the latest released
package. This tree integrates the validated 0.2--0.5 expansion waves; methods
that did not pass their scientific release gate remain private or absent.
::::

## Contents

- [Getting started](getting-started.md) — install pySHT and run a first test.
- [User guide](user-guide/index.md) — choose a procedure, prepare data, and
  interpret the result.
- [API reference](api/index.md) — inspect every public function and result type.
- [Validation](validation/index.md) — review formulas, numerical checks, and
  independent comparisons.
- [Project](project/index.md) — see the roadmap, migration notes, and
  contribution guidance.

## API categories

- **[1] Univariate Mean** — [API](api/univariate-mean.md)
- **[2] Multivariate Mean** — [API](api/multivariate-mean.md)
- **[3] Variance** — [API](api/variance.md)
- **[4] Covariance** — [API](api/covariance.md)
- **[5] Mean and Variance** — [API](api/mean-variance.md)
- **[6] Mean and Covariance** — [API](api/mean-covariance.md)
- **[7] Equality of Distributions** — [API](api/equaldist.md)
- **[8] Normality** — [API](api/normality.md)
- **[9] Rectangular Uniformity** — [API](api/uniformity.md)
- **[10] Special Domains** — [API](api/simplex.md)
- **[11] Independence** — [API](api/independence.md)
- **[12] Circular Data** — [API](api/circular.md)

Categories `[0]`--`[10]` preserve SHT 0.1.9's numbering and scientific
organization. The development API has 72 canonical functions: 51 SHT
compatibility functions covering 52 of 54 statistical routine identities and
21 independently cataloged pySHT-native methods. The
[migration crosswalk](migration/from-r.md) explains the one shared
implementation, the validation-blocked mean CLX and Fisher identities, and the
two R-only adapters; the [native catalog](methods/native-methods.md) records
the additional research methods and blocked candidates.

```{toctree}
:hidden:
:maxdepth: 2

Getting started <getting-started>
User guide <user-guide/index>
API <api/index>
```
