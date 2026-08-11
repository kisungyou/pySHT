# Citation

pySHT does not currently have an archived DOI. Cite the released software
version so the computation can be reconstructed; adding the exact source
revision is useful when an analysis depends on unreleased changes.

## Suggested citation

> You, K. (2026). *pySHT: Validated statistical hypothesis tests for scientific
> Python* (version 0.1.0) [Computer software].
> <https://kisungyou.com/pysht/>.

Add the Git commit identifier and access date when using an unreleased source
revision. If a future release provides an archive DOI, prefer that
release-specific citation.

## BibTeX

```bibtex
@software{you_pysht_2026,
  author  = {You, Kisung},
  title   = {{pySHT}: Validated Statistical Hypothesis Tests for Scientific Python},
  year    = {2026},
  version = {0.1.0},
  url     = {https://kisungyou.com/pysht/},
  note    = {Software release}
}
```

## Cite the statistical method too

Citing pySHT documents the implementation, but it does not replace the
primary statistical reference. Cite the paper or book associated with each
procedure used. Method-specific references appear in the API docstrings and
[validation ledgers](../validation/index.md).

## Relationship to SHT for R

The R package SHT, authored by Kyoungjae Lee, Lizhen Lin, and Kisung You,
provided the original method catalog. pySHT is a new, independently audited
implementation. If an analysis used both packages, cite both and report which
package and version produced each result. See the
[migration guide](../migration/from-r.md) for the current correspondence.
