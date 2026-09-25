# Citation

pySHT does not currently have an archived DOI. Cite the exact software version
used so the computation can be reconstructed. For an unpublished candidate or
other unreleased source, also include the Git commit identifier and access
date. The examples below describe candidate 0.5.0rc1;
replace the version when citing a different installation.

## Suggested citation

> You, K. (2026). *pySHT: Validated statistical hypothesis tests for scientific
> Python* (version 0.5.0rc1, release candidate) [Computer software].
> <https://www.kisungyou.com/pysht/>.

If a future release provides an archive DOI, prefer that release-specific
citation.

## BibTeX

```bibtex
@software{you_pysht_2026,
  author  = {You, Kisung},
  title   = {{pySHT}: Validated Statistical Hypothesis Tests for Scientific Python},
  year    = {2026},
  version = {0.5.0rc1},
  url     = {https://www.kisungyou.com/pysht/},
  note    = {Release candidate}
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
