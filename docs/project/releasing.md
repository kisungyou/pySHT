# Releasing pySHT

This page is the maintainer runbook for publishing pySHT. Production releases
use GitHub Actions and PyPI Trusted Publishing. No PyPI password or API token
is stored in the repository or in GitHub.

## One-time setup

### Protect the GitHub environment

In the repository's **Settings > Environments**, create an environment named
`pypi`.

- Require manual approval before deployment.
- Under **Deployment branches and tags**, choose **Selected branches and tags**
  and add a tag rule matching `v*`.
- Protect release tags and changes to `.github/workflows/publish.yml` through
  the repository ruleset.

If the repository has only one maintainer, do not enable **Prevent self-review**
until another reviewer is available; otherwise no one can approve a release.

### Register the PyPI publisher

Sign in to [PyPI](https://pypi.org/) with a verified email address and two-factor
authentication. For the existing `pysht` project, confirm that its Trusted
Publisher configuration uses the following exact values:

| Field | Value |
|---|---|
| PyPI project name | `pysht` |
| Owner | `kisungyou` |
| Repository | `pySHT` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

For a new project, use the
[pending-publisher page](https://pypi.org/manage/account/publishing/) instead.
A pending publisher creates the project during the first successful upload;
it does not reserve the name beforehand. The workflow filename, repository
capitalization, and environment must match exactly.

## Release version policy

The static version in `pyproject.toml` is the release source of truth. A release
tag must be exactly `v<version>`; for example, version `0.1.0` uses tag
`v0.1.0`. Mark GitHub releases for alpha, beta, release-candidate, or
development versions as pre-releases. A final release such as `0.1.0` must not
be marked as a GitHub pre-release.

The workflow accepts compact PEP 440 release forms, including `aN`, `bN`,
`rcN`, `.postN`, and `.devN`. Uploaded PyPI filenames cannot be reused, and
pySHT treats published releases as immutable. A correction after `0.1.0`
therefore requires a new version such as `0.1.1`.

## Prepare the release

Use a focused release pull request. Before merging it:

1. Set the intended version in `pyproject.toml`.
2. Move the relevant changelog entries from **Unreleased** to a dated version.
3. Update the citation, roadmap, installation wording, and other text that
   describes the current release state.
4. Run the complete local checks documented in
   [Contributing](contributing.md), then require the GitHub CI workflow to pass.
5. Review the public API inventory and validation status one final time.

For subsequent development, use the next intended development version rather
than reusing a published version.

## Publish

After the release pull request is merged into `master`:

1. Create a draft GitHub Release targeting the exact commit on `master`.
2. Set its tag to `v<version>` and review the generated release notes.
3. Select **Set as a pre-release** when required by the version.
4. Publish the GitHub Release.
5. After all builds and smoke tests pass, review and approve the protected
   `pypi` environment deployment.

Publishing the GitHub Release starts `.github/workflows/publish.yml`. The
workflow:

- verifies the project name, version, release tag, pre-release setting, and
  ancestry on `master`;
- reruns tests, typing, linting, formatting, and the strict documentation build;
- reruns the complete test suite against the declared minimum NumPy and SciPy
  versions;
- builds one source archive and CPython stable-ABI wheels for Linux x86-64,
  macOS Apple Silicon, macOS Intel, and Windows AMD64;
- smoke-tests every wheel on Python 3.12, 3.13, and 3.14;
- rejects incomplete, duplicate, mismatched, or unexpected artifacts and
  records their SHA-256 hashes; and
- gives only the final isolated publishing job permission to request a
  short-lived PyPI credential.

The publishing job does not check out or execute repository code. Trusted
Publishing also creates PyPI attestations for the uploaded files. Duplicate
uploads fail rather than being silently skipped.

## Verify the release

After the workflow succeeds, confirm the
[PyPI project page](https://pypi.org/project/pysht/) and install the exact
version in a clean environment:

```console
python -m venv /tmp/pysht-release-check
source /tmp/pysht-release-check/bin/activate
python -m pip install "pysht==0.1.0"
python -c "import pysht; print(pysht.__version__)"
```

Use the actual released version in place of `0.1.0`. Check the release on at
least one supported Python version not used to build the wheel.

## TestPyPI

TestPyPI is optional. Use it before the first production release or after a
substantial packaging-workflow change, not as a routine production gate. It
requires a separate account, publisher registration, GitHub environment, and
manual publishing job. The production workflow intentionally contains no
TestPyPI branch; add and review that separate path before registering its
publisher. Do not use TestPyPI to resolve NumPy or SciPy dependencies; those
projects may not have corresponding builds there.
