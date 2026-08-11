"""Tests for release-artifact validation and assembly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.assemble_release import assemble_release, write_manifest

VERSION = "0.1.0"
VALID_FILENAMES = (
    f"pysht-{VERSION}.tar.gz",
    f"pysht-{VERSION}-cp312-abi3-manylinux_2_17_x86_64.whl",
    f"pysht-{VERSION}-cp312-abi3-macosx_11_0_arm64.whl",
    f"pysht-{VERSION}-cp312-abi3-macosx_11_0_x86_64.whl",
    f"pysht-{VERSION}-cp312-abi3-win_amd64.whl",
)


def _write_artifacts(root: Path, filenames: tuple[str, ...]) -> None:
    for index, filename in enumerate(filenames):
        directory = root / f"artifact-{index}"
        directory.mkdir(parents=True)
        (directory / filename).write_bytes(filename.encode())


class AssembleReleaseTests(unittest.TestCase):
    def test_complete_release_is_copied_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            incoming.mkdir()
            _write_artifacts(incoming, VALID_FILENAMES)

            artifacts = assemble_release(
                incoming,
                root / "dist",
                distribution="pysht",
                version=VERSION,
            )
            manifest = root / "SHA256SUMS"
            write_manifest(artifacts, manifest)

            self.assertEqual({path.name for path in artifacts}, set(VALID_FILENAMES))
            self.assertEqual(len(manifest.read_text().splitlines()), 5)
            for filename in VALID_FILENAMES:
                self.assertIn(filename, manifest.read_text())

    def test_missing_platform_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            incoming.mkdir()
            _write_artifacts(incoming, VALID_FILENAMES[:-1])

            with self.assertRaisesRegex(ValueError, "exactly four wheels"):
                assemble_release(
                    incoming,
                    root / "dist",
                    distribution="pysht",
                    version=VERSION,
                )

    def test_duplicate_basename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            incoming.mkdir()
            _write_artifacts(incoming, VALID_FILENAMES)
            duplicate = incoming / "duplicate"
            duplicate.mkdir()
            (duplicate / VALID_FILENAMES[0]).write_bytes(b"duplicate")

            with self.assertRaisesRegex(ValueError, "duplicate release artifact"):
                assemble_release(
                    incoming,
                    root / "dist",
                    distribution="pysht",
                    version=VERSION,
                )

    def test_wrong_version_and_unknown_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            incoming.mkdir()
            _write_artifacts(incoming, (*VALID_FILENAMES, "notes.txt"))

            with self.assertRaisesRegex(ValueError, "unexpected release files"):
                assemble_release(
                    incoming,
                    root / "dist",
                    distribution="pysht",
                    version="0.1.1",
                )


if __name__ == "__main__":
    unittest.main()
