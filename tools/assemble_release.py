"""Validate and assemble the complete set of pySHT release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path


def _wheel_platform(filename: str) -> str | None:
    """Return the supported release platform represented by a wheel name."""
    if "manylinux" in filename and filename.endswith("_x86_64.whl"):
        return "linux-x86_64"
    if "macosx" in filename and filename.endswith("_arm64.whl"):
        return "macos-arm64"
    if "macosx" in filename and filename.endswith("_x86_64.whl"):
        return "macos-x86_64"
    if filename.endswith("-win_amd64.whl"):
        return "windows-amd64"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_release(
    input_directory: Path,
    output_directory: Path,
    *,
    distribution: str,
    version: str,
) -> tuple[Path, ...]:
    """Copy one sdist and the four supported wheels into a clean directory."""
    if not input_directory.is_dir():
        raise ValueError(f"artifact input directory does not exist: {input_directory}")
    if output_directory.exists():
        raise ValueError(
            f"artifact output directory already exists: {output_directory}"
        )
    if not re.fullmatch(r"[a-z0-9]+", distribution):
        raise ValueError("distribution must contain only lowercase letters and digits")
    if not version or any(character.isspace() for character in version):
        raise ValueError("version must be a nonempty string without whitespace")

    source_files = sorted(path for path in input_directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in source_files):
        raise ValueError("release artifacts must not be symbolic links")

    basenames = [path.name for path in source_files]
    duplicate_names = sorted(
        name for name in set(basenames) if basenames.count(name) != 1
    )
    if duplicate_names:
        raise ValueError(
            "duplicate release artifact names: " + ", ".join(duplicate_names)
        )

    expected_sdist = f"{distribution}-{version}.tar.gz"
    sdists = [path for path in source_files if path.name == expected_sdist]
    wheels = [path for path in source_files if path.suffix == ".whl"]
    unexpected = [
        path.name
        for path in source_files
        if path.name != expected_sdist and path.suffix != ".whl"
    ]
    if unexpected:
        raise ValueError("unexpected release files: " + ", ".join(unexpected))
    if len(sdists) != 1:
        raise ValueError(f"expected exactly one source archive named {expected_sdist}")
    if len(wheels) != 4:
        raise ValueError(f"expected exactly four wheels, found {len(wheels)}")

    wheel_prefix = f"{distribution}-{version}-cp312-abi3-"
    platforms: dict[str, Path] = {}
    for wheel in wheels:
        if not wheel.name.startswith(wheel_prefix):
            raise ValueError(
                f"wheel does not match project, version, or ABI: {wheel.name}"
            )
        platform = _wheel_platform(wheel.name)
        if platform is None:
            raise ValueError(f"unsupported wheel platform: {wheel.name}")
        if platform in platforms:
            raise ValueError(f"multiple wheels found for {platform}")
        platforms[platform] = wheel

    expected_platforms = {
        "linux-x86_64",
        "macos-arm64",
        "macos-x86_64",
        "windows-amd64",
    }
    if set(platforms) != expected_platforms:
        missing = sorted(expected_platforms - set(platforms))
        raise ValueError("missing release wheel platforms: " + ", ".join(missing))

    output_directory.mkdir(parents=True)
    assembled: list[Path] = []
    for source in sorted([*sdists, *wheels], key=lambda path: path.name):
        destination = output_directory / source.name
        shutil.copy2(source, destination)
        assembled.append(destination)
    return tuple(assembled)


def write_manifest(artifacts: tuple[Path, ...], destination: Path) -> None:
    """Write a deterministic SHA-256 manifest outside the upload directory."""
    lines = [f"{_sha256(path)}  {path.name}\n" for path in artifacts]
    destination.write_text("".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distribution", default="pysht")
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release-artifact assembly command."""
    arguments = _parser().parse_args(argv)
    try:
        artifacts = assemble_release(
            arguments.input,
            arguments.output,
            distribution=arguments.distribution,
            version=arguments.version,
        )
        write_manifest(artifacts, arguments.manifest)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
