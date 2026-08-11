"""Serve the compiled documentation without browser caching."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _NoCacheHandler(SimpleHTTPRequestHandler):
    """Serve static files while forcing development browsers to revalidate."""

    def end_headers(self) -> None:
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("docs/_build/site"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    directory = args.directory.resolve(strict=True)
    handler = partial(_NoCacheHandler, directory=str(directory))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(
            f"Serving {directory} at http://{args.host}:{args.port}/ "
            "with caching disabled"
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
