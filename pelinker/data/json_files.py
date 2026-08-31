"""Load JSON from plain UTF-8 files or gzip-compressed UTF-8 (e.g. ``*.json.gz``)."""

from __future__ import annotations

import gzip
import json
import pathlib
from typing import Any


def is_gzip_file_path(path: pathlib.Path | str) -> bool:
    """True when ``path`` ends with ``.gz`` (gzip-wrapped payload, e.g. ``*.json.gz``)."""
    return pathlib.Path(path).suffix.lower() == ".gz"


def load_json_path(path: pathlib.Path | str) -> Any:
    """
    Parse one JSON value from a filesystem path.

    If the path suffix is ``.gz``, the file is read as UTF-8 text through gzip;
    otherwise it is read as UTF-8 text from the raw file.
    """
    p = pathlib.Path(path).expanduser()
    if is_gzip_file_path(p):
        with gzip.open(p, mode="rt", encoding="utf-8") as fh:
            return json.load(fh)
    with p.open(mode="r", encoding="utf-8") as fh:
        return json.load(fh)
