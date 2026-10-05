"""Locating versioned knowledge-base files on disk."""

import re

_SYNTHESIS_RE = re.compile(r"^properties\.synthesis\.(\d+)\.csv$")
"""Only canonical ``properties.synthesis.<version>.csv`` files are versions.

Derived variants live beside them (``properties.synthesis.2.diverse.csv``), and matching
on ``".synthesis." in name`` used to pull those in and then crash converting ``diverse``
to an int — which made every KB regeneration run fail at the first step.
"""


def fetch_latest_kb(
    path_derived,
) -> tuple[str | None, int]:
    filename_versions = []
    for file in path_derived.iterdir():
        if not file.is_file():
            continue
        match = _SYNTHESIS_RE.match(file.name)
        if match is not None:
            filename_versions.append((file.name, int(match.group(1))))
    filename_versions.sort(key=lambda x: x[1])
    if filename_versions:
        return filename_versions[-1]
    else:
        return None, -1
