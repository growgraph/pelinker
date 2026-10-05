"""Version discovery over the derived KB directory."""

from __future__ import annotations

from pathlib import Path

from pelinker.kb.registry import fetch_latest_kb


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("entity_id,label\n", encoding="utf-8")


def test_picks_the_highest_version(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "properties.synthesis.0.csv",
        "properties.synthesis.1.csv",
        "properties.synthesis.10.csv",
        "properties.synthesis.2.csv",
    )

    assert fetch_latest_kb(tmp_path) == ("properties.synthesis.10.csv", 10)


def test_derived_variants_do_not_crash_discovery(tmp_path: Path) -> None:
    """``*.synthesis.2.diverse.csv`` used to raise and break every KB regeneration."""
    _touch(
        tmp_path,
        "properties.synthesis.2.csv",
        "properties.synthesis.2.diverse.csv",
        "properties.ro.csv",
        "notes.txt",
    )

    assert fetch_latest_kb(tmp_path) == ("properties.synthesis.2.csv", 2)


def test_empty_directory_reports_no_version(tmp_path: Path) -> None:
    assert fetch_latest_kb(tmp_path) == (None, -1)
