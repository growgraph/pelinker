"""Tilde and env-var expansion for CLI path options.

A shell expands ``~`` before the process ever sees it. Anything that builds argv directly
— an IDE run configuration, a systemd unit, ``subprocess`` with a list — does not, so a
literal ``~/data/x`` reached click and failed ``exists=True`` against a path that exists.
Unexpanded *output* paths are worse than an error: they create a directory named ``~``.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from pelinker.core.paths import ExpandedPath, expand_config_path


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows equivalent
    return tmp_path


def test_expand_config_path_handles_tilde_and_vars(home: Path, monkeypatch) -> None:
    monkeypatch.setenv("PELINKER_TEST_DIR", "sub")

    assert expand_config_path("~/x.csv") == home / "x.csv"
    assert expand_config_path("~/$PELINKER_TEST_DIR/x.csv") == home / "sub" / "x.csv"
    assert expand_config_path(None) is None


def test_existing_file_is_accepted_through_a_tilde(home: Path) -> None:
    target = home / "data" / "kb.csv"
    target.parent.mkdir(parents=True)
    target.write_text("entity_id,label\n", encoding="utf-8")

    @click.command()
    @click.option("--kb", type=ExpandedPath(exists=True))
    def cli(kb: str) -> None:
        click.echo(kb)

    result = CliRunner().invoke(cli, ["--kb", "~/data/kb.csv"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(target)


def test_plain_click_path_would_have_rejected_it(home: Path) -> None:
    """Pins why the custom type exists rather than the stock one."""
    target = home / "data" / "kb.csv"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")

    @click.command()
    @click.option("--kb", type=click.Path(exists=True))
    def cli(kb: str) -> None:  # pragma: no cover - the invocation fails first
        click.echo(kb)

    result = CliRunner().invoke(cli, ["--kb", "~/data/kb.csv"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_output_path_expands_instead_of_creating_a_tilde_directory(home: Path) -> None:
    @click.command()
    @click.option("--out", type=ExpandedPath())
    def cli(out: str) -> None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("ok", encoding="utf-8")

    result = CliRunner().invoke(cli, ["--out", "~/results/gold.json"])

    assert result.exit_code == 0, result.output
    assert (home / "results" / "gold.json").read_text(encoding="utf-8") == "ok"
    assert not Path("~").exists()  # never a literal tilde directory


def test_absolute_and_relative_paths_are_untouched(home: Path, tmp_path: Path) -> None:
    target = tmp_path / "plain.csv"
    target.write_text("x", encoding="utf-8")

    @click.command()
    @click.option("--p", type=ExpandedPath(exists=True))
    def cli(p: str) -> None:
        click.echo(p)

    result = CliRunner().invoke(cli, ["--p", str(target)])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(target)


def test_path_type_conversion_still_applies(home: Path) -> None:
    target = home / "kb.csv"
    target.write_text("x", encoding="utf-8")
    seen: list[object] = []

    @click.command()
    @click.option("--kb", type=ExpandedPath(exists=True, path_type=Path))
    def cli(kb: Path) -> None:
        seen.append(kb)

    result = CliRunner().invoke(cli, ["--kb", "~/kb.csv"])

    assert result.exit_code == 0, result.output
    assert isinstance(seen[0], Path) and seen[0] == target


def test_every_cli_path_option_uses_the_expanding_type() -> None:
    """Guards against a new script reintroducing the bug with stock click.Path."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for directory in ("run", "pelinker/cli"):
        for path in (root / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "click.Path(" in text:
                offenders.append(str(path.relative_to(root)))

    assert not offenders, f"use ExpandedPath instead of click.Path in: {offenders}"
