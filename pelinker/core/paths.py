"""Path expansion for config and CLI inputs."""

import os
from pathlib import Path

import click


def expand_config_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Expand environment variables and ``~`` in config or CLI path strings."""
    if path is None:
        return None
    return Path(os.path.expandvars(os.fspath(path))).expanduser()


class ExpandedPath(click.Path):
    """``click.Path`` that expands ``~`` and ``$VAR`` before validating.

    Click checks the literal string it is given, and a shell is what normally turns ``~``
    into a home directory. Anything that builds argv directly — an IDE run configuration,
    a systemd unit, ``subprocess`` with a list — passes the tilde through untouched, so
    ``--flag ~/data/x`` fails ``exists=True`` against a path that exists. On the writing
    side it is worse than an error: an unexpanded output path silently creates a
    directory literally named ``~`` in the working directory.
    """

    def convert(self, value, param, ctx):  # type: ignore[override]
        if isinstance(value, (str, os.PathLike)):
            expanded = expand_config_path(value)
            if expanded is not None:
                value = os.fspath(expanded)
        return super().convert(value, param, ctx)
