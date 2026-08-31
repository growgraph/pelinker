import os
from pathlib import Path


"""Path expansion for config and CLI inputs."""


def expand_config_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Expand environment variables and ``~`` in config or CLI path strings."""
    if path is None:
        return None
    return Path(os.path.expandvars(os.fspath(path))).expanduser()
