from datetime import date

import pytest

from pelinker.config import KBConfig


def test_kb_config_ok():
    c = KBConfig(
        name="test-kb",
        version="1.2.3",
        created_at=date(2025, 1, 15),
        description="demo",
        entity_count=42,
    )
    assert c.entity_count == 42


def test_kb_config_entity_count_optional():
    c = KBConfig(name="x", version="0.0.1", created_at=date.today())
    assert c.entity_count is None


def test_kb_config_semver_prerelease_build():
    KBConfig(
        name="x",
        version="1.0.0-rc.1+build.9",
        created_at=date.today(),
    )


@pytest.mark.parametrize(
    "version",
    ["", "1", "1.0", "v1.0.0", "01.0.0", "1.0.a"],
)
def test_kb_config_rejects_bad_semver(version: str):
    with pytest.raises(ValueError):
        KBConfig(name="x", version=version, created_at=date.today())


def test_kb_config_rejects_empty_name():
    with pytest.raises(ValueError):
        KBConfig(name="   ", version="1.0.0", created_at=date.today())


def test_kb_config_rejects_negative_entity_count():
    with pytest.raises(ValueError):
        KBConfig(
            name="x",
            version="1.0.0",
            created_at=date.today(),
            entity_count=-1,
        )
