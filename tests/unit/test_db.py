"""Migration-runner behaviour, exercised on SQLite. The real Postgres schema is tested in
tests/integration/test_migrations.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, text

from nem_forecast.db import (
    MIGRATIONS_DIR,
    MigrationError,
    apply_migrations,
    create_db_engine,
    discover_migrations,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")
    yield engine
    engine.dispose()


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "002_gadgets.sql").write_text("CREATE TABLE gadgets (id INTEGER PRIMARY KEY)")
    (directory / "001_widgets.sql").write_text("CREATE TABLE widgets (id INTEGER PRIMARY KEY)")
    return directory


def test_repo_migrations_are_numbered_contiguously() -> None:
    versions = [m.version for m in discover_migrations(MIGRATIONS_DIR)]

    assert versions == list(range(1, len(versions) + 1))
    assert len(versions) >= 4


def test_discovers_in_version_order(migrations_dir: Path) -> None:
    found = discover_migrations(migrations_dir)

    assert [(m.version, m.name) for m in found] == [(1, "widgets"), (2, "gadgets")]


@pytest.mark.parametrize("bad_name", ["1_widgets.sql", "003-sprockets.sql", "003_Sprockets.sql"])
def test_badly_named_file_rejected(migrations_dir: Path, bad_name: str) -> None:
    (migrations_dir / bad_name).write_text("SELECT 1")

    with pytest.raises(MigrationError, match="expected a name like"):
        discover_migrations(migrations_dir)


def test_duplicate_version_rejected(migrations_dir: Path) -> None:
    (migrations_dir / "001_other.sql").write_text("SELECT 1")

    with pytest.raises(MigrationError, match="version 1 already used"):
        discover_migrations(migrations_dir)


def test_checksum_ignores_line_endings(tmp_path: Path) -> None:
    (tmp_path / "001_unix.sql").write_bytes(b"SELECT 1;\nSELECT 2;\n")
    (tmp_path / "002_windows.sql").write_bytes(b"SELECT 1;\r\nSELECT 2;\r\n")

    unix, windows = discover_migrations(tmp_path)

    assert unix.checksum == windows.checksum


def test_applies_pending_migrations_once(engine: Engine, migrations_dir: Path) -> None:
    first = apply_migrations(engine, migrations_dir)
    second = apply_migrations(engine, migrations_dir)

    assert [m.version for m in first] == [1, 2]
    assert second == []
    assert {"widgets", "gadgets", "schema_migrations"} <= set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        recorded = conn.execute(text("SELECT version, name FROM schema_migrations")).all()
    assert sorted(recorded) == [(1, "widgets"), (2, "gadgets")]


def test_only_new_migrations_are_applied(engine: Engine, migrations_dir: Path) -> None:
    apply_migrations(engine, migrations_dir)
    (migrations_dir / "003_sprockets.sql").write_text("CREATE TABLE sprockets (id INTEGER)")

    applied = apply_migrations(engine, migrations_dir)

    assert [m.name for m in applied] == ["sprockets"]


def test_edited_migration_is_refused(engine: Engine, migrations_dir: Path) -> None:
    apply_migrations(engine, migrations_dir)
    (migrations_dir / "001_widgets.sql").write_text("CREATE TABLE widgets (id BIGINT)")

    with pytest.raises(MigrationError, match="changed after it was applied"):
        apply_migrations(engine, migrations_dir)


def test_failed_migration_is_not_recorded(engine: Engine, migrations_dir: Path) -> None:
    (migrations_dir / "003_broken.sql").write_text("CREATE TABLEE broken (id INTEGER)")

    with pytest.raises(Exception, match="syntax error"):
        apply_migrations(engine, migrations_dir)

    with engine.connect() as conn:
        versions = conn.execute(text("SELECT version FROM schema_migrations")).scalars().all()
    assert sorted(versions) == [1, 2]
