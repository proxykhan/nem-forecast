"""Database engine and a minimal forward-only migration runner.

Migrations are plain numbered SQL files in ``migrations/`` (``001_name.sql``). Each one runs in
its own transaction and is recorded in ``schema_migrations`` with a checksum, so editing an
already-applied file is caught instead of silently diverging from production.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, text

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_FILENAME = re.compile(r"^(?P<version>\d{3,})_(?P<name>[a-z0-9_]+)\.sql$")
# Arbitrary constant: serialises concurrent migrators (e.g. a deploy racing a CI job) on Postgres.
_ADVISORY_LOCK_KEY = 7_201_993_501

_CREATE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER     PRIMARY KEY,
    name       TEXT        NOT NULL,
    checksum   TEXT        NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def checksum(self) -> str:
        # Normalise line endings so a Windows checkout and a Linux CI runner agree.
        return hashlib.sha256(self.sql.replace("\r\n", "\n").encode()).hexdigest()


def create_db_engine(url: str) -> Engine:
    # pool_pre_ping absorbs Neon's auto-suspend: a stale pooled connection is replaced, not raised.
    return create_engine(url, pool_pre_ping=True)


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    migrations: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise MigrationError(f"{path.name}: expected a name like 001_create_table.sql")
        version = int(match["version"])
        if version in migrations:
            raise MigrationError(
                f"{path.name}: version {version} already used by {migrations[version].path.name}"
            )
        migrations[version] = Migration(version, match["name"], path)
    return [migrations[v] for v in sorted(migrations)]


def apply_migrations(engine: Engine, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Apply every pending migration in version order and return the ones applied now."""
    migrations = discover_migrations(directory)
    with engine.connect() as conn:
        is_postgres = conn.dialect.name == "postgresql"
        if is_postgres:
            conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _ADVISORY_LOCK_KEY})
            conn.commit()
        try:
            return _apply_pending(conn, migrations)
        finally:
            if is_postgres:
                conn.rollback()
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _ADVISORY_LOCK_KEY})
                conn.commit()


def _apply_pending(conn: Connection, migrations: list[Migration]) -> list[Migration]:
    with conn.begin():
        conn.execute(text(_CREATE_TRACKING_TABLE))
        rows = conn.execute(text("SELECT version, checksum FROM schema_migrations")).all()
    applied: dict[int, str] = {int(version): str(checksum) for version, checksum in rows}

    newly_applied: list[Migration] = []
    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None:
            if recorded != migration.checksum:
                raise MigrationError(
                    f"{migration.path.name} changed after it was applied; "
                    "add a new migration instead of editing an old one"
                )
            continue
        with conn.begin():
            # no_parameters sends the file as one simple query, so multi-statement files work.
            conn.exec_driver_sql(migration.sql, execution_options={"no_parameters": True})
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, name, checksum) "
                    "VALUES (:version, :name, :checksum)"
                ),
                {
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration.checksum,
                },
            )
        newly_applied.append(migration)
    return newly_applied
