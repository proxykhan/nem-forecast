"""The dashboard's database login can read the monitoring tables and nothing else."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError
from typer.testing import CliRunner

from nem_forecast.cli import READER_TABLES, app
from nem_forecast.db import apply_migrations

pytestmark = pytest.mark.integration


@pytest.fixture
def role(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    apply_migrations(pg_engine)
    monkeypatch.setenv("DATABASE_URL", pg_engine.url.render_as_string(hide_password=False))
    name = f"reader_{uuid.uuid4().hex[:8]}"
    yield name
    with pg_engine.begin() as conn:  # roles outlive the test database; remove this one
        conn.execute(text(f'DROP OWNED BY "{name}"'))
        conn.execute(text(f'DROP ROLE IF EXISTS "{name}"'))


def reader_url(output: str) -> str:
    match = re.search(r"^postgresql://\S+$", output, flags=re.MULTILINE)
    assert match is not None, output
    return match.group(0).replace("postgresql://", "postgresql+psycopg://", 1)


def test_the_reader_can_select_but_not_write(role: str) -> None:
    result = CliRunner().invoke(app, ["grant-reader", "--role", role])

    assert result.exit_code == 0, result.output
    assert f"Created read-only role {role}" in result.output
    reader = create_engine(reader_url(result.output))
    try:
        with reader.connect() as conn:
            for table in READER_TABLES:
                conn.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608 - fixed names
        with reader.connect() as conn, pytest.raises(DBAPIError, match="read-only"):
            conn.execute(text("CREATE TABLE sneaky (id INTEGER)"))
        with reader.connect() as conn, pytest.raises(DBAPIError):
            conn.execute(text("SELECT * FROM schema_migrations"))  # not granted
    finally:
        reader.dispose()


def test_running_it_again_rotates_the_password(role: str) -> None:
    first = CliRunner().invoke(app, ["grant-reader", "--role", role])
    second = CliRunner().invoke(app, ["grant-reader", "--role", role])

    assert f"Rotated read-only role {role}" in second.output
    old = create_engine(reader_url(first.output))
    try:
        with pytest.raises(DBAPIError, match="password authentication failed"):
            old.connect()
    finally:
        old.dispose()
