"""Integration fixtures backed by a real PostgreSQL server.

Each test gets a freshly created database (via pytest-postgresql), dropped afterwards. The tests
skip, rather than fail, when no server is reachable, so ``pytest`` still works on a laptop
without Docker running.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest
from psycopg import Connection
from pytest_postgresql import factories
from sqlalchemy import URL, Engine, create_engine

PG_HOST = os.getenv("TEST_PG_HOST", "localhost")
PG_PORT = int(os.getenv("TEST_PG_PORT", "5432"))
PG_USER = os.getenv("TEST_PG_USER", "postgres")
PG_PASSWORD = os.getenv("TEST_PG_PASSWORD", "postgres")

pg_server = factories.postgresql_noproc(
    host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname="nem_test"
)
pg_database = factories.postgresql("pg_server")


@pytest.fixture(scope="session")
def postgres_available() -> None:
    try:
        socket.create_connection((PG_HOST, PG_PORT), timeout=1).close()
    except OSError:
        pytest.skip(f"PostgreSQL not reachable at {PG_HOST}:{PG_PORT}")


@pytest.fixture
def pg_engine(postgres_available: None, request: pytest.FixtureRequest) -> Iterator[Engine]:
    """SQLAlchemy engine on a fresh, empty database."""
    database: Connection[tuple[object, ...]] = request.getfixturevalue("pg_database")
    url = URL.create(
        "postgresql+psycopg",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=database.info.dbname,
    )
    engine = create_engine(url)
    yield engine
    engine.dispose()
