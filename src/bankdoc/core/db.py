"""Database access helpers.

We use psycopg 3 with a connection pool. The pool keeps a few connections
open and reuses them, which is much faster than opening a new connection
for every request (important once the API is running).
"""

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from bankdoc.core.config import get_settings

# Module-level pool, created lazily on first use.
_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    """Called once for every new connection created by the pool.

    register_vector teaches psycopg how to convert Python lists / numpy
    arrays to and from the PostgreSQL VECTOR type.
    """
    register_vector(conn)


def get_pool() -> ConnectionPool:
    """Create the connection pool on first call and return it afterwards."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            configure=_configure_connection,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT 1")

    The connection is committed on success, rolled back on error, and
    returned to the pool automatically when the block exits.
    """
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    """Close all pooled connections. Call this on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
