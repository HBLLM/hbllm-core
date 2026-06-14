"""
Alembic Environment Configuration.

Supports both sync and async PostgreSQL connections.
Falls back to sync mode if asyncpg is not available.
"""

import os
from logging.config import fileConfig

from alembic import context

# Alembic Config object
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from environment if available
db_url = os.environ.get("HBLLM_DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without connecting."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects to the database."""
    from sqlalchemy import create_engine, pool

    url = config.get_main_option("sqlalchemy.url")
    # Convert async URLs to sync for Alembic
    if url and "+asyncpg" in url:
        url = url.replace("+asyncpg", "")

    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
