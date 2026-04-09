"""
database_controller.py
-----------------------
Utilities for managing the local Supabase (PostgreSQL + pgvector) database
used by the RAG Chunking Strategy Benchmark.

The primary function, `execute_sql_file_if_needed`, is an idempotent helper
that initialises the schema defined in `semantic_database.sql` only when the
target table does not yet exist.  This makes it safe to call on every run
without risk of overwriting existing data.

Connection defaults target a locally hosted Supabase instance (127.0.0.1:54322)
started with the Supabase CLI (`supabase start`).  Adjust the constants below
or pass parameters explicitly when using a remote project.
"""

import psycopg2


def table_exists(cursor: "psycopg2.extensions.cursor", table_name: str) -> bool:
    """
    Check whether a table exists in the `public` schema.

    Uses the information_schema view so the query works across PostgreSQL
    versions without requiring superuser privileges.

    Args:
        cursor:     An open psycopg2 cursor connected to the target database.
        table_name: Name of the table to check (case-sensitive).

    Returns:
        True if the table exists in the public schema, False otherwise.
    """
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM   information_schema.tables
            WHERE  table_schema = 'public'
            AND    table_name   = %s
        );
        """,
        (table_name,),
    )
    return cursor.fetchone()[0]


def execute_sql_file_if_needed(filepath: str, table_name: str) -> None:
    """
    Execute a SQL setup file only if the target table does not yet exist.

    This is an idempotent initialisation helper.  On first run it creates
    the pgvector extension, the `documents` table, the required indexes, and
    the `match_documents` RPC function (as defined in `semantic_database.sql`).
    On subsequent runs it exits early to avoid re-running DDL statements.

    Database connection targets a locally hosted Supabase instance.  If your
    Supabase project is remote, update the connection parameters accordingly.

    Args:
        filepath:   Path to the .sql file to execute (e.g. "semantic_database.sql").
        table_name: Name of the table used as the existence check guard.

    Raises:
        Does not raise; errors are printed to stdout so the calling script can
        continue with a partial setup where appropriate.
    """
    conn = psycopg2.connect(
        host     = "127.0.0.1",
        port     = 54322,
        database = "postgres",
        user     = "postgres",
        password = "postgres",
    )
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        if table_exists(cursor, table_name):
            print(f"[database_controller] Table '{table_name}' already exists — skipping setup.")
            return

        print(f"[database_controller] Table '{table_name}' not found — executing '{filepath}'...")

        with open(filepath, "r", encoding="utf-8") as f:
            sql = f.read()

        cursor.execute(sql)
        print("[database_controller] Schema initialised successfully.")

    except Exception as exc:
        print(f"[database_controller] Error during setup: {exc}")

    finally:
        cursor.close()
        conn.close()
