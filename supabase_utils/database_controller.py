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
import time
from langchain_core.documents import Document
from tqdm import tqdm
from supabase import create_client


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def insert_chunks_to_db(
    chunks:           list[Document],
    embeddings_model,
    supabase_client,
    table_name:       str  = "documents",
    batch_size:       int  = 50,
) -> None:
    """
    Generate embeddings for every chunk and bulk-insert them into Supabase.

    Each chunk row stored in the database contains:
        paper_id    — identifier of the source document
        method      — chunking strategy that produced this chunk
        chunk_index — zero-based position within the strategy's output
        content     — raw text of the chunk
        embedding   — 1536-dimensional float vector (text-embedding-3-small)
        metadata    — full metadata dict (JSON)

    Rows are sent in batches to avoid hitting Supabase request size limits.
    A short sleep between batches prevents rate-limit errors from the
    embeddings API.

    Args:
        chunks:           List of LangChain Documents to embed and store.
        embeddings_model: Initialised LangChain embeddings instance.
        table_name:       Target Supabase table (default: "documents").
        batch_size:       Number of rows per insert request (default: 50).
    """
    rows: list[dict] = []

    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings & inserting")):
        embedding_vector = embeddings_model.embed_query(chunk.page_content)

        rows.append({
            "paper_id":    chunk.metadata["paper_id"],
            "method":      chunk.metadata["method"],
            "chunk_index": chunk.metadata["chunk_index"],
            "content":     chunk.page_content,
            "embedding":   embedding_vector,
            "metadata":    chunk.metadata,
        })

        if len(rows) >= batch_size:
            try:
                supabase_client.table(table_name).insert(rows).execute()
            except Exception as exc:
                print(f"[insert_chunks_to_db] Error inserting batch at chunk {i}: {exc}")
            rows = []
            time.sleep(0.2)   # brief pause to respect API rate limits

    # Flush any remaining rows
    if rows:
        try:
            supabase_client.table(table_name).insert(rows).execute()
        except Exception as exc:
            print(f"[insert_chunks_to_db] Error inserting final batch: {exc}")

    print(f"\nSuccessfully inserted {len(chunks)} chunks into '{table_name}'.")



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
