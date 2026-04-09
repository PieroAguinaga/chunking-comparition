"""
chunking_strategy_comparison.py
--------------------------------
Entry point for the RAG Chunking Strategy Benchmark.

Reads a folder of Markdown-formatted academic papers, applies four distinct
chunking strategies to each document, generates vector embeddings for every
chunk, and stores everything in a Supabase/pgvector database for later
evaluation.

Chunking strategies applied:
    - markdown_header : splits on Markdown heading hierarchy (H1 / H2 / H3)
    - page            : splits on ---PAGE--- delimiters from the PDF conversion
    - semantic        : embedding-similarity-based adaptive splitting
    - agentic         : LangGraph + LLM-driven splitting (see agentic_chunker.py)

Usage:
    Ensure the .env file contains the required credentials (see README), then:

        python chunking_strategy_comparison.py

The script will:
    1. Load all .md files from the `converted_files/` directory.
    2. Apply all four chunking strategies.
    3. Generate embeddings via Azure OpenAI (text-embedding-3-small, 1536-dim).
    4. Insert every chunk into the `documents` table in Supabase.

Environment variables required:
    SUPABASE_URL            — Supabase project URL
    SUPABASE_KEY            — Supabase service-role key
    AZURE_OPENAI_API_KEY    — Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT   — Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION— API version (e.g. "2024-02-01")
    AZURE_CHAT_DEPLOYMENT   — Deployment name for the chat model (agentic chunker)
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from supabase import create_client
from supabase_utils.database_controller import insert_chunks_to_db
from tqdm import tqdm

from chunk_agent.interface import agentic_chunker
from chunkers import markdown_header_chunker, page_chunker, semantic_chunker
from utils.llm import get_llm, get_embeddings
from config.settings import settings

load_dotenv()

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------

def read_markdown(filepath: str) -> Document:
    """
    Load a single Markdown file into a LangChain Document.

    The document metadata records the file path and a `paper_id` derived from
    the filename stem, which is used as a grouping key in the database.

    Args:
        filepath: Absolute or relative path to the .md file.

    Returns:
        A LangChain Document with `page_content` and `metadata`.
    """
    path = Path(filepath)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Document(
        page_content=content,
        metadata={
            "source":   str(path),
            "paper_id": path.stem,
        },
    )


def read_markdown_folder(folder_path: str) -> list[Document]:
    """
    Load all .md files from a directory into a list of LangChain Documents.

    Args:
        folder_path: Path to the directory containing Markdown files.

    Returns:
        List of Documents, one per .md file found.
    """
    path      = Path(folder_path)
    documents = [read_markdown(str(file)) for file in path.glob("*.md")]
    return documents



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Models ---------------------------------------------------------------
    embeddings = get_embeddings()
    llm = get_llm()


    # -- Load documents -------------------------------------------------------
    documents = read_markdown_folder("converted_files")
    print(f"Loaded {len(documents)} document(s) from 'converted_files/'.")

    # -- Apply all chunking strategies ----------------------------------------
    all_chunks: list[Document] = []

    print("\n[1/4] Applying Markdown Header chunker...")
    all_chunks.extend(markdown_header_chunker(documents))

    print("[2/4] Applying Page chunker...")
    all_chunks.extend(page_chunker(documents))

    print("[3/4] Applying Semantic chunker...")
    all_chunks.extend(semantic_chunker(documents, embeddings))

    print("[4/4] Applying Agentic chunker (LLM-driven)...")
    all_chunks.extend(agentic_chunker(documents, llm))

    # -- Summary --------------------------------------------------------------
    method_counts = {}
    for chunk in all_chunks:
        m = chunk.metadata["method"]
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\nTotal chunks generated: {len(all_chunks)}")
    for method, count in method_counts.items():
        print(f"  {method:<20}: {count} chunks")

    # -- Insert into Supabase -------------------------------------------------
    print("\nInserting chunks into Supabase...")
    insert_chunks_to_db(all_chunks, supabase_client=supabase, embeddings_model=embeddings)
