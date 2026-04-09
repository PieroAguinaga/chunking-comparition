"""
retrieval_chain.py
-------------------
Vector-similarity retrieval layer for the RAG Chunking Strategy Benchmark.

Exposes a single public function, `search_similar_chunks`, which:
    1. Converts a natural-language query into a 1536-dim embedding.
    2. Calls the `match_documents` RPC function defined in Supabase/pgvector.
    3. Returns the top-k most similar chunks, optionally filtered by paper or
       chunking method.

The function is used by `test_retrieval.py` to evaluate how well each
chunking strategy retrieves relevant context for a given question.

Environment variables required (loaded via .env):
    SUPABASE_URL            — Supabase project URL
    SUPABASE_KEY            — Supabase service-role key
    AZURE_OPENAI_API_KEY    — Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT   — Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION— API version (e.g. "2024-02-01")
"""

import os

from dotenv import load_dotenv
from supabase import create_client
from config.settings import settings
from llm import get_embeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Shared clients (initialised once at import time)
# ---------------------------------------------------------------------------

supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

embeddings = get_embeddings()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_similar_chunks(
    query:    str,
    top_k:    int = 5,
    paper_id: str = "",
    method:   str = "",
) -> list[dict]:
    """
    Retrieve the most semantically similar chunks for a given query.

    Embeds the query with the same model used at ingestion time and delegates
    the nearest-neighbour search to the `match_documents` PostgreSQL function
    (IVFFlat cosine index).  Optional filters allow narrowing results to a
    specific paper or chunking strategy, which is essential for per-method
    evaluation in the benchmark.

    Args:
        query:    Natural-language question or search string.
        top_k:    Maximum number of chunks to return (default: 5).
        paper_id: If provided, restrict results to this paper identifier.
                  Pass an empty string (default) to search all papers.
        method:   If provided, restrict results to this chunking strategy
                  ("markdown_header", "page", "semantic", or "agentic").
                  Pass an empty string (default) to search all methods.

    Returns:
        List of result dicts, each containing:
            id          — database row id
            paper_id    — source paper identifier
            method      — chunking strategy that produced this chunk
            chunk_index — position of the chunk within its strategy's output
            content     — raw text of the chunk
            metadata    — full metadata JSON object
            similarity  — cosine similarity score in [0, 1] (higher is better)

        Returns an empty list if no results are found or the RPC call fails.
    """
    query_embedding = embeddings.embed_query(query)

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count":     top_k,
            "filter_paper_id": paper_id,
            "filter_method":   method,
        },
    ).execute()

    if not response.data:
        print(f"[search_similar_chunks] No results found for query: '{query[:80]}'")
        return []

    return response.data


# ---------------------------------------------------------------------------
# Quick manual test (run this file directly to verify connectivity)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = "What is a neural network in machine learning?"

    print(f"Query: {test_query}\n")
    results = search_similar_chunks(test_query, top_k=5, method="page")

    for i, r in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"  Score  : {r['similarity']:.4f}")
        print(f"  Method : {r['method']}")
        print(f"  Chunk  : {r['chunk_index']}")
        print(f"  Content: {r['content'][:300]}...")
        print()
