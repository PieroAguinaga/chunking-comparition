"""
interface.py
----------
Public entry point for the agentic chunking pipeline. Iterates over a list
of documents, pre-splits each one into sections, and runs the LangGraph
workflow to produce LLM-generated chunks enriched with title and key_concepts.
"""

from langchain_core.documents import Document

from chunck_agent.graph import build_agentic_chunker_graph
from chunck_agent.schemas import ChunkerState
from chunck_agent.utils import _split_into_sections


def agentic_chunker(documents: list[Document], llm) -> list[Document]:
    """
    Splits a list of Markdown documents into RAG-optimised chunks using an
    LLM-driven LangGraph workflow. The graph is compiled once and reused
    across all documents.

    Returns a list of Documents with metadata:
        method, chunk_index, title, key_concepts.
    """
    app        = build_agentic_chunker_graph(llm)
    all_chunks: list[Document] = []

    for doc in documents:
        sections = _split_into_sections(doc.page_content, doc.metadata)
        if not sections:
            continue

        initial_state: ChunkerState = {
            "sections":      sections,
            "current_index": 0,
            "chunk_counter": 0,
            "all_chunks":    [],
        }

        final_state = app.invoke(initial_state)
        all_chunks.extend(final_state["all_chunks"])

    return all_chunks