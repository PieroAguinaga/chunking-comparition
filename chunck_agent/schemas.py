"""
schemas.py
----------
Pydantic models for the LLM's structured output and the LangGraph state
used across the chunking pipeline.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class ChunkItem(BaseModel):
    """A single self-contained chunk returned by the LLM."""

    title: str = Field(description="Short descriptive title for this chunk (5-10 words).")
    content: str = Field(description="The chunk text, preserving the original wording of the paper.")
    key_concepts: list[str] = Field(description="2 to 4 key concepts or technical terms covered in this chunk.")


class SectionChunks(BaseModel):
    """Top-level structured output returned by the LLM for a single section."""

    chunks: list[ChunkItem] = Field(description="Ordered list of self-contained chunks extracted from the section.")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ChunkerState(TypedDict):
    """
    Mutable state shared across all nodes of the chunking graph.

    - sections:      ordered list of sections to process.
    - current_index: points to the section being processed.
    - chunk_counter: globally unique index assigned to each new chunk.
    - all_chunks:    accumulated Documents; operator.add merges partial lists
                     instead of overwriting them.
    """

    sections:      list[dict]
    current_index: int
    chunk_counter: int
    all_chunks:    Annotated[list[Document], operator.add]