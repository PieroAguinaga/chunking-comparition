"""
chunkers.py
-----------
Defines the three rule-based and embedding-based chunking strategies used
in the RAG benchmark:

    1. markdown_header_chunker — splits on Markdown heading hierarchy
    2. page_chunker            — splits on explicit ---PAGE--- delimiters
    3. semantic_chunker        — embedding-similarity-based adaptive splitting

The fourth strategy (agentic) lives in its own module: agentic_chunker.py

Each function accepts a list of LangChain `Document` objects and returns a
new list of `Document` objects enriched with the following metadata fields:
    method       — string identifier for the chunking strategy
    chunk_index  — zero-based position of the chunk within the strategy output
"""

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ---------------------------------------------------------------------------
# 1. Markdown Header Chunker
# ---------------------------------------------------------------------------

def markdown_header_chunker(documents: list[Document]) -> list[Document]:
    """
    Split documents at Markdown heading boundaries (H1, H2, H3).

    Each resulting chunk inherits the source document's metadata and receives
    the heading hierarchy as additional context (e.g. {"h1": "Introduction",
    "h2": "Background"}), which LangChain's MarkdownHeaderTextSplitter adds
    automatically.

    Args:
        documents: List of LangChain Documents containing Markdown text.

    Returns:
        List of Document chunks, one per heading-delimited section.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ]
    )

    chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)

        for i, split in enumerate(splits):
            split.metadata.update(doc.metadata)
            split.metadata["method"]      = "markdown_header"
            split.metadata["chunk_index"] = i
            chunks.append(split)

    return chunks


# ---------------------------------------------------------------------------
# 2. Page Chunker
# ---------------------------------------------------------------------------

def page_chunker(documents: list[Document]) -> list[Document]:
    """
    Split documents using explicit ---PAGE--- delimiters inserted during
    the PDF-to-Markdown conversion step.

    This strategy preserves the original page boundaries of the source PDF,
    making it straightforward to reference specific pages in retrieved chunks.

    Args:
        documents: List of LangChain Documents whose content uses ---PAGE---
                   as a page separator.

    Returns:
        List of Document chunks, one per page.
    """
    chunks: list[Document] = []

    for doc in documents:
        pages = doc.page_content.split("---PAGE---")

        for i, page in enumerate(pages):
            page_text = page.strip()
            if not page_text:
                continue

            chunks.append(Document(
                page_content=page_text,
                metadata={
                    **doc.metadata,
                    "method":      "page",
                    "chunk_index": i,
                }
            ))

    return chunks


# ---------------------------------------------------------------------------
# 3. Semantic Chunker
# ---------------------------------------------------------------------------

def semantic_chunker(documents: list[Document], embeddings) -> list[Document]:
    """
    Split documents at points of high semantic dissimilarity using embeddings.

    LangChain's SemanticChunker computes cosine similarity between consecutive
    sentences and introduces a split wherever similarity drops below a learned
    threshold, producing chunks that are internally coherent.

    Args:
        documents:  List of LangChain Documents to split.
        embeddings: An initialised LangChain embeddings model (e.g.
                    AzureOpenAIEmbeddings). Must match the model used at
                    query time.

    Returns:
        List of Document chunks ordered as they appear in the source text.
    """
    splitter = SemanticChunker(embeddings)
    chunks   = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["method"]      = "semantic"
        chunk.metadata["chunk_index"] = i

    return chunks
