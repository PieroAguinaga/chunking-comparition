"""
make_section_node.py
--------
LangGraph node that processes one document section per invocation, calling
the LLM to split it into chunks. Short sections (<40 words) bypass the LLM,
and any LLM failure falls back to returning the raw section as a single chunk.
"""

from langchain_core.documents import Document

from chunck_agent.prompts import CHUNKING_PROMPT
from chunck_agent.schemas import ChunkerState, SectionChunks


def _make_chunk_section_node(llm):
    """
    Returns a node callable that chunks one section from the graph state.
    Uses structured output to parse the LLM response directly into SectionChunks.
    """
    structured_chain = CHUNKING_PROMPT | llm.with_structured_output(SectionChunks)

    def chunk_section_node(state: ChunkerState) -> dict:
        section       = state["sections"][state["current_index"]]
        content       = section["content"]
        base_metadata = section["metadata"]
        counter       = state["chunk_counter"]

        # Short sections bypass the LLM entirely
        if len(content.split()) < 40:
            return {
                "all_chunks": [Document(
                    page_content=content,
                    metadata={**base_metadata, "method": "agentic", "chunk_index": counter, "title": "", "key_concepts": []},
                )],
                "current_index": state["current_index"] + 1,
                "chunk_counter": counter + 1,
            }

        # LLM call with structured output
        try:
            result: SectionChunks = structured_chain.invoke({"section_content": content})
            llm_chunks = result.chunks
        except Exception as exc:
            print(f"[agentic_chunker] LLM call failed for section '{content[:60]}...': {exc}")
            llm_chunks = []

        # Fallback: preserve raw section if LLM returned nothing
        if not llm_chunks:
            return {
                "all_chunks": [Document(
                    page_content=content,
                    metadata={**base_metadata, "method": "agentic", "chunk_index": counter, "title": "", "key_concepts": []},
                )],
                "current_index": state["current_index"] + 1,
                "chunk_counter": counter + 1,
            }

        # Convert Pydantic objects to LangChain Documents
        new_docs: list[Document] = []
        for chunk_item in llm_chunks:
            if not chunk_item.content.strip():
                continue
            new_docs.append(Document(
                page_content=chunk_item.content,
                metadata={
                    **base_metadata,
                    "method":       "agentic",
                    "chunk_index":  counter,
                    "title":        chunk_item.title,
                    "key_concepts": chunk_item.key_concepts,
                },
            ))
            counter += 1

        return {
            "all_chunks":    new_docs,
            "current_index": state["current_index"] + 1,
            "chunk_counter": counter,
        }

    return chunk_section_node