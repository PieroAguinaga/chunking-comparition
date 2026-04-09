"""
graph.py
--------
Builds and compiles the LangGraph chunking workflow.

Graph topology:
    [START] ──► chunk_section ──► [route] ──► chunk_section  (loop)
                                          └──► END
"""

from langgraph.graph import END, StateGraph

from chunck_agent.nodes.make_section_node import _make_chunk_section_node
from chunck_agent.schemas import ChunkerState


def _route_sections(state: ChunkerState) -> str:
    """Returns 'chunk_section' if sections remain, 'end' when all are processed."""
    if state["current_index"] < len(state["sections"]):
        return "chunk_section"
    return "end"


def build_agentic_chunker_graph(llm):
    """
    Compiles the LangGraph chunking workflow. The graph loops over sections
    one at a time until all are processed, then terminates.
    """
    graph = StateGraph(ChunkerState)

    graph.add_node("chunk_section", _make_chunk_section_node(llm))

    graph.set_entry_point("chunk_section")
    graph.add_conditional_edges(
        "chunk_section",
        _route_sections,
        {"chunk_section": "chunk_section", "end": END},
    )

    return graph.compile()