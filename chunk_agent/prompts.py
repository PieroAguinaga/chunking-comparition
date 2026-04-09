"""
prompts.py
----------
Prompt template used to instruct the LLM to split an academic paper section
into self-contained, RAG-optimised chunks.
"""

from langchain_core.prompts import ChatPromptTemplate


CHUNKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert in information retrieval and academic text processing.
Your task is to split a section from an academic paper into self-contained,
semantically coherent chunks optimised for retrieval-augmented generation (RAG).

Guidelines:
- Each chunk must be independently understandable without the others.
- Keep complete concepts, definitions, and equations inside a single chunk.
- Each chunk should answer a specific identifiable question or explain one concept.
- Target 150-400 words per chunk; never exceed 500 words.
- Do NOT split mathematical formulas, tables, or figures across chunks.
- If the section is already short and coherent, return it as a single chunk.""",
    ),
    (
        "human",
        "Split the following academic paper section into chunks:\n\n{section_content}",
    ),
])