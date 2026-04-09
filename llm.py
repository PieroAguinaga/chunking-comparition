"""
agent/llm.py

Factory functions for Azure OpenAI models.
This makes it trivial to swap models or add tracing later.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config.settings import settings

load_dotenv()


def get_llm(temperature: float | None = None, streaming: bool = False) -> AzureChatOpenAI:
    """
    Return a configured AzureChatOpenAI instance.

    Args:
        temperature: Overrides settings.llm_temperature when provided.
        streaming:   Enables token streaming (used by the SSE endpoint).
    """
    return AzureChatOpenAI(
        azure_deployment=settings.azure_chat_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    """
    Return a configured AzureOpenAIEmbeddings instance.
    Used by the RAG pipeline to embed paper abstracts and query vectors.
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
