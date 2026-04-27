"""
LLM gateway configuration.

make_llm() is a custom factory — not an official LangChain API.
It wraps langchain_openai.ChatOpenAI with your gateway's base_url and api_key
so every skill and the orchestrator share the same connection config.

Swap the gateway by changing LLM_GATEWAY_URL.
Swap a model by changing the MODEL_* env var — zero other code changes needed.
"""

import os

from langchain_openai import ChatOpenAI

LLM_GATEWAY_URL: str = os.environ["LLM_GATEWAY_URL"]
LLM_GATEWAY_KEY: str = os.environ["LLM_GATEWAY_KEY"]

MODEL_GENERAL: str = os.getenv("MODEL_GENERAL", "general")
MODEL_CODE: str    = os.getenv("MODEL_CODE",    "coder")
MODEL_FAST: str    = os.getenv("MODEL_FAST",    "fast")


def make_llm(model: str = MODEL_GENERAL, temperature: float = 0.1) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the internal LLM gateway."""
    return ChatOpenAI(
        model=model,
        base_url=LLM_GATEWAY_URL,
        api_key=LLM_GATEWAY_KEY,
        temperature=temperature,
    )
