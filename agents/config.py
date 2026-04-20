from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = "http://10.0.0.224:11434"

MODEL_GENERAL = "qwen3:8b"
MODEL_CODE = "qwen2.5-coder:14b"
MODEL_FAST = "llama3.1:8b"


def make_llm(model: str = MODEL_GENERAL, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
