"""
Skills — specialized tools that make their own LLM calls.
Each skill targets a different model based on capability.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from .config import make_llm, MODEL_GENERAL, MODEL_CODE

_MOCK_CORPUS = {
    "sales": "Q1 revenue: $1.2M (up 15% YoY). Top product: Widget Pro at $450K.",
    "users": "Active users: 42,000. Churn rate: 3.2%. NPS score: 67.",
    "infra": "K8s cluster: 12 nodes, 87% CPU utilization. Postgres 15.2, 3 replicas.",
    "team": "Engineering: 18 FTEs. On-call rotation: 6 engineers. Sprint velocity: 42 pts.",
}


@tool
def analyze_data(data: str) -> str:
    """
    Analyze the provided data and return structured insights.
    Uses the general reasoning model (qwen3:8b).
    """
    llm = make_llm(MODEL_GENERAL, temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior data analyst. Given raw data, produce a concise analysis: "
            "key findings, patterns, anomalies, and 2-3 actionable recommendations. "
            "Be specific and use numbers where present."
        )),
        HumanMessage(content=f"Analyze this data:\n\n{data}"),
    ])
    return f"[ANALYSIS SKILL — {MODEL_GENERAL}]\n{response.content}"


@tool
def write_code(task_description: str, language: str = "python") -> str:
    """
    Write clean, working code for a given task.
    Uses the specialized coding model (qwen2.5-coder:14b).
    """
    llm = make_llm(MODEL_CODE, temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=(
            f"You are an expert {language} programmer. Write clean, well-structured, "
            f"runnable {language} code. Include only code and brief inline comments. "
            "No markdown fences, no extra prose."
        )),
        HumanMessage(content=f"Task: {task_description}"),
    ])
    return f"[CODE SKILL — {MODEL_CODE} / {language}]\n{response.content}"


@tool
def search_knowledge(query: str) -> str:
    """
    Search internal knowledge base (simulates Glean).
    Returns relevant documents and uses LLM to synthesize an answer.
    """
    query_lower = query.lower()
    hits = [doc for key, doc in _MOCK_CORPUS.items() if key in query_lower or query_lower in doc.lower()]

    if not hits:
        hits = list(_MOCK_CORPUS.values())[:2]

    context = "\n".join(hits)
    llm = make_llm(MODEL_GENERAL, temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a knowledge base assistant (like Glean). "
            "Answer the user's query using only the provided context documents. "
            "Be concise and cite relevant facts."
        )),
        HumanMessage(content=f"Query: {query}\n\nContext documents:\n{context}"),
    ])
    return f"[SEARCH SKILL — {len(hits)} doc(s) found]\n{response.content}"
