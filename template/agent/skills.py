"""
Skill tools — each skill makes one independent LLM call using the model
best suited for that type of work.

The orchestrator LLM decides which skill to call; the skill LLM does the
specialised work and returns a plain string. No nesting, no recursion.

search_knowledge is a placeholder — replace the body with your Glean API
call (or any retrieval backend) followed by the make_llm synthesis call.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent.config import MODEL_CODE, MODEL_GENERAL, make_llm


@tool
def analyze_data(data: str) -> str:
    """Analyze data and return structured insights."""
    return make_llm(MODEL_GENERAL).invoke([
        SystemMessage("You are a senior data analyst. Return concise, structured findings."),
        HumanMessage(data),
    ]).content


@tool
def write_code(task_description: str, language: str = "python") -> str:
    """Write production-quality code for a given task."""
    return make_llm(MODEL_CODE).invoke([
        SystemMessage(
            f"You are an expert {language} developer. "
            "Write clean, well-commented, runnable code. Return only the code."
        ),
        HumanMessage(task_description),
    ]).content


@tool
def search_knowledge(query: str) -> str:
    """
    Search and synthesize an answer from available context.

    Replace this implementation with your retrieval backend:
      hits = glean_search(query)                     # call Glean / vector DB / etc.
      docs = format_hits(hits)
      return make_llm(MODEL_GENERAL).invoke([
          SystemMessage("Synthesize a precise answer from the search results."),
          HumanMessage(f"Results:\n{docs}\n\nQuestion: {query}"),
      ]).content
    """
    return make_llm(MODEL_GENERAL).invoke([
        SystemMessage("Answer the question concisely and factually."),
        HumanMessage(query),
    ]).content
