"""
MedBoard AI Agent
=================
An ambient intelligence layer for medical board management.
Surfaces critical signals from patient data, scheduling, and team communications.
"""

from __future__ import annotations

import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MedBoardState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_context: str | None
    triage_level: str | None  # "critical" | "urgent" | "routine"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_patient_summary(patient_id: str) -> str:
    """
    Retrieve a concise clinical summary for a given patient ID.
    Returns vitals, recent notes, active medications, and open alerts.
    """
    # Stub — wire to your EHR / FHIR API in production
    return (
        f"[Patient {patient_id}] Vitals stable. "
        "Active meds: Metformin 500mg, Lisinopril 10mg. "
        "Open alert: HbA1c overdue (last: 8.2% — 4 months ago). "
        "No critical flags."
    )


@tool
def triage_signal(description: str) -> dict:
    """
    Classify a clinical signal as 'critical', 'urgent', or 'routine'
    and return a recommended action.
    """
    description_lower = description.lower()
    if any(w in description_lower for w in ["chest pain", "stroke", "sepsis", "code"]):
        return {"level": "critical", "action": "Activate rapid response immediately."}
    if any(w in description_lower for w in ["overdue", "missed", "alert", "elevated"]):
        return {"level": "urgent", "action": "Schedule follow-up within 48 hours."}
    return {"level": "routine", "action": "Add to next scheduled review."}


@tool
def search_clinical_guidelines(query: str) -> str:
    """
    Search internal clinical guidelines and protocols for relevant recommendations.
    """
    # Stub — wire to your guideline vector store / RAG pipeline in production
    return (
        f"Guidelines for '{query}': "
        "Per ADA 2024 standards, HbA1c targets for most non-pregnant adults with T2DM "
        "should be <7%. Re-test every 3 months if uncontrolled, every 6 months if stable. "
        "Consider GLP-1 agonist if BMI > 30 and cardiovascular risk factors present."
    )


TOOLS = [get_patient_summary, triage_signal, search_clinical_guidelines]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _build_llm():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0).bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are MedBoard AI — the ambient intelligence layer for a medical team.

Your job:
1. Surface critical signals quickly and clearly.
2. Triage every clinical concern: critical / urgent / routine.
3. Recommend a specific, actionable next step for each item.
4. Never leave a question unanswered. Never pad with filler.

You have access to:
- get_patient_summary(patient_id) — pull patient context from the EHR
- triage_signal(description) — classify urgency and get a recommended action
- search_clinical_guidelines(query) — look up evidence-based protocols

Be concise, clinical, and decisive."""


def call_model(state: MedBoardState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    llm = _build_llm()
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MedBoardState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

tool_node = ToolNode(TOOLS)

builder = StateGraph(MedBoardState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()
graph.name = "MedBoard AI"
