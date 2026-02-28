from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from agents.classifier_agent import run_classifier
from agents.resolution_agent import run_resolution
from agents.escalation_agent import run_escalation
from agents.learning_agent import run_learning

class TicketState(TypedDict):
    ticket_id: str
    title: str
    description: str
    category: str
    priority: str
    confidence_score: float
    similar_tickets: List[dict]
    suggested_resolution: str
    action: str
    explanation: str
    resolved: bool

def route_ticket(state: TicketState) -> str:
    if state['confidence_score'] > 0.75:
        return "resolution_agent"
    else:
        return "escalation_agent"

def build_pipeline(llm):
    def classifier_node(state): return run_classifier(state, llm)
    def resolution_node(state): return run_resolution(state, llm)
    def escalation_node(state): return run_escalation(state, llm)
    def learning_node(state): return run_learning(state)

    workflow = StateGraph(TicketState)
    workflow.add_node("classifier_agent", classifier_node)
    workflow.add_node("resolution_agent", resolution_node)
    workflow.add_node("escalation_agent", escalation_node)
    workflow.add_node("learning_agent", learning_node)

    workflow.set_entry_point("classifier_agent")
    workflow.add_conditional_edges(
        "classifier_agent",
        route_ticket,
        {
            "resolution_agent": "resolution_agent",
            "escalation_agent": "escalation_agent"
        }
    )
    workflow.add_edge("resolution_agent", "learning_agent")
    workflow.add_edge("learning_agent", END)
    workflow.add_edge("escalation_agent", "learning_agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)