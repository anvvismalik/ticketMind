import uuid
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from db.vector_store import load_knowledge_base
from graph.ticket_graph import build_pipeline
import os

load_dotenv()

# Initialize
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load knowledge base
load_knowledge_base()

# Build pipeline
app = build_pipeline(llm)

def process_ticket(title: str, description: str, priority: str = "medium") -> dict:
    """Process a single ticket through the full pipeline."""
    result = app.invoke(
        {
            "ticket_id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "category": "",
            "priority": priority,
            "confidence_score": 0.0,
            "similar_tickets": [],
            "suggested_resolution": "",
            "action": "",
            "explanation": "",
            "resolved": False
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )
    return result

if __name__ == "__main__":
    # Test ticket
    result = process_ticket(
        title="Network connectivity issues affecting multiple devices",
        description="Multiple devices losing connection intermittently across the office.",
        priority="high"
    )
    print(f"\nAction: {result['action']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Resolved: {result['resolved']}")