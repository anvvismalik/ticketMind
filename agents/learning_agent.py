from db.models import AuditLog, Session
from db.vector_store import get_collection, embedder
from datetime import datetime
import uuid

def run_learning(state: dict) -> dict:
    if not state['resolved']:
        print("[Learning] Ticket not resolved — skipping")
        return {"resolved": False}

    collection = get_collection()

    learned_doc = f"""Subject: {state['title']}
Description: {state['description']}
Category: {state['category']}
Resolution: {state['suggested_resolution']}"""

    embedding = embedder.encode(learned_doc).tolist()

    collection.add(
        documents=[learned_doc],
        embeddings=[embedding],
        metadatas=[{
            "subject": state['title'],
            "answer": state['suggested_resolution'],
            "type": "learned",
            "queue": state['category'],
            "priority": state['priority']
        }],
        ids=[f"learned-{state['ticket_id']}"]
    )

    session = Session()
    log = AuditLog(
        ticket_id=state['ticket_id'],
        agent="learning_agent",
        action="learned",
        reasoning=f"Embedded resolved ticket. KB size: {collection.count()}",
        confidence=state['confidence_score']
    )
    session.add(log)
    session.commit()
    session.close()

    print(f"[Learning] Ticket learned. KB size: {collection.count()}")
    return {"resolved": True}


def embed_human_resolution(ticket_id: str, title: str, description: str,
                            category: str, priority: str, resolution: str) -> int:
    """Called directly from dashboard when human approves a ticket."""
    collection = get_collection()

    learned_doc = f"""Subject: {title}
Description: {description}
Category: {category}
Resolution: {resolution}"""

    embedding = embedder.encode(learned_doc).tolist()

    collection.add(
        documents=[learned_doc],
        embeddings=[embedding],
        metadatas=[{
            "subject": title,
            "answer": resolution,
            "type": "human_learned",
            "queue": category,
            "priority": priority
        }],
        ids=[f"human-learned-{ticket_id}"]
    )

    session = Session()
    log = AuditLog(
        ticket_id=ticket_id,
        agent="learning_agent",
        action="human_learned",
        reasoning=f"Human resolution embedded into KB. KB size: {collection.count()}",
        confidence=0.0
    )
    session.add(log)
    session.commit()
    session.close()

    return collection.count()