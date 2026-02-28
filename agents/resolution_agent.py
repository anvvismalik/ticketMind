from db.models import Ticket, AuditLog, Session
from datetime import datetime

def run_resolution(state: dict, llm) -> dict:
    context = ""
    for i, ticket in enumerate(state['similar_tickets'][:3]):
        context += f"Past ticket {i+1}:\nIssue: {ticket['subject']}\nResolution: {ticket['answer']}\n\n"

    prompt = f"""You are an IT support expert resolving a ticket directly.

Current ticket:
Title: {state['title']}
Description: {state['description']}
Category: {state['category']}

Similar past cases for reference:
{context}

Give a DIRECT technical resolution in 3-4 sentences.
- Do NOT ask for more information
- Give specific actionable steps to fix the problem right now
- Start with the most likely fix first"""

    response = llm.invoke(prompt)
    resolution = response.content.strip()
    explanation = f"Auto-resolved with {state['confidence_score']:.0%} confidence based on {len(state['similar_tickets'])} similar past tickets."

    session = Session()
    ticket_db = Ticket(
        id=state['ticket_id'],
        title=state['title'],
        description=state['description'],
        category=state['category'],
        priority=state['priority'],
        confidence_score=state['confidence_score'],
        action="auto_resolve",
        suggested_resolution=resolution,
        final_resolution=resolution,
        explanation=explanation,
        resolved=True,
        resolved_at=datetime.now()
    )
    session.add(ticket_db)
    log = AuditLog(
        ticket_id=state['ticket_id'],
        agent="resolution_agent",
        action="auto_resolved",
        reasoning=explanation,
        confidence=state['confidence_score']
    )
    session.add(log)
    session.commit()
    session.close()

    print(f"[Resolution] Auto-resolved ticket {state['ticket_id']}")
    return {
        "suggested_resolution": resolution,
        "action": "auto_resolve",
        "explanation": explanation,
        "resolved": True
    }