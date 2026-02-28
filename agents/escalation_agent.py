from db.models import Ticket, AuditLog, Session
from datetime import datetime
from langgraph.types import interrupt

def run_escalation(state: dict, llm) -> dict:
    context = ""
    for i, ticket in enumerate(state['similar_tickets'][:3]):
        context += f"Past ticket {i+1}:\nIssue: {ticket['subject']}\nResolution: {ticket['answer'][:300]}\n\n"

    prompt = f"""You are an IT support expert. This ticket needs human review.

Ticket:
Title: {state['title']}
Description: {state['description']}
Category: {state['category']}
Confidence Score: {state['confidence_score']:.0%}

Similar past tickets for context:
{context}

Provide exactly 3 things:
1. SUMMARY: 2 sentence summary of the issue
2. PROBABLE CAUSE: Most likely root cause in 1 sentence
3. SUGGESTED FIRST STEP: One specific action the human agent should take first

Be concise and technical. Do not ask for more information."""

    response = llm.invoke(prompt)
    suggestion = response.content.strip()
    explanation = f"Escalated — confidence {state['confidence_score']:.0%} below 75% threshold. Requires human review."

    print(f"[Escalation] Ticket {state['ticket_id']} — waiting for human input")

    # ⚡ INTERRUPT — graph pauses here
    # Everything above runs before pause
    # Everything below runs after human approves
    human_input = interrupt({
        "ticket_id": state['ticket_id'],
        "title": state['title'],
        "ai_suggestion": suggestion,
        "confidence": state['confidence_score']
    })

    # Graph resumes here with human resolution
    human_resolution = human_input.get('resolution', suggestion)

    print(f"[Escalation] Human resolution received — saving and resuming")

    # Save to SQLite AFTER human approves — only runs once
    session = Session()
    ticket_db = Ticket(
        id=state['ticket_id'],
        title=state['title'],
        description=state['description'],
        category=state['category'],
        priority=state['priority'],
        confidence_score=state['confidence_score'],
        action="escalate",
        suggested_resolution=suggestion,
        final_resolution=human_resolution,
        explanation=explanation,
        resolved=True,
        resolved_at=datetime.now()
    )
    session.merge(ticket_db)
    log = AuditLog(
        ticket_id=state['ticket_id'],
        agent="escalation_agent",
        action="escalated_and_resolved",
        reasoning=explanation,
        confidence=state['confidence_score']
    )
    session.add(log)
    session.commit()
    session.close()

    return {
        "suggested_resolution": human_resolution,
        "action": "escalate",
        "explanation": explanation,
        "resolved": True
    }