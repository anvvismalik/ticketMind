# -*- coding: utf-8 -*-
import logging
from db.models import Ticket, AuditLog, Session
from datetime import datetime


def run_resolution(state: dict, llm) -> dict:

    # Build RAG context from retrieved KB tickets
    context = ""
    relevant_cases = 0
    for i, ticket in enumerate(state['similar_tickets'][:3]):
        answer = ticket.get('answer', '').strip()
        subject = ticket.get('subject', '').strip()
        domain = ticket.get('domain', 'general')
        distance = ticket.get('distance', 999)

        # Only use tickets with actual resolution content and reasonable similarity
        if answer and subject and distance < 1.6:
            relevant_cases += 1
            context += f"Past Case {relevant_cases} (similarity distance: {distance}):\n"
            context += f"  Issue: {subject}\n"
            context += f"  Domain: {domain}\n"
            context += f"  Resolution: {answer[:500]}\n\n"

    has_relevant_context = relevant_cases > 0

    if has_relevant_context:
        prompt = f"""You are a support expert. Use the past resolved cases below as your PRIMARY source.

━━━ PAST RESOLVED CASES ━━━
{context}
━━━ CURRENT TICKET ━━━
Title: {state['title']}
Description: {state['description']}
Category: {state['category']}
Domain: {state.get('domain', 'general')}

━━━ INSTRUCTIONS ━━━
1. Look at the past cases above — if any are relevant, adapt their resolution directly
2. Mention which past case you are basing your resolution on
3. Give 3-4 specific, actionable steps
4. Start with the most likely fix first
5. Do NOT ask for more information
6. Be direct and technical

Resolution:"""
    else:
        # No good KB matches — fall back to LLM general knowledge
        prompt = f"""You are a support expert. No closely matching past cases were found in the knowledge base.

━━━ CURRENT TICKET ━━━
Title: {state['title']}
Description: {state['description']}
Category: {state['category']}
Domain: {state.get('domain', 'general')}

━━━ INSTRUCTIONS ━━━
1. Use your general knowledge to resolve this ticket
2. Give 3-4 specific, actionable steps
3. Start with the most likely fix first
4. Do NOT ask for more information
5. Be direct and technical

Resolution:"""

    logging.warning(f"[Resolution] Ticket {state['ticket_id']} — {relevant_cases} relevant past cases found")

    response = llm.invoke(prompt)
    resolution = response.content.strip()

    if has_relevant_context:
        explanation = f"Auto-resolved with {state['confidence_score']:.0%} confidence based on {relevant_cases} similar past tickets from KB."
    else:
        explanation = f"Auto-resolved with {state['confidence_score']:.0%} confidence using general knowledge (no close KB matches found)."

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
    session.merge(ticket_db)

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

    logging.warning(f"[Resolution] Auto-resolved ticket {state['ticket_id']} | KB-grounded: {has_relevant_context}")

    return {
        "suggested_resolution": resolution,
        "action": "auto_resolve",
        "explanation": explanation,
        "resolved": True
    }