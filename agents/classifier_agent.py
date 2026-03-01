# -*- coding: utf-8 -*-
import logging
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from db.vector_store import get_collection, embedder
from db.models import AuditLog, Session
import json

@tool
def search_knowledge_base(query: str, domain: str = "all") -> str:
    """Search ChromaDB for similar past support tickets, filtered by domain."""
    collection = get_collection()
    embedding = embedder.encode(query).tolist()

    if domain and domain not in ["all", "general"]:
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=5,
                where={"domain": {"$eq": domain}}
            )
            if not results['documents'][0]:
                results = collection.query(query_embeddings=[embedding], n_results=5)
        except Exception:
            results = collection.query(query_embeddings=[embedding], n_results=5)
    else:
        results = collection.query(query_embeddings=[embedding], n_results=5)

    similar = []
    for i in range(len(results['documents'][0])):
        similar.append({
            "subject": results['metadatas'][0][i].get('subject', ''),
            "answer": results['metadatas'][0][i].get('answer', ''),
            "queue": results['metadatas'][0][i].get('queue', 'general'),
            "priority": results['metadatas'][0][i].get('priority', 'medium'),
            "domain": results['metadatas'][0][i].get('domain', 'general'),
            "distance": round(results['distances'][0][i], 4)
        })
    return json.dumps(similar)


@tool
def save_audit_log(ticket_id: str, agent: str, action: str, reasoning: str) -> str:
    """Save agent decision to audit log. Do NOT pass confidence — it is handled automatically by the system."""
    return f"Logged: {action} by {agent}"


def compute_confidence(similar_tickets: list) -> float:
    if not similar_tickets:
        logging.warning("[Confidence] No similar tickets found, returning 0.0")
        return 0.0

    distances = [t['distance'] for t in similar_tickets]
    best = distances[0]
    worst = distances[-1]

    MAX_DISTANCE = 1.8

    top_score = max(0.0, 1.0 - (best / MAX_DISTANCE))
    gap_bonus = ((worst - best) / worst * 0.15) if worst > best else 0.0

    top3 = distances[:3]
    avg_top3 = sum(top3) / len(top3)
    avg_score = max(0.0, 1.0 - (avg_top3 / MAX_DISTANCE))

    final = round(0.6 * (min(1.0, top_score + gap_bonus)) + 0.4 * avg_score, 3)
    logging.warning(f"[Confidence] best={best:.3f} avg_top3={avg_top3:.3f} -> {final}")
    return final


def get_tools():
    return [search_knowledge_base, save_audit_log]


def run_classifier(state: dict, llm) -> dict:
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="""You are a support ticket classifier.
Follow these steps EXACTLY ONCE in order:

1. Detect the domain of this ticket:
   - IT_support: computer, network, software, hardware, email, VPN, printer, Windows, Office issues
   - customer_support: product issues, refunds, billing, orders, shipping, returns
   - HR_support: employee, payroll, benefits, leave, onboarding
   - finance_support: invoices, payments, accounting, expenses
   - general: unclear or mixed

2. Call search_knowledge_base ONCE with the query AND the detected domain

3. Call save_audit_log ONCE with ONLY these fields: ticket_id, agent, action, reasoning
   Do NOT pass confidence — it is handled automatically by the system.

4. Return JSON only: {"category": "...", "sentiment": "...", "domain": "..."}

Do NOT call calculate_confidence — confidence is handled automatically.
Do NOT repeat any tool call."""),
        HumanMessage(content=f"""Classify this ticket:
Title: {state['title']}
Description: {state['description']}
Ticket ID: {state['ticket_id']}

Detect domain first, then search within that domain.
Respond with JSON only: {{"category": "...", "sentiment": "...", "domain": "..."}}""")
    ]

    similar_tickets = []
    confidence = 0.0
    detected_domain = "general"

    for _ in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            logging.warning(f"[Classifier] Calling tool: {tool_name}")

            if tool_name == "search_knowledge_base":
                detected_domain = tool_args.get('domain', 'general')
                logging.warning(f"[Classifier] Domain: {detected_domain}")
                result = search_knowledge_base.invoke(tool_args)
                similar_tickets = json.loads(result)
                confidence = compute_confidence(similar_tickets)
                distances = [t['distance'] for t in similar_tickets]
                logging.warning(f"[Classifier] Distances: {distances}")
                logging.warning(f"[Classifier] Confidence computed: {confidence}")

            elif tool_name == "save_audit_log":
                # Save to DB directly with our computed confidence
                # Never trust LLM's confidence value — always use ours
                session = Session()
                log = AuditLog(
                    ticket_id=state['ticket_id'],
                    agent=tool_args.get('agent', 'classifier'),
                    action=tool_args.get('action', 'classified'),
                    reasoning=tool_args.get('reasoning', ''),
                    confidence=confidence
                )
                session.add(log)
                session.commit()
                session.close()
                result = f"Logged: {tool_args.get('action', 'classified')} by classifier"

            else:
                logging.warning(f"[Classifier] Unknown tool called: {tool_name} — skipping")
                result = f"Unknown tool {tool_name} ignored"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

    try:
        final_text = response.content.strip().replace('```json', '').replace('```', '')
        extracted = json.loads(final_text)
        category = extracted.get('category', 'Other')
        sentiment = extracted.get('sentiment', 'neutral')
        detected_domain = extracted.get('domain', detected_domain)
    except Exception as e:
        logging.warning(f"[Classifier] Failed to parse LLM response: {e} | Raw: {response.content[:200]}")
        category = similar_tickets[0].get('queue', 'Other') if similar_tickets else 'Other'
        sentiment = 'neutral'

    logging.warning(f"[Classifier] Category: {category} | Domain: {detected_domain} | Confidence: {confidence:.2f} | Sentiment: {sentiment}")
    return {
        "category": category,
        "confidence_score": confidence,
        "similar_tickets": similar_tickets,
        "domain": detected_domain
    }