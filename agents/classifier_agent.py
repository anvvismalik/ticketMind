from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from db.vector_store import get_collection, embedder
from db.models import AuditLog, Session
import json

@tool
def search_knowledge_base(query: str) -> str:
    """Search ChromaDB for similar past support tickets."""
    collection = get_collection()
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=5)
    similar = []
    for i in range(len(results['documents'][0])):
        similar.append({
            "subject": results['metadatas'][0][i].get('subject', ''),
            "answer": results['metadatas'][0][i].get('answer', ''),
            "queue": results['metadatas'][0][i].get('queue', 'general'),
            "priority": results['metadatas'][0][i].get('priority', 'medium'),
            "distance": round(results['distances'][0][i], 4)
        })
    return json.dumps(similar)

@tool
def calculate_confidence(distances_json: str) -> float:
    """Calculate confidence score from similarity distances."""
    distances = json.loads(distances_json)
    top_3 = distances[:3]
    avg_distance = sum(top_3) / len(top_3)
    confidence = max(0.0, min(1.0, 1 - (avg_distance / 2)))
    return round(confidence, 3)

@tool
def save_audit_log(ticket_id: str, agent: str, action: str, reasoning: str, confidence: float) -> str:
    """Save agent decision to audit log."""
    session = Session()
    log = AuditLog(
        ticket_id=ticket_id,
        agent=agent,
        action=action,
        reasoning=reasoning,
        confidence=confidence
    )
    session.add(log)
    session.commit()
    session.close()
    return f"Logged: {action} by {agent}"

def get_tools():
    return [search_knowledge_base, calculate_confidence, save_audit_log]

def run_classifier(state: dict, llm) -> dict:
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="""You are an IT support ticket classifier.
Follow these steps EXACTLY ONCE in order:
1. Call search_knowledge_base ONCE
2. Call calculate_confidence ONCE with the distances
3. Call save_audit_log ONCE
4. Return JSON only: {"category": "...", "sentiment": "..."}
Do NOT repeat any tool call."""),
        HumanMessage(content=f"""Classify this ticket:
Title: {state['title']}
Description: {state['description']}
Ticket ID: {state['ticket_id']}

Respond with JSON only: {{"category": "...", "sentiment": "..."}}""")
    ]

    similar_tickets = []
    confidence = 0.0

    for _ in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"[Classifier] Calling tool: {tool_name}")
            if tool_name == "search_knowledge_base":
                result = search_knowledge_base.invoke(tool_args)
                similar_tickets = json.loads(result)
            elif tool_name == "calculate_confidence":
                distances = [t['distance'] for t in similar_tickets]
                result = calculate_confidence.invoke({"distances_json": json.dumps(distances)})
                confidence = result
            elif tool_name == "save_audit_log":
                tool_args['ticket_id'] = state['ticket_id']
                tool_args['confidence'] = confidence
                result = save_audit_log.invoke(tool_args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

    try:
        final_text = response.content.strip().replace('```json', '').replace('```', '')
        extracted = json.loads(final_text)
        category = extracted.get('category', 'Other')
        sentiment = extracted.get('sentiment', 'neutral')
    except:
        category = similar_tickets[0].get('queue', 'Other') if similar_tickets else 'Other'
        sentiment = 'neutral'

    print(f"[Classifier] Category: {category} | Confidence: {confidence:.2f} | Sentiment: {sentiment}")
    return {
        "category": category,
        "confidence_score": confidence,
        "similar_tickets": similar_tickets
    }