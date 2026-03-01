---
title: TicketMind API
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# TicketMind AI

> **AI-powered multi-agent IT support system that automatically classifies, resolves, and learns from support tickets using RAG, LangGraph, and Human-in-the-Loop workflows.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-blue?style=flat)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Cloud-orange?style=flat)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-336791?style=flat&logo=postgresql)](https://supabase.com)

---

## Live Demo

| Service | URL |
|---|---|
| Dashboard | [ticketmind-j8agjxbnhscsxlfvvozpdo.streamlit.app](https://ticketmind-j8agjxbnhscsxlfvvozpdo.streamlit.app) |
| API | [anvviiii-ticketmind-api.hf.space](https://anvviiii-ticketmind-api.hf.space) |
| API Docs | [anvviiii-ticketmind-api.hf.space/docs](https://anvviiii-ticketmind-api.hf.space/docs) |

---

## What It Does

TicketMind accepts support tickets from any company, automatically:

1. **Classifies** the ticket by category and domain (IT, customer support, HR, finance)
2. **Searches** a knowledge base of 7,700+ past resolved tickets using vector similarity
3. **Auto-resolves** high-confidence tickets with RAG-grounded resolutions
4. **Escalates** uncertain tickets to human agents for review
5. **Learns** from every human-approved resolution, growing smarter over time

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         LangGraph Pipeline           │
                        │                                      │
  Ticket Input  ──────► │  ┌─────────────┐                    │
                        │  │  Classifier  │ ── domain detect   │
                        │  │    Agent     │ ── KB search       │
                        │  │             │ ── confidence score │
                        │  └──────┬──────┘                    │
                        │         │                            │
                        │    confidence?                       │
                        │   >= 0.55 │ < 0.55                  │
                        │         ▼         ▼                  │
                        │  ┌──────────┐ ┌──────────┐          │
                        │  │Resolution│ │Escalation│          │
                        │  │  Agent   │ │  Agent   │          │
                        │  │  (RAG)   │ │  (HITL)  │──► Human │
                        │  └────┬─────┘ └────┬─────┘  Review  │
                        │       │             │                │
                        │       └──────┬──────┘               │
                        │              ▼                       │
                        │      ┌──────────────┐               │
                        │      │   Learning   │               │
                        │      │    Agent     │ ── embed to KB │
                        │      └──────────────┘               │
                        └─────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ChromaDB Cloud           PostgreSQL                  Streamlit
      (Vector KB 7,776         (Ticket records,           (Dashboard +
        tickets)                audit logs)               Human Queue)
```

---

## Agent Details

### 1. Classifier Agent
Uses LangChain tool calling to:
- Detect ticket domain (IT, customer support, HR, finance)
- Search ChromaDB with domain-filtered vector similarity
- Compute confidence score from embedding distances
- Log decision to audit trail

### 2. Resolution Agent (True RAG)
- Retrieves top 3 similar past tickets from ChromaDB
- Injects retrieved resolutions as primary context
- LLM generates resolution grounded in past cases
- Explicitly references which past case it based the answer on

### 3. Escalation Agent (HITL)
- Generates AI analysis before pausing (summary, probable cause, first step)
- Uses LangGraph `interrupt()` to pause graph execution
- Human reviews in dashboard and provides resolution
- Graph resumes via `Command(resume=...)` pattern

### 4. Learning Agent
- Embeds every resolved ticket back into ChromaDB
- KB grows with each human-approved resolution
- System improves over time without retraining

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Agent Orchestration | LangGraph | Stateful multi-agent graphs with HITL support |
| LLM | Groq (llama-3.3-70b) | Fast inference, tool calling, free tier |
| Vector DB | ChromaDB Cloud | Persistent embeddings, metadata filtering |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, accurate semantic search |
| API | FastAPI | Async, auto-docs, production ready |
| Database | PostgreSQL (Supabase) | Ticket records, audit logs |
| Dashboard | Streamlit | Rapid UI, Plotly charts |
| API Hosting | Hugging Face Spaces | Free ML-optimized hosting |
| Dashboard Hosting | Streamlit Cloud | Free, auto-deploy from GitHub |

---

## Key Features

- **Smart Dataset Adapter** — Upload any CSV format. LLM automatically maps columns to required fields and detects domain. No manual configuration needed.
- **Domain-Tagged Knowledge Base** — IT tickets search only IT past cases. Customer support tickets search only customer support cases. Prevents cross-domain confidence dilution.
- **True RAG Resolutions** — Resolution explicitly references which past case it is based on, with similarity distance shown.
- **HITL Interrupt Pattern** — Graph execution pauses at escalation, resumes after human approval. Thread state preserved via LangGraph MemorySaver.
- **Critical Priority Override** — Critical tickets always escalate to human regardless of confidence score.
- **Full Audit Trail** — Every agent decision logged to PostgreSQL with confidence scores and reasoning.
- **Self-Improving KB** — Each human-approved resolution embedded back into ChromaDB. System gets smarter with every ticket.

---

## Knowledge Base

| Source | Tickets | Domain |
|---|---|---|
| Original IT Support Dataset | 5,006 | IT_support |
| Kaggle Customer Support Dataset | 2,770 | customer_support |
| **Total** | **7,776** | Mixed |

---

## Project Structure

```
ticketMind/
├── agents/
│   ├── classifier_agent.py   # Domain detection, KB search, confidence
│   ├── resolution_agent.py   # RAG-grounded resolution generation
│   ├── escalation_agent.py   # HITL interrupt/resume pattern
│   └── learning_agent.py     # KB embedding, self-improvement
├── graph/
│   └── ticket_graph.py       # LangGraph StateGraph, routing logic
├── db/
│   ├── models.py             # SQLAlchemy models (Ticket, AuditLog)
│   └── vector_store.py       # ChromaDB client, domain retagging
├── utils/
│   └── dataset_adapter.py    # Smart CSV ingestion, LLM column mapping
├── dashboard/
│   └── app.py                # Streamlit UI, human queue, analytics
├── api.py                    # FastAPI endpoints
├── Dockerfile                # HF Spaces deployment
└── requirements.txt
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/ticket` | POST | Submit a new support ticket |
| `/tickets` | GET | List all tickets |
| `/tickets/escalated` | GET | Get tickets awaiting human review |
| `/tickets/{id}/resolve` | POST | Human resolves an escalated ticket |
| `/stats` | GET | Dashboard statistics |
| `/audit` | GET | Full agent audit trail |
| `/upload-dataset` | POST | Upload CSV to expand knowledge base |

---

## Known Limitations & Future Improvements

- **MemorySaver vs PostgreSQL checkpointer** — HITL thread state stored in RAM. Server restart loses pending escalation threads (graceful fallback exists). Future: migrate to `langgraph-checkpoint-postgres`.
- **Dynamic confidence thresholds** — Currently fixed at 0.55. Future: auto-calibrate per domain based on KB distribution statistics.
- **Per-category routing** — Future: route tickets to specialized agent queues (network team, software team, etc.) based on category.
- **Feedback loop** — Future: track resolution quality scores to weight KB entries.