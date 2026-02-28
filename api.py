from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import uuid
import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pandas as pd
import io
sys.path.append(os.path.dirname(__file__))

from db.models import Session, Ticket, AuditLog
from db.vector_store import load_knowledge_base
from graph.ticket_graph import build_pipeline
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Global pipeline
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    load_knowledge_base()
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    pipeline = build_pipeline(llm)
    print("Pipeline ready")
    yield

app = FastAPI(title="TicketMind AI API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TicketRequest(BaseModel):
    title: str
    description: str
    priority: str = "medium"

    @field_validator('title')
    @classmethod
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v

    @field_validator('priority')
    @classmethod
    def valid_priority(cls, v):
        if v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError('Priority must be low/medium/high/critical')
        return v

class ResolveRequest(BaseModel):
    resolution: str

    @field_validator('resolution')
    @classmethod
    def resolution_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Resolution cannot be empty')
        return v

# ─── ENDPOINTS ───

@app.get("/")
async def root():
    return {"message": "TicketMind AI API", "docs": "/docs"}

@app.post("/ticket")
async def submit_ticket(request: TicketRequest):
    try:
        ticket_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())

        result = pipeline.invoke(
            {
                "ticket_id": ticket_id,
                "title": request.title,
                "description": request.description,
                "category": "",
                "priority": request.priority,
                "confidence_score": 0.0,
                "similar_tickets": [],
                "suggested_resolution": "",
                "action": "",
                "explanation": "",
                "resolved": False
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        # Priority override — critical always escalates
        if request.priority == "critical" and result.get('action') == 'auto_resolve':
            result['action'] = 'escalate'
            result['resolved'] = False
            result['explanation'] = "Critical priority — escalated to human regardless of confidence score."

        action = result.get('action', '')
        confidence = result.get('confidence_score', 0.0)

        # If graph paused at interrupt — save pending record
        if action == '':
            session = Session()
            pending = Ticket(
                id=ticket_id,
                title=request.title,
                description=request.description,
                category=result.get('category', 'Unknown'),
                priority=request.priority,
                confidence_score=confidence,
                action="escalate",
                suggested_resolution="AI analysis in progress — awaiting human review",
                explanation="Escalated — awaiting human review",
                resolved=False,
                thread_id=thread_id
            )
            session.merge(pending)
            session.commit()
            session.close()

            return {
                "ticket_id": ticket_id,
                "thread_id": thread_id,
                "action": "escalate",
                "confidence_score": confidence,
                "category": result.get('category', 'Unknown'),
                "suggested_resolution": "Awaiting human review",
                "resolved": False,
                "explanation": "Escalated — awaiting human review"
            }

        # Save thread_id for auto-resolved tickets
        session = Session()
        ticket_db = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket_db:
            ticket_db.thread_id = thread_id
            session.commit()
        session.close()

        return {
            "ticket_id": ticket_id,
            "thread_id": thread_id,
            "action": result.get('action', ''),
            "confidence_score": result.get('confidence_score', 0.0),
            "category": result.get('category', ''),
            "suggested_resolution": result.get('suggested_resolution', ''),
            "resolved": result.get('resolved', False),
            "explanation": result.get('explanation', '')
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tickets")
async def get_tickets():
    try:
        session = Session()
        tickets = session.query(Ticket).order_by(Ticket.created_at.desc()).all()
        session.close()
        return [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "category": t.category,
                "priority": t.priority,
                "confidence_score": t.confidence_score,
                "action": t.action,
                "suggested_resolution": t.suggested_resolution,
                "final_resolution": t.final_resolution,
                "explanation": t.explanation,
                "resolved": t.resolved,
                "created_at": str(t.created_at),
                "resolved_at": str(t.resolved_at) if t.resolved_at else None,
                "thread_id": t.thread_id
            }
            for t in tickets
        ]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tickets/escalated")
async def get_escalated():
    try:
        session = Session()
        tickets = session.query(Ticket).filter(
            Ticket.action == 'escalate',
            Ticket.resolved == False
        ).order_by(Ticket.created_at.desc()).all()
        session.close()
        return [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "category": t.category,
                "priority": t.priority,
                "confidence_score": t.confidence_score,
                "action": t.action,
                "suggested_resolution": t.suggested_resolution,
                "explanation": t.explanation,
                "resolved": t.resolved,
                "created_at": str(t.created_at),
                "thread_id": t.thread_id
            }
            for t in tickets
        ]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: str, request: ResolveRequest):
    try:
        session = Session()
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        thread_id = ticket.thread_id

        if thread_id:
            from langgraph.types import Command
            result = pipeline.invoke(
                Command(resume={"resolution": request.resolution}),
                config={"configurable": {"thread_id": thread_id}}
            )
            print(f"[API] Graph resumed. Result action: {result.get('action')}")
        else:
            from agents.learning_agent import embed_human_resolution
            embed_human_resolution(
                ticket_id=ticket_id,
                title=ticket.title,
                description=ticket.description,
                category=ticket.category or 'Other',
                priority=ticket.priority or 'medium',
                resolution=request.resolution
            )

        # Update using SQLAlchemy
        ticket.resolved = True
        ticket.final_resolution = request.resolution
        ticket.resolved_at = datetime.now()
        session.commit()
        session.close()

        return {"status": "resolved", "ticket_id": ticket_id}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit")
async def get_audit():
    try:
        session = Session()
        logs = session.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
        session.close()
        return [
            {
                "id": l.id,
                "ticket_id": l.ticket_id,
                "agent": l.agent,
                "action": l.action,
                "reasoning": l.reasoning,
                "confidence": l.confidence,
                "timestamp": str(l.timestamp)
            }
            for l in logs
        ]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        session = Session()
        total = session.query(Ticket).count()
        auto_resolved = session.query(Ticket).filter(Ticket.action == 'auto_resolve').count()
        escalated = session.query(Ticket).filter(Ticket.action == 'escalate').count()
        pending = session.query(Ticket).filter(
            Ticket.action == 'escalate',
            Ticket.resolved == False
        ).count()
        session.close()

        return {
            "total": total,
            "auto_resolved": auto_resolved,
            "escalated": escalated,
            "pending_review": pending,
            "resolution_rate": round(auto_resolved / total * 100, 1) if total > 0 else 0
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    source_name: str = Form(default="uploaded_dataset")
):
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Detect encoding and parse CSV
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")
        
        print(f"[API] Dataset uploaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"[API] Columns: {df.columns.tolist()}")
        
        # Use LLM to detect mapping
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        from utils.dataset_adapter import detect_column_mapping, validate_mapping, load_dataset
        
        mapping = detect_column_mapping(df, llm)
        
        valid, message = validate_mapping(mapping, df)
        if not valid:
            raise HTTPException(status_code=400, detail=f"Invalid mapping: {message}")
        
        # Load dataset into ChromaDB
        result = load_dataset(df, mapping, source_name)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error'))
        
        return {
            "success": True,
            "filename": file.filename,
            "source_name": source_name,
            "rows_in_file": len(df),
            "columns_detected": df.columns.tolist(),
            "mapping_detected": mapping,
            "processed": result['processed'],
            "skipped": result['skipped'],
            "tickets_added": result['tickets_added'],
            "kb_before": result['kb_before'],
            "kb_after": result['kb_after']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))