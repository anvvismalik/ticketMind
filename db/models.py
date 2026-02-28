from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class Ticket(Base):
    __tablename__ = 'tickets'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String)
    priority = Column(String)
    sentiment = Column(String)
    confidence_score = Column(Float, default=0.0)
    action = Column(String)
    suggested_resolution = Column(Text)
    final_resolution = Column(Text)
    explanation = Column(Text)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    thread_id = Column(String)

class AuditLog(Base):
    __tablename__ = 'audit_log'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(String, nullable=False)
    agent = Column(String)
    action = Column(String)
    reasoning = Column(Text)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Use PostgreSQL if DATABASE_URL exists, otherwise fallback to SQLite
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    print("Using PostgreSQL")
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'ticketmind.db')
    engine = create_engine(f'sqlite:///{DB_PATH}')
    print("Using SQLite")

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)