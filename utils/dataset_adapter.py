import pandas as pd
import json
import re
from langchain_groq import ChatGroq
from db.vector_store import get_collection, embedder
from db.models import Session, AuditLog
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


def detect_column_mapping(df: pd.DataFrame, llm) -> dict:
    """Use LLM to map CSV columns to required fields: subject, body, answer"""
    columns = df.columns.tolist()
    sample = df.head(3).to_dict(orient='records')

    sample_clean = []
    for row in sample:
        clean_row = {k: str(v)[:100] for k, v in row.items()}
        sample_clean.append(clean_row)

    prompt = f"""You are analyzing a CSV file from an IT support system.

The CSV has these columns:
{columns}

Here are 3 sample rows:
{json.dumps(sample_clean, indent=2)}

Your job is to map these columns to 3 required fields:
- "subject": the ticket title or issue summary (short)
- "body": the full ticket description or details (longer)  
- "answer": the resolution or solution provided

Rules:
- Each required field must map to exactly one CSV column
- Choose the BEST matching column for each field
- If there is no good match for a field, use the closest available column
- Return ONLY a valid JSON object on a single line, nothing else, no explanation

Example output:
{{"subject": "ticket_title", "body": "description", "answer": "resolution_notes"}}

Return the JSON mapping now:"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Clean markdown
    raw = re.sub(r'```json|```', '', raw).strip()

    # Extract just the JSON object — find first { and last }
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    print(f"[Adapter] Raw LLM response: {raw}")
    mapping = json.loads(raw)
    print(f"[Adapter] Mapping detected: {mapping}")
    return mapping


def detect_dataset_domain(df: pd.DataFrame, llm) -> str:
    """Use LLM to detect what domain/industry this dataset is from"""
    sample = df.head(5).to_dict(orient='records')
    sample_clean = [{k: str(v)[:100] for k, v in row.items()} for row in sample]

    prompt = f"""Analyze these support ticket samples and identify the domain.

Sample tickets:
{json.dumps(sample_clean, indent=2)}

Choose the best domain from:
- IT_support (computer, network, software, hardware issues)
- customer_support (product issues, refunds, billing, orders)
- HR_support (employee, payroll, benefits, leave)
- finance_support (invoices, payments, accounting)
- general (mixed or unclear)

Return ONLY the domain name, nothing else. No explanation."""

    response = llm.invoke(prompt)
    domain = response.content.strip().split()[0]
    # Clean any punctuation
    domain = re.sub(r'[^a-zA-Z_]', '', domain)
    print(f"[Adapter] Domain detected: {domain}")
    return domain


def validate_mapping(mapping: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that mapped columns exist in dataframe"""
    required = ['subject', 'body', 'answer']
    for field in required:
        if field not in mapping:
            return False, f"Missing mapping for '{field}'"
        col = mapping[field]
        if col not in df.columns:
            return False, f"Column '{col}' not found in CSV"
    return True, "OK"


def load_dataset(df: pd.DataFrame, mapping: dict, source_name: str = "uploaded", domain: str = "general") -> dict:
    """Apply mapping and embed all rows into ChromaDB"""

    collection = get_collection()
    kb_size_before = collection.count()

    subject_col = mapping['subject']
    body_col = mapping['body']
    answer_col = mapping['answer']

    # Prepare data
    documents = []
    metadatas = []
    ids = []

    skipped = 0
    processed = 0

    for idx, row in df.iterrows():
        subject = str(row.get(subject_col, '')).strip()
        body = str(row.get(body_col, '')).strip()
        answer = str(row.get(answer_col, '')).strip()

        # Skip empty rows
        if not subject or not answer or subject == 'nan' or answer == 'nan':
            skipped += 1
            continue

        # Create document text
        doc_text = f"Issue: {subject}\nDetails: {body}\nResolution: {answer}"

        documents.append(doc_text)
        metadatas.append({
            'subject': subject[:500],
            'body': body[:1000],
            'answer': answer[:1000],
            'source': source_name,
            'ticket_id': f"{source_name}_{idx}",
            'domain': domain
        })
        ids.append(f"{source_name}_{idx}_{uuid.uuid4().hex[:8]}")
        processed += 1

    if not documents:
        return {
            "success": False,
            "error": "No valid rows found after filtering empty entries",
            "processed": 0,
            "skipped": skipped
        }

    # Embed in batches of 100
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"[Adapter] Embedding {len(documents)} tickets in {total_batches} batches...")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        batch_embeddings = embedder.encode(batch_docs).tolist()

        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

        batch_num = (i // batch_size) + 1
        print(f"[Adapter] Batch {batch_num}/{total_batches} done")

    kb_size_after = collection.count()

    # Log to audit
    session = Session()
    log = AuditLog(
        ticket_id=f"dataset_{source_name}",
        agent="dataset_adapter",
        action="dataset_loaded",
        reasoning=f"Loaded {processed} tickets from '{source_name}' domain='{domain}'. KB: {kb_size_before} → {kb_size_after}",
        confidence=1.0
    )
    session.add(log)
    session.commit()
    session.close()

    print(f"[Adapter] Done. KB: {kb_size_before} → {kb_size_after}")

    return {
        "success": True,
        "processed": processed,
        "skipped": skipped,
        "kb_before": kb_size_before,
        "kb_after": kb_size_after,
        "tickets_added": kb_size_after - kb_size_before,
        "mapping_used": mapping,
        "domain": domain
    }