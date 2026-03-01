import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
CHROMA_TENANT = os.getenv('CHROMA_TENANT')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE')

# Use cloud if credentials exist, otherwise fall back to local
if CHROMA_API_KEY:
    print("Using ChromaDB Cloud")
    chroma_client = chromadb.HttpClient(
        ssl=True,
        host='api.trychroma.com',
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        headers={
            'x-chroma-token': CHROMA_API_KEY
        }
    )
else:
    print("Using ChromaDB Local")
    CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chromadb')
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

ticket_collection = chroma_client.get_or_create_collection(name="ticket_kb")

def get_collection():
    return ticket_collection

def load_knowledge_base():
    count = ticket_collection.count()
    if count == 0:
        print("KB empty — loading from pkl...")
        import pickle
        import os as _os
        pkl_path = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'ticketmind_embeddings.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        batch_size = 100
        docs = data['documents']
        metas = data['metadatas']
        ids = data['ids']
        embeddings = data['embeddings']

        for i in range(0, len(docs), batch_size):
            ticket_collection.add(
                embeddings=embeddings[i:i+batch_size],
                documents=docs[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            print(f"Loaded batch {i//batch_size + 1}/{len(docs)//batch_size + 1}")
        print(f"KB loaded: {ticket_collection.count()} tickets")
    else:
        print(f"KB already loaded: {count} tickets")


def retag_existing_tickets():
    """Add domain tags to existing tickets that don't have them"""
    try:
        collection = get_collection()
        total = collection.count()
        print(f"[Retag] Total tickets: {total}")

        batch_size = 300
        offset = 0
        ids_to_update = []
        metas_to_update = []

        while offset < total:
            results = collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset
            )
            
            for i, meta in enumerate(results['metadatas']):
                if 'domain' not in meta or not meta['domain']:
                    ticket_id = results['ids'][i]
                    source = meta.get('source', '')
                    
                    source_lower = source.lower()
                    doc_text = meta.get('subject', '') + ' ' + meta.get('body', '')
                    doc_lower = doc_text.lower()

                    if any(k in source_lower for k in ['kaggle', 'customer', 'support_ticket', 'ecommerce', 'retail']):
                        domain = 'customer_support'
                    elif any(k in doc_lower for k in ['refund', 'return', 'exchange', 'order', 'shipping', 'billing', 'invoice', 'payment', 'purchase']):
                        domain = 'customer_support'
                    elif any(k in doc_lower for k in ['network', 'vpn', 'outlook', 'windows', 'software', 'hardware', 'printer', 'server', 'email', 'computer', 'laptop', 'wifi', 'password', 'login', 'access']):
                        domain = 'IT_support'
                    else:
                        domain = 'general'

                    updated_meta = dict(meta)
                    updated_meta['domain'] = domain
                    ids_to_update.append(ticket_id)
                    metas_to_update.append(updated_meta)

            offset += batch_size
            print(f"[Retag] Scanned {min(offset, total)}/{total}")

        if not ids_to_update:
            print("[Retag] All tickets already have domain tags")
            return

        print(f"[Retag] Updating {len(ids_to_update)} tickets...")
        
        update_batch = 100
        for i in range(0, len(ids_to_update), update_batch):
            collection.update(
                ids=ids_to_update[i:i+update_batch],
                metadatas=metas_to_update[i:i+update_batch]
            )
            print(f"[Retag] Updated {min(i+update_batch, len(ids_to_update))}/{len(ids_to_update)}")

        print(f"[Retag] Done — {len(ids_to_update)} tickets tagged")

    except Exception as e:
        print(f"[Retag] Error: {e} — continuing without retagging")