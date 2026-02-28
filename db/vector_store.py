import chromadb
from chromadb.auth.token_authn import TokenAuthClientProvider
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
    import os
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