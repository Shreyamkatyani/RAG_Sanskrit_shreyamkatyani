import os
import sys


from pypdf import PdfReader
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# Configuration
DATA_DIR = "../data"
DB_DIR = "../db_storage"
# Updated paths to use local models
MODEL_NAME = "local_llm/orca-mini-3b-gguf2-q4_0.gguf" 
EMBEDDING_MODEL = "local_embedding_model"
COLLECTION_NAME = "sanskrit_docs"

def load_documents(data_path: str) -> List[str]:
    """Load text and PDF documents from the data directory."""
    documents = []
    if not os.path.exists(data_path):
        print(f"Error: Data directory '{data_path}' not found.")
        return documents

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract:
                        text += extract + "\n"
                documents.append(text)
            except Exception as e:
                print(f"Failed to load PDF {filename}: {e}")
        elif filename.endswith(".txt"):
            print(f"Loading Text: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Failed to load TXT {filename}: {e}")
    
    return documents

def split_text(documents: List[str], chunk_size=200, overlap=50) -> List[str]:
    """Simple text splitter."""
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size - overlap):
            chunk = doc[i : i + chunk_size]
            if len(chunk) > 50: # Ignore tiny chunks
                chunks.append(chunk)
    print(f"Created {len(chunks)} chunks.")
    return chunks



def setup_vector_store(chunks: List[str]):
    """Initialize ChromaDB and index chunks."""
    print("Initializing Vector Store...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Get or create collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    if collection.count() == 0:
        print("Collection is empty. Embedding documents...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embedder.encode(chunks).tolist()
        
        ids = [str(i) for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print("Indexing complete.")
    else:
        print(f"Loaded existing collection with {collection.count()} documents.")
        
    return collection, client

def retrieve_context(collection, query: str, n_results=3) -> str:
    """Retrieve relevant chunks for a query."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embed = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embed,
        n_results=n_results
    )
    
    context = "\n\n".join(results['documents'][0])
    return context

def generate_answer(llm, context: str, query: str):
    """Generate answer using CPU LLM."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (in simple English or Sanskrit):"
    
    # Simple generation
    output = llm.generate(prompt, max_tokens=200)
    return output

def main():
    print("=== Sanskrit RAG System (CPU) ===")
    
    # 1. Ingest
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Please add files to 'data/' folder.")
        # Proceeding anyway to allow query if DB exists, but usually we need data.
        # If DB exists we might skip ingestion, but logic here assumes ingestion checks data folder.
    
    # 2. Process & Index
    if docs:
        chunks = split_text(docs)
        collection, _ = setup_vector_store(chunks)
    else:
        # Try finding existing DB
        client = chromadb.PersistentClient(path=DB_DIR)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print("Loaded existing DB.")
        except:
            print("No DB and no Data. Exiting.")
            return

    # 3. Load LLM
    print(f"Loading LLM model: {MODEL_NAME}...")
    # Correct way for local file:
    model_dir = os.path.dirname(MODEL_NAME)
    model_file = os.path.basename(MODEL_NAME)
    
    llm = GPT4All(model_file, model_path=model_dir, allow_download=False)
    
    # 4. Loop
    print("\nSystem Ready. Enter your query (type 'exit' to quit).")
    while True:
        query = input("\nQuery: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("Retrieving context...")
        context = retrieve_context(collection, query, n_results=1)
        
        print("Generating answer...")
        answer = generate_answer(llm, context, query)
        
        print(f"\nResponse:\n{answer}")

if __name__ == "__main__":
    main()
