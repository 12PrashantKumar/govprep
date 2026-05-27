import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# DATABASE CONNECTION
# ==========================================
def get_collection():
    """Connect to existing ChromaDB and return the collection."""
    # Pointing to the local folder we fixed yesterday
    client = chromadb.PersistentClient(path="./db")
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Connecting to the exact table you created in the ingestion script
    return client.get_or_create_collection(
        name="ncert_polity_ch01",
        embedding_function=embedding_fn,
    )

# ==========================================
# RETRIEVAL LOGIC (THE 'R' IN RAG)
# ==========================================
def retrieve(query, k=3):
    """
    Retrieve top-k most relevant chunks for a query.
    Returns: [{"text": ..., "distance": ..., "metadata": ...}, ...]
    """
    collection = get_collection()
    
    # Search the database
    results = collection.query(query_texts=[query], n_results=k)
    
    # Unpack ChromaDB's nested arrays into a clean, standard Python list
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "distance": results['distances'][0][i],
            "metadata": results['metadatas'][0][i],
        })
    return chunks

# ==========================================
# UI HELPER
# ==========================================
def format_chunks_for_display(chunks):
    """Pretty-print chunks for human reading."""
    output = []
    for i, chunk in enumerate(chunks):
        output.append(f"--- Chunk {i+1} (dist: {chunk['distance']:.4f}) ---")
        output.append(chunk['text'].strip())
        output.append("")
    return "\n".join(output)

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Test queries matching your NCERT Polity data
    test_queries = [
        "what is the constitution of india",
        "what are the fundamental rights in india",
        "who is the head of state in india",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        # 1. Fetch the data using our clean abstraction
        retrieved_chunks = retrieve(query, k=3)
        
        # 2. Print it to the terminal beautifully
        print(format_chunks_for_display(retrieved_chunks))