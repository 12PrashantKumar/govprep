import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"

def get_collection():
    print("Connecting to Chroma DB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(name="govprep_multi", embedding_function=ef)

def retrieve(query, k=4,source_filter=None):
    """
    Retrieve top-k chunks. 
    Optionally pass source_filter="polity" to restrict search.
    """
    collection = get_collection()
    
    # Only build the WHERE clause if a filter was actually provided
    where_clause = {"Source":source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_clause
    )
    
    chunks = []

    if not results["documents"]:
        print("No results found.")
        return chunks
    
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]

        })
    return chunks

if __name__ == "__main__":
    tests = [
        "what are fundamental rights",      # Expect: polity
        "physical features of india",       # Expect: geography
        "ancient indian history",           # Expect: history
    ]
    for  q in tests:
        print(f"\n{'='*60}\nQUERY: '{q}'\n{'='*60}")
        results = retrieve(q)
        
        if not results:
            print("No results found.")
            continue
        else:
            for c in results:
                # Beautifully formatting the provenance data
                print(f"[{c['source'].upper()} - Page {c['page']} | dist={c['distance']:.3f}]")
                print(f"{c['text'][:120].strip()}...\n")

    
