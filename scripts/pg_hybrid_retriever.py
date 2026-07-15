import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def hybrid_retriever(query:str,k:int =5) -> list[Document]:
    """
    Executes a Dense (Vector) search and a Sparse ( BM25 Keyword) search, 
    then combines the results mathematically using Reciprocal Rank Fusion (RRF).
    """
    print(f"🔍 Running Hybrid Search for: '{query}'")

    # embeding the query for vector search
    
    query_vector = EMBEDDING_MODEL.embed_query(query)

    #  PostgreSQL Strict 1-D Safeguard 
    # If LangChain returns a nested list [[0.1...]], extract the inner list
    if isinstance(query_vector, list) and isinstance(query_vector[0], list):
        query_vector = query_vector[0]
    # If LangChain returns a multidimensional numpy array, flatten it to a standard list
    elif hasattr(query_vector, "tolist"):
        import numpy as np
        query_vector = np.array(query_vector).flatten().tolist()

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:

            # A. DENSE SEARCH (VECTOR) 
            # Fetches the top 20 conceptual matches
            cur.execute("""
                    SELECT content,metadata, subject, source
                        FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT 20;
                    """,(query_vector,))
            dense_results = cur.fetchall()

            # B .SPARSE (BM25) SEARCH (exact keyword match)
            # Fetches the top 20 exact keyword matches using Postgres Full-Text Search
            cur.execute("""
                        SELECT content,metadata,subject,source
                        FROM chunks
                        WHERE search_vector @@ websearch_to_tsquery('english',%s)
                        LIMIt 20;
                        """,(query,))
            sparse_results = cur.fetchall()


    # RECIPROCAL RANK FUSION (RRF)
    # We use a dictionary to keep track of scores. 
    # The chunk's text (content) acts as the unique identifier.

    rrf_scores = {}
    doc_store = {}
    k_rrf = 60   # industry standard constant for rrf


    # Score the Dense Results
    for rank, row in enumerate(dense_results):
        content = row[0]
        doc_store[content] = row
        # RRF Formula: 1/ (rank +60)
        rrf_scores[content] = rrf_scores.get(content,0.0) + (1.0/ (rank+1+k_rrf))

    
    # Score the Sparse Results (BM25)
    for rank, row in enumerate(sparse_results):
        content = row[0]
        doc_store[content] = row
        # If a chunk is found in BOTH searches, its score compounds here!
        rrf_scores[content] = rrf_scores.get(content,0.0) + (1.0 /(rank + 1 + k_rrf))

#   SORT AND FORMAT 
    # Sort the chunks by their final RRF score from highest to lowest
    sorted_docs = sorted(rrf_scores.items(), key = lambda item: item[1], reverse =True)

    # Format the absolute top K results into LangChain Documents
    documents = []
    for content, score in sorted_docs[:k]:
        row = doc_store[content]

        # row[1] is metadata (already a dictionary in psycopg)
        metadata = row[1] if row[1] else {}
        metadata["subject"] = row[2]
        metadata["source"] = row[3]
        metadata["rrf_score"] = round(score,4)   # We attach the score for observability

        doc = Document(page_content=content, metadata = metadata)
        documents.append(doc)

    return documents

# Testing ---
if __name__ == "__main__":
    test_query = "fundamental rights"
    docs = hybrid_retriever(test_query, k=3)
    
    print(f"\n✅ Hybrid Fusion Complete! Top {len(docs)} Results:\n")
    for i, doc in enumerate(docs):
        print(f"--- Rank {i+1} (RRF Score: {doc.metadata.get('rrf_score')}) ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:150]}...\n")

            


