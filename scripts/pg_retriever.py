import os
import json
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def postgres_retriever(query:str, k:int = 5) ->list[Document]:
    """
    Takes a user query, turns it into a vector, and fetches the top K 
    closest chunks from PostgreSQL using cosine distance.
    """
    print(f"🔍 Searching Postgres for: '{query}'")
    
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model = "gemini-embedding-2",
        output_dimensionality=768
    )
    query_vector  = embeddings_model.embed_query(query)

    # Search Postgres using vector math (<=> is the cosine distance operator)
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # We select the content and metadata, ordering by closest vector match
            cur.execute("""
                    SELECT content, metadata, subject, source
                        FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;

        """,(query_vector,k))

            results = cur.fetchall()
    
#    Format into LangChain Documents to maintain parity with the old system
    documents = []
    for row in results:
        content = row[0]
        
        metadata = row[1] if row[1] else {}

        # Add our custom Postgres columns back into the metadata
        metadata["subject"] = row[2]
        metadata["source"] = row[3]

        # Create the LangChain Document object
        doc = Document(page_content=content,metadata=metadata)
        documents.append(doc)
    
    return documents

#testing
if __name__ == "__main__":
    test_query = "What are the fundamental rights of a citizen?"
    retrieved_docs = postgres_retriever(test_query, k=3)


    print(f"\n✅ Successfully retrieved {len(retrieved_docs)} documents!\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:200]}...\n")



    