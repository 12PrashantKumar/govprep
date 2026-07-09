import os
import re
import json
import glob
import time
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

#  THE DATA SCRUBBER 
def scrub_ncert_text(raw_text: str) -> str:
    """Cleans dirty PDF text before it gets chunked and embedded."""
    # 1. Remove standalone page numbers
    clean_text = re.sub(r'(?m)^\s*\d+\s*$', '', raw_text)
    # 2. Remove common NCERT headers
    clean_text = re.sub(r'Indian Constitution at Work', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'Chapter \d+.*', '', clean_text, flags=re.IGNORECASE)
    # 3. Fix fragmented sentences (replace single newlines with space)
    clean_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', clean_text)
    # 4. Strip out multiple consecutive spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def ingest_folder(folder_path, subject):
    print(f"Starting ingestion for {subject} from '{folder_path}'....")

    # 1. Initialize Gemini Embeddings (768 dimensions)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=768
    )

    # 2. Initialize Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 3. Find all PDFs in the target folder
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    if not pdf_files:
        print(f" No PDFs found in {folder_path}")
        return
    
    # 4. Connect to PostgreSQL
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            for pdf_file in pdf_files:
                print(f"\nProcessing: {pdf_file}")

                # Step A: Load PDF
                loader = PyPDFLoader(pdf_file)
                raw_docs = loader.load()

                #  APPLY SCRUBBER
                for doc in raw_docs:
                    doc.page_content = scrub_ncert_text(doc.page_content)

                # Step B: Chunk text
                chunks = text_splitter.split_documents(raw_docs)
                print(f" -> Created {len(chunks)} clean chunks. Asking Gemini for embeddings...")

                # Step C: Get Embeddings
                text_strings = [chunk.page_content for chunk in chunks]
                vectors = []

                batch_size = 10 # only 10 chunks send to api at a time
                for i in range(0, len(text_strings), batch_size):
                    batch = text_strings[i:i + batch_size]
                    print(f"    -> Embedding batch {i//batch_size + 1} (Chunks {i} to {min(i+batch_size, len(text_strings))})...")

                    try:
                        batch_vectors = embeddings_model.embed_documents(batch)
                        vectors.extend(batch_vectors)
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            print("    Hit API limit! Sleeping for 30 seconds to clear quota...")
                            time.sleep(30)
                            # Retry the batch after sleeping (Fixed to synchronous embed_documents)
                            batch_vectors = embeddings_model.embed_documents(batch)
                            vectors.extend(batch_vectors)
                        else:
                            raise e
                    time.sleep(5)

                # Step D: Prepare for Database Insert 
                insert_query = """
                INSERT INTO chunks (content, subject, source, metadata, embedding, search_vector)
                VALUES(%s, %s, %s, %s, %s, to_tsvector('english', %s));
                """

                data_to_insert = []
                source_name = os.path.basename(pdf_file)

                for i, chunk in enumerate(chunks):
                    data_to_insert.append((
                        chunk.page_content,
                        subject,
                        source_name,
                        json.dumps(chunk.metadata), # convert LangChain dict to JSON string for Postgres
                        vectors[i],
                        chunk.page_content          # Passing content again to generate the keyword search vector
                    ))

                # Step E: Batch Execute and commit
                print(" -> Writing to PostgreSQL...")
                cur.executemany(insert_query, data_to_insert)
                conn.commit()

                print(f"  ✅ Successfully inserted {len(chunks)} rows for {source_name}.")

if __name__ == "__main__":
    
    ingest_folder("data/polity", "Polity")

    print("\n Ingestion Complete!")