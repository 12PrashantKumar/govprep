import os
import re
import json
import glob
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Load the embedding model ONCE at module level (not per folder).
# all-mpnet-base-v2 -> 768 dims, matches the VECTOR(768) column.
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


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
    print(f"\nStarting ingestion for {subject} from '{folder_path}'....")

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Find all PDFs in the target folder
    pdf_files = sorted(glob.glob(f"{folder_path}/*.pdf"))
    if not pdf_files:
        print(f" No PDFs found in {folder_path}")
        return

    ingested, skipped = 0, 0

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            for pdf_file in pdf_files:
                source_name = os.path.basename(pdf_file)

                # --- IDEMPOTENCY GUARD ---
                # Runs BEFORE loading/embedding so skipped files cost nothing.
                cur.execute(
                    "SELECT 1 FROM chunks WHERE source = %s LIMIT 1",
                    (source_name,)
                )
                if cur.fetchone():
                    print(f"  -> {source_name} already ingested, skipping.")
                    skipped += 1
                    continue

                print(f"\nProcessing: {pdf_file}")

                # Step A: Load PDF
                loader = PyPDFLoader(pdf_file)
                raw_docs = loader.load()

                # Apply scrubber
                for doc in raw_docs:
                    doc.page_content = scrub_ncert_text(doc.page_content)

                # Step B: Chunk text
                chunks = text_splitter.split_documents(raw_docs)
                if not chunks:
                    print(f"  !! No chunks produced for {source_name} - skipping.")
                    continue
                print(f" -> Created {len(chunks)} clean chunks.")

                # Step C: Get embeddings (local model, no rate limits)
                text_strings = [chunk.page_content for chunk in chunks]
                print(f"    -> Embedding {len(text_strings)} chunks locally...")
                vectors = EMBEDDINGS_MODEL.embed_documents(text_strings)

                # Step D: Prepare for database insert
                # NOTE: search_vector is a GENERATED column - Postgres fills it
                # automatically from content. Never insert into it.
                insert_query = """
                INSERT INTO chunks (content, subject, source, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """

                data_to_insert = []
                for i, chunk in enumerate(chunks):
                    data_to_insert.append((
                        chunk.page_content,
                        subject,
                        source_name,
                        json.dumps(chunk.metadata),  # LangChain dict -> JSON for Postgres
                        vectors[i],
                    ))

                # Step E: Batch execute and commit
                print(" -> Writing to PostgreSQL...")
                cur.executemany(insert_query, data_to_insert)
                conn.commit()
                ingested += 1

                print(f"  OK Inserted {len(chunks)} rows for {source_name}.")

    print(f"\n[{subject}] done - {ingested} file(s) ingested, {skipped} skipped.")


if __name__ == "__main__":
    ingest_folder("data/polity", "Polity")
    ingest_folder("data/history", "History")
    ingest_folder("data/geography", "Geography")

    print("\nIngestion Complete!")