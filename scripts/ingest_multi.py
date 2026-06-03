import os
import sys
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

def load_pdf_with_pages(filepath):
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append((i+1,text))
    return pages

def chunk_text(text, chunk_size=500,overlap=50):
    chunks = []
    start = 0
    while start <len(text):
        chunks.append((text[start:start+chunk_size]))
        start += chunk_size - overlap
    return chunks


DOCUMENTS = {
    "polity" : DATA_DIR / "ncert_polity_ch01.pdf",
    "history" : DATA_DIR / "ncert_history_ch01.pdf",
    "geography" : DATA_DIR / "ncert_geography_ch01.pdf",
}


# build DB with metadata
if __name__ == "__main__":
    print("connection to Chroma DB ...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(name="govprep_multi", embedding_function=ef)

    if collection.count()  == 0:
        ids, docs, metas = [], [],[]
        
        # Loop 1: Go through each textbook
        for source, filepath in DOCUMENTS.items():
            if not filepath.exists():
                print(f"File {filepath} does not exist. Skipping.")
                continue

            # Loop 2: Go through each page in the textbook
            for page_num ,page_text in load_pdf_with_pages(filepath):

                # Loop 3: Go through each chunk on that page
                for ci, chunk in enumerate(chunk_text(page_text)):
                    ids.append(f"{source}_page{page_num}_chunk{ci}")
                    docs.append(chunk)
                    metas.append({"source": source, "page": page_num})
        
        if len(docs) > 0:
            print(f"Adding {len(docs)} chunks with metadata...")
            collection.add(
                ids=ids,
                documents= docs,
                metadatas = metas
            )
           
        else:
            print("No data was processed. Check if your PDFs exist in the data folder.")
    else:

        """query_result = collection.query(
                query_texts=["fundamental rights"],
                n_results = 3)
        print(query_result)"""  #just to check if the collection is working fine

        print(f"Collection 'govprep_multi' already has {collection.count()} chunks. Skipping ingestion.")





