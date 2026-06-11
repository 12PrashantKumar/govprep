"""
Loads ALL chapter PDFs from data/polity, data/history, data/geography,
chunks them, and stores in a NAMED ChromaDB collection.

You choose two things on the command line:
  1. chunker:         'fixed' (baseline) or 'recursive' 
  2. collection name: where to store it (so experiments don't overwrite)

Examples:
  #  baseline - full corpus, fixed-size chunking (no chunkers.py needed):
  python ingest_v2.py fixed govprep_baseline

  # - full corpus, recursive chunking (requires chunkers.py):
  python ingest_v2.py recursive govprep_recursive_500

NOTE on imports:
  recursive_chunk is imported LAZILY inside get_chunker(), NOT at the top.
  This means  ('fixed') runs even if chunkers.py doesn't exist yet.
  chunkers.py is a  file - it only needs to exist when you use 'recursive'.

Folder layout expected:
  govprep/
    data/
      polity/     ch01.pdf ch02.pdf ...
      history/    ch01.pdf ch02.pdf ...
      geography/  ch01.pdf ch02.pdf ...
    scripts/
      ingest_v2.py     <- this file
      chunkers.py      <- created day 3; only needed for 'recursive'
    db/

Run from inside scripts/:
  cd govprep/scripts
  python ingest_v2.py fixed govprep_baseline
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# ---- config ----
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = str(Path(__file__).resolve().parent.parent / "db")
SUBJECTS = ["polity", "history", "geography"]

CHUNK_SIZE = 500
OVERLAP = 50
BATCH_SIZE = 200


# ---------------- chunkers ----------------
def fixed_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """The original week-3/4 fixed-size chunker. Used for the BASELINE (day 2)."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def get_chunker(name):
    """
    Return the chunk function for the chosen method.

    The recursive import is done HERE (lazily), not at module top, so that
    running 'fixed' on day 2 does not require chunkers.py to exist yet.
    """
    if name == "fixed":
        return fixed_chunk
    elif name == "recursive":
        # Imported only when you actually choose 'recursive' 
        try:
            from chunkers import recursive_chunk
        except ImportError:
            raise ImportError(
                "Could not import recursive_chunk from chunkers.py. "
                "You only need this for the 'recursive' option (day 3+). "
                "Create scripts/chunkers.py with recursive_chunk() first, "
                "or use 'fixed' for the day-2 baseline."
            )
        return lambda t: recursive_chunk(t, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    else:
        raise ValueError(f"Unknown chunker '{name}'. Use 'fixed' or 'recursive'.")


# ---------------- loading ----------------
def load_pdf_pages(filepath):
    reader = PdfReader(str(filepath))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def gather_documents():
    """Return list of (subject, book_name, page_number, page_text)."""
    items = []
    for subject in SUBJECTS:
        folder = DATA_DIR / subject
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder} (skipping {subject})")
            continue
        pdf_files = sorted(folder.glob("*.pdf"))
        print(f"  {subject}: found {len(pdf_files)} chapter PDF(s)")
        for pdf in pdf_files:
            pages = load_pdf_pages(pdf)
            if not pages:
                print(f"    WARNING: no extractable text in {pdf.name} "
                      f"(scanned PDF?) - skipping")
                continue
            for page_num, text in pages:
                items.append((subject, pdf.stem, page_num, text))
    return items


# ---------------- ingestion ----------------
def build(chunker_name, collection_name):
    chunk_fn = get_chunker(chunker_name)

    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    if collection.count() > 0:
        print(f"Collection '{collection_name}' already has "
              f"{collection.count()} chunks. Skipping.")
        print("To rebuild this collection: delete db/ (or use a new name) and rerun.")
        return collection

    print(f"Chunker: {chunker_name}  |  Collection: {collection_name}")
    print("Gathering documents...")
    raw = gather_documents()
    print(f"Loaded {len(raw)} pages.")

    ids, docs, metas = [], [], []
    for subject, book, page_num, text in raw:
        for ci, chunk in enumerate(chunk_fn(text)):
            if not chunk.strip():
                continue
            ids.append(f"{book}_p{page_num}_c{ci}")
            docs.append(chunk)
            metas.append({"source": subject, "book": book, "page": page_num})

    total = len(docs)
    if total == 0:
        print("ERROR: 0 chunks produced. Check PDFs have selectable text "
              "and the subject folders contain PDFs.")
        return collection

    print(f"Adding {total} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            documents=docs[i:i + BATCH_SIZE],
            metadatas=metas[i:i + BATCH_SIZE],
        )
        print(f"  ...{min(i + BATCH_SIZE, total)}/{total}")

    print(f"Done. '{collection_name}' now has {collection.count()} chunks.")
    return collection


def usage():
    print("Usage: python ingest_v2.py <fixed|recursive> <collection_name>")
    print("  e.g. python ingest_v2.py fixed govprep_baseline")
    print("       python ingest_v2.py recursive govprep_recursive_500")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    chunker_name = sys.argv[1].lower()
    collection_name = sys.argv[2]
    if chunker_name not in ("fixed", "recursive"):
        usage()
        sys.exit(1)
    build(chunker_name, collection_name)