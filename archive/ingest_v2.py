"""
Loads ALL chapter PDFs from data/polity, data/history, data/geography,
chunks them, and stores in a NAMED ChromaDB collection.

Command line:
  python ingest_v2.py <fixed|recursive> <collection_name> [chunk_size] [overlap]

Examples:
  python ingest_v2.py fixed govprep_baseline
  python ingest_v2.py recursive govprep_recursive_500
  python ingest_v2.py recursive tmp_sweep_300_30 300 30 

If chunk_size/overlap are omitted, defaults (500/50) are used.
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = str(Path(__file__).resolve().parent.parent / "db")
SUBJECTS = ["polity", "history", "geography"]

#these were hardcoded constants CHUNK_SIZE/OVERLAP used directly in the chunkers. Now they are only DEFAULTS - the real values can be passed in per call, which is what lets the sweep test multiple sizes.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
BATCH_SIZE = 200


# fixed_chunk  RECEIVES chunk_size/overlap as parameters
#     instead of reading global constants. using the module constants.)
def fixed_chunk(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# get_chunker now takes chunk_size + overlap and passes them into whichever chunker it returns. 
#     The lazy 'from chunkers import recursive_chunk' inside the recursive
#     branch is still here so 'fixed' runs without chunkers.py.
def get_chunker(name, chunk_size, overlap):
    if name == "fixed":
        return lambda t: fixed_chunk(t, chunk_size, overlap)
    elif name == "recursive":
        try:
            from chunkers import recursive_chunk
        except ImportError:
            raise ImportError(
                "Could not import recursive_chunk from chunkers.py. "
                "Needed only for 'recursive'. Create scripts/chunkers.py first, "
                "or use 'fixed'."
            )
        return lambda t: recursive_chunk(t, chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunker '{name}'. Use 'fixed' or 'recursive'.")


#load_pdf_pages + gather_documents  ----
def load_pdf_pages(filepath):
    reader = PdfReader(str(filepath))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def gather_documents():
    """Return list of (subject, book_name, page_number, page_text) for ALL chapters."""
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
                print(f"    WARNING: no text in {pdf.name} (scanned?) - skipping")
                continue
            for page_num, text in pages:
                items.append((subject, pdf.stem, page_num, text))
    return items


#build() signature gained chunk_size, overlap, and replace.
#     - chunk_size/overlap: passed through to get_chunker (default 500/50)
#     - replace: NEW. When True, deletes the collection first so each sweep
#       config starts clean. When False, keeps old behaviour (skip if filled).
def build(chunker_name, collection_name,
          chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP,
          replace=False):
    chunk_fn = get_chunker(chunker_name, chunk_size, overlap)

    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # optional clean-slate delete, used by the sweep
    if replace:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=ef
    )

    # < "and not replace" so a sweep rebuild isn't skipped
    if collection.count() > 0 and not replace:
        print(f"Collection '{collection_name}' already has {collection.count()} "
              f"chunks. Skipping. (delete db/ or use a new name to rebuild)")
        return collection

    #  print line now shows size/overlap so each run is labelled
    print(f"Chunker: {chunker_name} | size={chunk_size} overlap={overlap} "
          f"| Collection: {collection_name}")
    raw = gather_documents()
    print(f"Loaded {len(raw)} pages.")

    # --chunk building + batched add ----
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
        print("ERROR: 0 chunks produced. Check PDFs/folders.")
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
    print("Usage: python ingest_v2.py <fixed|recursive> <collection_name> "
          "[chunk_size] [overlap]")


#  __main__ now reads optional 3rd/4th args (chunk_size, overlap).
#     If you don't pass them, it behaves exactly like before (500/50).
if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)
    chunker_name = sys.argv[1].lower()
    collection_name = sys.argv[2]
    cs = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHUNK_SIZE
    ov = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_OVERLAP
    if chunker_name not in ("fixed", "recursive"):
        usage()
        sys.exit(1)
    build(chunker_name, collection_name, chunk_size=cs, overlap=ov)