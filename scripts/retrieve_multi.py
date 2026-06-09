"""
retrieve_multi.py  (Week 5 - parameterized version, path-robust)

Retrieval across the multi-document corpus.
Takes a collection_name so you can point it at different collections
(baseline vs recursive vs sweep experiments) without editing code.

DB path is computed from THIS FILE's location (absolute), so it works no
matter which directory you run from - matches ingest_v2.py exactly.

Used by: score.py (evaluation) and generate_v1.py (the assistant).
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Absolute path to govprep/db, computed from this file's location.
# scripts/retrieve_multi.py -> parent is scripts/ -> parent.parent is govprep/
DB_DIR = str(Path(__file__).resolve().parent.parent / "db")

DEFAULT_COLLECTION = "govprep_baseline"

_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection(collection_name=DEFAULT_COLLECTION):
    """Connect to ChromaDB (absolute db path) and return the named collection."""
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_EMBED_FN,
    )


def retrieve(query, k=3, collection_name=DEFAULT_COLLECTION, source=None):
    """
    Retrieve top-k chunks for a query from the given collection.

    query           : the question text
    k               : how many chunks to return
    collection_name : which ChromaDB collection to search
    source          : optional subject filter ('polity'/'history'/'geography')

    Returns list of dicts: {text, source, book, page, distance}
    """
    collection = get_collection(collection_name)
    where = {"source": source} if source else None

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for i in range(len(docs)):
        meta = metas[i]
        chunks.append({
            "text": docs[i],
            "source": meta.get("source", "?"),
            "book": meta.get("book", "?"),
            "page": meta.get("page", "?"),
            "distance": dists[i],
        })
    return chunks


if __name__ == "__main__":
    import sys
    coll = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COLLECTION
    print(f"DB path: {DB_DIR}")
    print(f"Testing retrieval against collection: {coll}")

    # First, show how many chunks the collection actually has - this tells us
    # immediately whether we're connected to the populated DB.
    try:
        c = get_collection(coll)
        print(f"Collection '{coll}' contains {c.count()} chunks.\n")
    except Exception as e:
        print(f"Could not open collection: {e}\n")

    for q in ["what are fundamental rights",
              "layers of the atmosphere",
              "ancient indian civilization"]:
        print(f"QUERY: {q}")
        rows = retrieve(q, k=3, collection_name=coll)
        if not rows:
            print("  (no results - collection may be empty or wrong path)")
        for r in rows:
            snip = r["text"][:80].replace("\n", " ")
            print(f"  [{r['source']:>9} {r['book']} p{r['page']} "
                  f"d={r['distance']:.3f}] {snip}...")
        print()