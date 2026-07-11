import chromadb
from chromadb.utils import embedding_functions
from load_and_chunk import load_pdf, chunk_text


# ==========================================
# PART 1: INGESTION (THE WRITE PHASE)
# ==========================================
client = chromadb.PersistentClient(path="./db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="ncert_polity_ch01", 
                                             embedding_function=sentence_transformer_ef)

if collection.count()>0:
    print(f"Collection already has {collection.count()} documents. Deleting existing documents...")
else:
    print("Loading and chunking pdf...")
    text = load_pdf(r"C:\Users\prash\Desktop\govprep\data\ncert_polity_ch01.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks.")

    ids = []
    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")
    
    metadatas = []
    for i in range(len(chunks)):
        metadata = {
                "source" : "ncert_polity_ch01_",
                "chunk_index" : i

            }
        
        metadatas.append(metadata)

        print("embedding and storing chunks...")
    collection.add(
            ids=ids,
            documents = chunks,
            metadatas = metadatas
    )
    print(f"stored {collection.count()} chunks in the DB")


# ==========================================
# PART 2: RETRIEVAL (THE READ PHASE) [Query Test]
# ==========================================

queries = [
    "what is the constitution of india",
    "what are the fundamental rights in india",
    "who is the head of state in india",
]

for query in queries:
    print(f"\n\nQuery: {query}")
    results = collection.query(
        query_texts = [query],
        n_results = 3
    )

    

    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        print(f"\nResult {i+1} (distance: {distance:.4f}):")
        print(f"{doc[:200]}")