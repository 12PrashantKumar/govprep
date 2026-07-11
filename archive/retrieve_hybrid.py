from retrieve_multi import retrieve as dense_retrieve 
# Import the new tools just built
from hybrid_search import BM25Retriever
from rrf_fusion import reciprocal_rank_fusion


# 1. INITIALIZE BM25 GLOBALLY
# We load the entire corpus into BM25 once when the server starts
print("⚙️ Booting up Hybrid Search Engine...")
try:
    from retrieve_multi import retrieve as dense_retrieve, get_all_chunks
    
    # We load the entire corpus into BM25 once when the server starts
    all_database_chunks = get_all_chunks(collection_name="govprep_v2")
    keyword_engine = BM25Retriever(chunks=all_database_chunks)
except ImportError as e:
    print(f"⚠️ Warning: Could not auto-load chunks: {e}")
    keyword_engine = None


# 2. THE HYBRID SEARCH FUNCTION
def hybrid_search(query: str, k: int = 5) -> list:
    """
    Executes Dense Vector Search and BM25 Keyword Search in parallel, 
    then fuses the results using RRF.
    """
    print(f"\n🔍 [Hybrid Search Initiated] Query: '{query}'")
    
    # Step A: Get results from ChromaDB (Semantics)
    print("   [↳] Running Dense Vector Search...")
    dense_results = dense_retrieve(query, k=k, collection_name="govprep_v2")
    
    # Step B: Get results from BM25 (Keywords)
    print("   [↳] Running BM25 Keyword Search...")
    if keyword_engine:
        keyword_results = keyword_engine.search(query, k=k)
    else:
        keyword_results = []
        
    # Step C: Fuse with RRF
    print("   [↳] Fusing lists with Reciprocal Rank Fusion...")
    final_fused_list = reciprocal_rank_fusion(dense_results, keyword_results)
    
    # Return the top K results from the newly fused Master List
    return final_fused_list[:k]


# TESTING
if __name__ == "__main__":
    # Test a query that requires exact matching
    test_query = "Article 21A"
    results = hybrid_search(test_query, k=3)
    
    print("\n✅ FINAL HYBRID RESULTS:")
    for i, res in enumerate(results, 1):
        print(f"[{i}] Source: {res.get('source', 'N/A')} | Page: {res.get('page', 'N/A')}")
        print(f"    Text: {res.get('text', '')[:100]}...\n")