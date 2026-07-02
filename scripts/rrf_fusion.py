def reciprocal_rank_fusion(dense_results: list, bm25_results: list, k: int = 60) -> list:
    """
    Fuses two ranked lists of chunks using the Reciprocal Rank Fusion formula.
    """
    rrf_scores = {}
    chunk_map = {}                  # Used to store the actual chunk data so we can return it later

    # 1. Process the Dense Vector Results
    # enumerate() gives us the rank index (0, 1, 2...)
    for rank, chunk in enumerate(dense_results):
        text_key = chunk['text']
        # The Formula: 1 / (k + rank)
        rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + 1 / (k + rank)
        chunk_map[text_key] = chunk
        
    # 2. Process the BM25 Keyword Results
    for rank, chunk in enumerate(bm25_results):
        text_key = chunk['text']
        rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + 1 / (k + rank)
        chunk_map[text_key] = chunk
        
    # 3. Sort all chunks by their final RRF score (highest to lowest)
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 4. Rebuild the final ordered list of chunks
    fused_results = [chunk_map[key] for key in sorted_keys]
    
    return fused_results



# testing
if __name__ == "__main__":
    # Mock result from ChromaDB (Dense)
    list_a = [
        {"text": "The Right to Education is a fundamental right.", "source": "polity", "page": 46}, # Rank 0
        {"text": "Article 21 provides the right to life.", "source": "polity", "page": 45}        # Rank 1
    ]
    
    # Mock result from BM25 (Keyword)
    list_b = [
        {"text": "Article 21 provides the right to life.", "source": "polity", "page": 45},       # Rank 0
        {"text": "The Directive Principles are non-justiciable.", "source": "polity", "page": 50} # Rank 1
    ]
    
    print("🧠 Fusing Lists with RRF...")
    final_list = reciprocal_rank_fusion(list_a, list_b)
    
    for i, res in enumerate(final_list, 1):
        print(f"[{i}] {res['text']}")