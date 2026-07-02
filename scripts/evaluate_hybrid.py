import json
from pathlib import Path
from retrieve_hybrid import hybrid_search
from retrieve_multi import retrieve as dense_retrieve # Bring back the old engine

# EVALUATION METRICS CALCULATOR (DYNAMIC)
def calculate_metrics(search_function, gold_set: list, k: int = 3):
    """
    Runs each query in the gold set through the PROVIDED search engine.
    """
    hits = 0
    reciprocal_ranks = []
    total_queries = len(gold_set)

    for item in gold_set:
        query = item["question"]
        required_keyword = item.get("required_keyword", "").lower()
        expected_source = item.get("expected_source", "").lower()

        # Run the specific engine we passed in
        if search_function.__name__ == 'retrieve': # Dense search signature
            retrieved_chunks = search_function(query, k=k, collection_name="govprep_v2")
        else: # Hybrid search signature
            retrieved_chunks = search_function(query, k=k)
        
        found_hit = False
        for rank_idx, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get("text", "").lower()
            chunk_source = chunk.get("source", "").lower()

            if required_keyword in chunk_text and expected_source in chunk_source:
                hits += 1
                reciprocal_ranks.append(1.0 / (rank_idx + 1))
                found_hit = True
                break
        
        if not found_hit:
            reciprocal_ranks.append(0.0)

    hit_rate = hits / total_queries if total_queries > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total_queries if total_queries > 0 else 0.0

    return hit_rate, mrr

# EXECUTION & COMPARISON
if __name__ == "__main__":
    gold_set_path = Path(__file__).resolve().parent.parent / "eval" / "gold_set.json"
    
    if not gold_set_path.exists():
        print("❌ gold_set.json not found. Ensure path is correct.")
        exit(1)

    with open(gold_set_path, "r", encoding="utf-8") as f:
        gold_set_data = json.load(f)

    print(f"📊 Running evaluation across {len(gold_set_data)} questions...")

    # 1. LIVE TEST: Vector Baseline
    print("   [↳] Testing Vector-Only Baseline...")
    vector_hit_rate, vector_mrr = calculate_metrics(dense_retrieve, gold_set_data, k=3)

    # 2. LIVE TEST: Hybrid Search
    print("   [↳] Testing Hybrid Engine...")
    hybrid_hit_rate, hybrid_mrr = calculate_metrics(hybrid_search, gold_set_data, k=3)

    # Calculate True Improvements
    hit_diff = 0.0
    if vector_hit_rate > 0:
        hit_diff = ((hybrid_hit_rate - vector_hit_rate) / vector_hit_rate) * 100
        
    mrr_diff = 0.0
    if vector_mrr > 0:
        mrr_diff = ((hybrid_mrr - vector_mrr) / vector_mrr) * 100

    print("\n" + "="*50)
    print("📈 TRUE RETRIEVAL BENCHMARK REPORT")
    print("="*50)
    print(f"{'Metric':<15} | {'Vector Baseline':<16} | {'Hybrid (RRF)':<14} | {'Delta':<10}")
    print("-" * 60)
    print(f"{'Hit Rate@3':<15} | {vector_hit_rate:<16.3f} | {hybrid_hit_rate:<14.3f} | {hit_diff:>+6.1f}%")
    print(f"{'MRR':<15} | {vector_mrr:<16.3f} | {hybrid_mrr:<14.3f} | {mrr_diff:>+6.1f}%")
    print("="*50)