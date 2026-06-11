"""
Scores retrieval quality against your gold set using Hit Rate@k and MRR.
Takes the collection name + k on the command line, so you can score any
collection (baseline, recursive, sweep experiments) without editing code.

Usage:
  cd govprep/eval
  python score.py <collection_name> [k]

Examples:
  python score.py govprep_baseline          # baseline, k=3 (default)
  python score.py govprep_recursive_500     # recursive chunking, k=3
  python score.py govprep_baseline 5        # baseline, k=5

Reads gold_set.json from the same eval/ folder.
"""

import json
import sys
from pathlib import Path

# Make scripts/ importable so we can use retrieve_multi.retrieve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from retrieve_multi import retrieve


def load_gold():
    path = Path(__file__).resolve().parent / "gold_set.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hit_and_rank(gold_item, k, collection_name):
    """Return (hit:bool, rank:int or None) for one question."""
    chunks = retrieve(
        gold_item["question"],
        k=k,
        collection_name=collection_name,
    )
    keyword = gold_item["required_keyword"].lower()
    target_source = gold_item["expected_source"].lower()

    for rank, c  in enumerate(chunks, start=1):
        chunk_text = c["text"].lower()
        chunk_source = c["source"].lower()

        # it is only hit if it is right book and has  keyword
        if keyword in chunk_text and chunk_source == target_source:
            return True, rank
        
    return False, None


def evaluate(collection_name, k=3):
    gold = load_gold()
    hits, reciprocal_ranks = 0, []

    print(f"\nScoring collection '{collection_name}' at k={k}")
    print("=" * 60)
    for item in gold:
        hit, rank = hit_and_rank(item, k, collection_name)
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            status = f"hit@{rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"
        print(f"[{status:>6}] {item['question'][:55]}")

    n = len(gold)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    print("=" * 60)
    print(f"Collection : {collection_name}")
    print(f"Hit Rate @{k}: {hit_rate:.3f}  ({hits}/{n})")
    print(f"MRR        : {mrr:.3f}")
    return {"collection": collection_name, "k": k,
            "hit_rate": hit_rate, "mrr": mrr}


def usage():
    print("Usage: python score.py <collection_name> [k]")
    print("  e.g. python score.py govprep_baseline")
    print("       python score.py govprep_recursive_500 3")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    collection = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    evaluate(collection, k=k)