import json , sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

# import retreival function and python gold set
from retrieve_multi import retrieve

def load_gold():
    gold_path =  Path(__file__).resolve().parent / "gold_set.json"
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def hit_and_rank(gold_item, k=3):
    """Return (hit:bool, rank:int or None)for one question"""
    chunks = retrieve(gold_item["question"], k=k)

    
    keyword = gold_item.get("required_keyword", "").lower()
    target_source = gold_item["expected_source"].lower()

    for rank, c  in enumerate(chunks, start=1):
        chunk_text = c["text"].lower()
        chunk_source = c["source"].lower()

        # it is only hit if it is right book and has  keyword
        if keyword in chunk_text and chunk_source == target_source:
            return True, rank
        
    return False, None

def evaluate(k=3):
    gold = load_gold()
    hits = 0
    reciprocal_ranks = []

    print(f"Starting evaluation across {len(gold)} queries (k={k})....\n")

    for item in gold:
        hit, rank = hit_and_rank(item,k=k)

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0/ rank)
            status = f"hit@{rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"

        print(f"[{status:>6}] {item['question'][:55]}...")


    n = len(gold)
    hit_rate = hits/n
    mrr = sum(reciprocal_ranks)/n

    print(f"\n{'='*50}")
    print(f"Hit Rate @{k}: {hit_rate:.3f}  ({hits}/{n})")
    print(f"MRR @{k}:      {mrr:.3f}")
    print(f"{'='*50}\n")

    return {"hit_rate": hit_rate, "mrr": mrr, "k": k}

if __name__ == "__main__":
    evaluate(k=3)