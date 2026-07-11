import json
from pg_hybrid_retriever import hybrid_retriever

with open("eval/gold_set.json", encoding="utf-8") as f:
    gold = json.load(f)

hits, rr_total = 0, 0.0
for item in gold:
    q = item["question"]
    keyword = item["required_keyword"].lower()
    docs = hybrid_retriever(q, k=3)

    rank = 0
    for i, d in enumerate(docs, start=1):
        if keyword in d.page_content.lower():
            rank = i
            break

    if rank > 0:
        hits += 1
        rr_total += 1.0 / rank
    print(f"{'✅' if rank else '❌'} rank={rank or '-'}  {q[:55]}")

n = len(gold)
print(f"\nHit Rate@3: {hits/n:.3f}   MRR: {rr_total/n:.3f}   (n={n})")