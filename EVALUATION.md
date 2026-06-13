# govprep — Retrieval Evaluation

Measuring and improving retrieval quality on govprep, with metrics instead of guesswork.

# Method
- Corpus: full NCERT textbooks — Polity, History, Geography (all chapters)
- Gold set: 15 hand-written questions, each tagged with a required keyword and expected subject
- Metrics: Hit Rate@k (fraction of questions where the correct chunk appeared in top-k) and MRR (rewards the correct chunk being ranked higher)
- Hit rule: counted as a hit only when BOTH the required keyword AND the correct subject matched — stricter than keyword-only, to avoid false positives

# Results

| Config                        | Hit Rate@3 | MRR   |
|-------------------------------|------------|-------|
| Baseline (fixed 500-char)     | 0.533      | 0.433 |
| Recursive (500/50)            | 0.600      | 0.489 |
| Recursive (1000/100)          | 0.733      | 0.656 |

Chunk-size sweep showed bigger chunks won; top-k sweep showed performance plateaus at k=3.
Final config: recursive chunking, 1000/100, k=3 — a +37% hit rate and +52% MRR improvement over baseline.