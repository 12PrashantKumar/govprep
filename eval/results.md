# govprep Retrieval Evaluation Results

Corpus: full NCERT (polity + history + geography, all chapters)
Metric: Hit Rate@3 + MRR. Hit = correct keyword AND correct subject in top-3.

# Baseline vs Recursive chunking (size 500)

| Config                          | Hit Rate@3 | MRR   |
|---------------------------------|------------|-------|
| Baseline (fixed 500-char)       | 0.533      | 0.433 |
| Recursive (500/50)              | 0.600      | 0.489 |

Recursive chunking beat fixed: +0.067 hit rate, +0.056 MRR.

# Chunk-size sweep (recursive chunking)

| Size / Overlap | Hit Rate@3 | MRR   |
|----------------|------------|-------|
| 300 / 30       | 0.600      | 0.489 |
| 500 / 50       | 0.600      | 0.489 |
| 800 / 80       | 0.667      | 0.478 |
| 1000 / 100     | 0.733      | 0.656 |  <- best

Bigger chunks win on this corpus. NCERT paragraphs are long, so larger
chunks keep complete ideas together. Winner: recursive 1000/100.

# Best config so far
recursive chunking, chunk_size=1000, overlap=100
Hit Rate@3 = 0.733, MRR = 0.656 (up from 0.533 / 0.433 baseline)