"""
Tests several recursive-chunk configs on the FULL corpus and prints
Hit Rate@3 + MRR for each, so you can pick the best size with data.

It REUSES your existing, tested code:
  - ingest_v2.build()  for ingestion (full corpus, subfolders, correct)
  - score.evaluate()   for scoring
No duplicated PDF-loading or scoring logic = fewer bugs, one source of truth.
"""

import sys
from pathlib import Path

# make scripts/ importable
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from ingest_v2 import build          # reuse the real ingestion
from score import evaluate           # reuse the real scorer


CONFIGS = [
    {"chunk_size": 300, "overlap": 30},
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 800, "overlap": 80},
    {"chunk_size": 1000, "overlap": 100},
]


def main():
    results = []
    for cfg in CONFIGS:
        size, overlap = cfg["chunk_size"], cfg["overlap"]
        name = f"tmp_sweep_{size}_{overlap}"
        print("\n" + "#" * 60)
        print(f"# CONFIG: size={size} overlap={overlap}")
        print("#" * 60)

        # 1. ingest full corpus with this config (replace=True -> clean each time)
        build("recursive", name, chunk_size=size, overlap=overlap, replace=True)

        # 2. score it using the same scorer you use everywhere
        metrics = evaluate(name, k=3)
        if metrics:
            results.append((size, overlap, metrics["hit_rate"], metrics["mrr"]))

    # 3. summary table
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY (full corpus, recursive chunking)")
    print("=" * 60)
    print(f"{'size':>6} {'overlap':>8} {'HitRate@3':>11} {'MRR':>8}")
    for size, overlap, hr, mrr in results:
        print(f"{size:>6} {overlap:>8} {hr:>11.3f} {mrr:>8.3f}")
    if results:
        best = max(results, key=lambda r: (r[2], r[3]))
        print(f"\nBest by Hit Rate@3: size={best[0]} overlap={best[1]} "
              f"-> {best[2]:.3f} hit, {best[3]:.3f} mrr")
        


if __name__ == "__main__":
    main()