# Roadmap
- Retrieval tuning (chunking strategy, embedding model)
- Semantic caching (Redis / Postgres)
- Explicit agentic router node
- CI/CD pipeline (GitHub Actions)
- Re-run RAGAS eval with OpenAI judge (Groq incompatible with RAGAS n>1 sampling)
- Re-wire centralized llm.py with 429 backoff
- Replace eval() in calculator tool with a safe math parser