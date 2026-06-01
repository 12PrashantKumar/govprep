# govprep

An AI-powered exam preparation assistant for Indian government exam aspirants (UPSC / CDS).
Uses RAG (Retrieval-Augmented Generation) over study material to deliver grounded,
citable answers — and refuses to answer when the source material doesn't support it,
instead of hallucinating.

**Status:** v0 — working CLI. Single-document RAG over NCERT Political Science (Class 11)
content. Multi-document support and a web UI are on the roadmap.

## What it does

- Loads study material from PDF (currently NCERT Class 11 Polity)
- Chunks text into overlapping segments
- Embeds chunks using sentence-transformers (all-MiniLM-L6-v2)
- Stores + retrieves vectors with ChromaDB
- Generates answers with Gemini 2.5 Flash, grounded only in retrieved passages
- Returns "not enough information" for out-of-scope questions (no hallucinating)

## Stack

- Python 3.11+
- google-genai (Gemini 2.5 Flash)
- chromadb (local vector database)
- sentence-transformers (embeddings)
- pypdf (document loading)

## Setup

1. Clone the repo and `cd` into it
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with `GEMINI_API_KEY=your_key`
6. Place a text-layer PDF in the `data/` folder (NCERT textbooks work well)

## Run