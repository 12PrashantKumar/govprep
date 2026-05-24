# govprep

An AI-powered exam preparation assistant for UPSC CDS aspirants.
Uses RAG (Retrieval-Augmented Generation) over previous year question papers
to deliver grounded, citable answers.

**Status:** v0 in development (week 1 of build).

## Stack

- Python 3.11+
- google-genai (Gemini 2.5 Flash for generation)
- chromadb (local vector database)
- sentence-transformers (all-MiniLM-L6-v2 for embeddings)
- pypdf (document loading)

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with `GEMINI_API_KEY=your_key`
6. Download UPSC CDS papers (PDF) from upsc.gov.in into `data/` folder.

## Run