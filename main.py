

from fastapi import FastAPI, Request
import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from memory import ConversationMemory
from generate_v1 import answer

# Create FastAPI application
app = FastAPI(
    title="govprep API",
    description="RAG-powered UPSC/CDS study assistant",
    version="1.0"
)

# Shared memory (simple version for now)
memory = ConversationMemory()


@app.get("/")
def home():
    """
    Health check endpoint.
    """
    return {
        "status": "running",
        "service": "govprep API"
    }


@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint.

    Expected JSON:
    {
        "question": "What are Fundamental Rights?"
    }
    """

    data = await request.json()

    question = data["question"]

    result = answer(question, memory)

    return {
        "answer": result["answer"],
        "rewritten": result["rewritten"],
        "sources": [
            {
                "source": c["source"],
                "page": c["page"]
            }
            for c in result["chunks"]
        ]
    }


@app.get("/history")
def history():
    """
    View current conversation history.
    """

    return {
        "history": memory.as_text()
    }


@app.post("/reset")
def reset():
    """
    Clear conversation memory.
    """

    global memory
    memory = ConversationMemory()

    return {
        "status": "success",
        "message": "Conversation memory cleared."
    }