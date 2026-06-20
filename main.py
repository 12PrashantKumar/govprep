from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from memory import ConversationMemory
from generate_v1 import answer


app = FastAPI(
    title="govprep API",
    description="RAG-powered UPSC/CDS study assistant",
    version="1.0"
)

memory = ConversationMemory()

class ChatRequest(BaseModel):
    question : str

class Source(BaseModel):
    source : str
    page : int

class ChatResponse(BaseModel):
    answer : str
    rewritten : str
    sources : List[Source]




@app.get("/")
def home():
    return {
        "status" :"running",
        "service" :"govprep api "
    }




@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer(req.question,memory)
    return ChatResponse (
        answer= result["answer"],
        rewritten=result["rewritten"],
        sources = [
            Source(
                source = c["source"],
                page=c["page"]
                )
            for c in result["chunks"]
        ]
    )
