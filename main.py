from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from memory import ConversationMemory
from generate_v1 import answer


app = FastAPI(
    title="govprep API",
    description="RAG-powered UPSC/CDS study assistant",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
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

    if not req.question.strip():
        raise HTTPException(   
            status_code = 400,
            detail = "Question is Empty"
        )
    try:
        result = answer(req.question,memory)

    except Exception :
        raise HTTPException(
            status_code=503,
            detail = "The model is busy. Please try again."
        )
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

@app.get("/health")
def health():
    return {"status":"OK"}

