import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from lanchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title = "GovPrep AI API")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorestore = Chroma(
    persist_directory = "./govprep_lanchain_db",
    embeddings_function = embeddings
)
retriever = vectorestore.as_retriever(search_kwargs = {"k":3})

# Build the LCEL Chain

template = """ You are a tutor for competitice exams. Answer the question using ONLY this context:
{context}

Question: {question}
"""

prompt  = PromptTemplate.from_template(template)
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

rag_chain = (
    {"context" :retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    question:str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """The endpoint the frontend calls when a student asks a question."""

    answer = rag_chain.invoke(request.question)

    return {"answer":answer}
