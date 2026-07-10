from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from scripts.pg_hybrid_retriever import hybrid_retriever
from scripts.pg_rewriter import rewrite_query
from scripts.pg_guardrails import check_input_safety, mask_pii

load_dotenv()

app = FastAPI(title="GovPrep AI Backend", version="1.1.0")
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant", 
    temperature=0.2
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def home():
    return {
        "status" :"running",
        "service" :"govprep api "
    }


@app.get("/health")
def health_check():
    return {"status": "GovPrep API is running perfectly"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        print(f"📥 Raw User Input: '{request.question}'")

        # --- 2A. EXECUTE GUARDRAIL 1: Prompt Injection Check ---
        is_safe = check_input_safety(request.question)
        
        # If the injection checker returns False (or however you designed it to flag a threat)
        if not is_safe:
             print("🚨 SECURITY ALERT: Malicious prompt detected. Blocking request.")
             return ChatResponse(
                 answer="I cannot answer this question as it violates safety guidelines.",
                 sources=[]
             )

        # --- 2B. EXECUTE GUARDRAIL 2: PII Scrubber ---
        # If we made it here, the prompt is safe from injection. Now we clean the data.
        clean_query = mask_pii(request.question)
        print(f"🧼 Scrubbed Query (PII removed): '{clean_query}'")

        # --- 3. EXECUTE THE REWRITER ---
        # Pass the CLEAN query into the rewriter, not the raw request!
        optimized_query = rewrite_query(clean_query)
        print(f"🔄 Rewritten Optimized Query: '{optimized_query}'")
        
        # --- 4. RETRIEVE USING OPTIMIZED TERMS ---
        print(f"🔍 Searching database with expanded terms...")
        retrieved_docs = hybrid_retriever(optimized_query, k=3)
        
        context_list = []
        sources_list = []
        for doc in retrieved_docs:
            context_list.append(doc.page_content)
            source_info = doc.metadata.get("page", "Unknown Page")
            sources_list.append(f"Page {source_info}")
            
        context_string = "\n\n".join(context_list)
        
        # --- 5. GENERATE THE ANSWER ---
        prompt = f"""You are an expert tutor for Indian government exams.
        Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I don't have that information in my current NCERT database."
        
        Context:
        {context_string}
        
        Question: {clean_query}
        Answer:"""
        
        response = llm.invoke(prompt)
        
        return ChatResponse(
            answer=response.content.strip(),
            sources=list(set(sources_list))
        )

    except Exception as e:
        print(f"❌ Error in pipeline execution: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")