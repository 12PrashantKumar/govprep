import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()



def rewrite_query(user_input:str)->str:
    """
    Takes a messy user query and expands it for optimal database retrieval.
    """
    print(f" Original Query :'{user_input}'")

    llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

    Prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""You are an AI assistant specialized in Indian government, civil service, and defense exams. 
        Your job is to rewrite the user's query to make it highly optimized for a vector database and exact-keyword search.
        
        Rules:
        1. Fix any spelling or grammar mistakes.
        2. Expand acronyms (e.g., 'UPSC', 'NDA', 'FRs').
        3. Clarify the intent (e.g., if they say 'article 21', expand it to 'Article 21 of the Indian Constitution').
        4. Return ONLY the rewritten query. Do not add conversational filler.
        
        User Query: {query}
        Rewritten Query:"""

    )
    chain = Prompt_template | llm

    response = chain.invoke({"query":user_input})
    rewritten_string = response.content.strip()

    print(f" Rewritten Query: '{rewritten_string}'")
    return rewritten_string

# --- Quick Test ---
if __name__ == "__main__":
    # A classic, messy student query
    test_query = "what is art 21 and how it protect me?"
    
    optimized_query = rewrite_query(test_query)
