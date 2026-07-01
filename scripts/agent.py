import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

load_dotenv()


# 1. DEFINE  TOOLS 
@tool
def search_corpus(query: str) -> str:
    """Use this tool FIRST to search the NCERT textbooks for factual information regarding Indian Polity, history, and the Constitution."""
    print(f"   [🔧 Action] Searching GovPrep database for: '{query}'")
    try:
        from retrieve_multi import retrieve
        chunks = retrieve(query, k=5, collection_name="govprep_v2")
        return "\n".join(f"[Source: {c['source']} | Page: {c['page']}] {c['text']}" for c in chunks)
    except Exception as e:
        return f"Tool error: {str(e)}. Try a different search query or approach."

@tool
def calculate(expression: str) -> str:
    """Use this tool ONLY to evaluate basic mathematical expressions (like '1976 - 1949' or '32 / 2').
    Input must be a valid Python mathematical expression."""
    print(f"   [🔧 Action] Calculating: '{expression}'")
    try:
        return str(eval(expression)) 
    except Exception as e:
        return f"Tool error: {str(e)}. Check your mathematical syntax. Ensure you are using numbers and operators."

@tool
def web_search(query: str) -> str:
    """Use this tool ONLY for current affairs, recent news, or general knowledge NOT found in the NCERT corpus. 
    Do not use this for historical constitutional facts."""
    print(f"   [🔧 Action] Searching Wikipedia for: '{query}'")
    try:
        import wikipedia
        wikipedia.set_lang("en")
        
        # Step 1: Search for the best matching page title first
        search_results = wikipedia.search(query)
        
        if not search_results:
            return "Tool error: No Wikipedia page found. Try using shorter keywords instead of a full sentence."
            
        top_page = search_results[0]
        print(f"   [↳ Detail] Fetching summary for page: '{top_page}'")
        
        # Step 2: Fetch the summary of that specific page
        result = wikipedia.summary(top_page, sentences=4)
        return result
        
    except Exception as e:
        return f"Tool error: Wikipedia search failed - {str(e)}. Try using a very short, specific keyword (e.g., 'Chief of Defence Staff')."
    
#  the tools list
tools_list = [search_corpus, calculate, web_search]


#CREATE THE MODEL & AGENT (Global)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent = create_agent(llm, tools=tools_list)

# ==========================================
# THE EXPORTABLE FUNCTION
# ==========================================
def answer_agentic(question: str) -> dict:
    """
    Runs the ReAct agent on a user's question.
    Returns a dictionary with the final answer text.
    """
    print(f"\n🤖 [AGENT INITIATED] Question: '{question}'")
    
    try:
        final_state = agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": 5} 
        )
        
        # Extract the final answer text cleanly
        final_message = final_state["messages"][-1]
        if isinstance(final_message.content, list):
            answer_text = final_message.content[0]['text']
        else:
            answer_text = final_message.content
            
        return {"answer": answer_text}
        
    except GraphRecursionError:
        print("🛑 [SYSTEM INTERVENTION] Max Iteration Limit Reached.")
        return {"answer": "I'm sorry, I couldn't resolve that question within my allowed reasoning steps. Please try rephrasing."}
    except Exception as e:
        print(f"🛑 [CRITICAL ERROR] {str(e)}")
        return {"answer": "I encountered an unexpected system error while trying to think."}


# 4. QUICK TEST BLOCK

if __name__ == "__main__":
    # Test 1: Should use search_corpus
    # test_result = answer_agentic("Compare fundamental rights and directive principles.")
    
    # Test 2: Should use web_search
    test_result = answer_agentic("What is the capital of Australia?")
    
    print("\n✅ FINAL ANSWER:")
    print(test_result["answer"])