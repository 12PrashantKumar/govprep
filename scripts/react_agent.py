import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def search_corpus(query:str)->str:
    """Search the NCERT textbooks for relevant passages regarding Indian Polity and Constitution."""
    print(f"   [🔧 Action] Searching GovPrep database for: '{query}'")
    from retrieve_multi import retrieve
    try:
        chunks = retrieve(query,k=3,collection_name="govprep_v2")
        return "\n".join(f"[Source: {c['source']} | Page: {c['page']}] {c['text']}" for c in chunks)
    except Exception as e:
        return f"Error reading database: {str(e)}"
    
@tool
def calculate(expression:str)->str:
    """Evaluate a basic  mathematical expression string"""
    print(f"   [🔧 Action] Calculating: '{expression}'")
    try:
        return str(eval(expression))
    except Exception:
        return "Could not calculate expression."
    
tool_list = [search_corpus,calculate]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

agent  =  create_agent(llm,tools = tool_list)

if __name__ == "__main__":
    print("\n STARTING prebuilt react agent.....")

    # multi-step question
    question = "Compare fundamental rights and directive principles."
    print(f"\nUser: {question}\n")

    # run agent
    final_state = agent.invoke({"messages":[("user",question)]})


    # Loop through the history and print the steps cleanly to see the reasoning
    print("\n--- 🧠 REASONING TRACE ---")
    for msg in final_state["messages"]:
        msg.pretty_print()


    
