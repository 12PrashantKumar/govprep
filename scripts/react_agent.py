import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError
load_dotenv()

@tool
def search_corpus(query:str)->str:
    """Use this tool to search the NCERT textbooks for ANY factual information regarding Indian Polity, history, and the Constitution. 
    Do NOT use this tool for general knowledge, math, or outside topics."""
    print(f"   [ Action] Searching GovPrep database for: '{query}'")
    from retrieve_multi import retrieve
    try:
        chunks = retrieve(query,k=3,collection_name="govprep_v2")
        return "\n".join(f"[Source: {c['source']} | Page: {c['page']}] {c['text']}" for c in chunks)
    except Exception as e:
        return f"Tool error: {str(e)}. Try a different search query or approach."
    
@tool
def calculate(expression:str)->str:
    """Use this tool ONLY to evaluate basic mathematical expressions (like '1976 - 1949' or '32 / 2').
    Input must be a valid Python mathematical expression."""
    print(f"   [ Action] Calculating: '{expression}'")
    try:
        return str(eval(expression))
    except Exception:
        return f"Tool error: {str(e)}. Check your mathematical syntax. Ensure you are using numbers and operators, not words."
    
tool_list = [search_corpus,calculate]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

agent  =  create_agent(llm,tools = tool_list)

if __name__ == "__main__":
    print("\n STARTING prebuilt react agent.....")

    # multi-step question
    question = "Compare fundamental rights and directive principles."
    print(f"\nUser: {question}\n")
    try:
    # run agent
        final_state = agent.invoke({"messages":[("user",question)]} ,config = {"recursion_limit":3})


    # Loop through the history and print the steps cleanly to see the reasoning
        print("\n---  REASONING TRACE ---")
        for msg in final_state["messages"]:
            msg.pretty_print()
    except GraphRecursionError:
        print("\n🛑 [SYSTEM INTERVENTION] Agent exceeded max iterations (3 steps).")
        print("Safety brake applied to prevent infinite looping and API costs.")

    
