import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()

# the tool
@tool
def search_corpus(query:str)->str:
    """Search the NCERT textbooks for relevant passages regarding Indian Polity, History, and Constitution."""
    print(f"   [ Action] Searching GovPrep Database for: '{query}'")
    from retrieve_multi import retrieve
    try:
        chunks = retrieve(query,k=7,collection_name="govprep_v2")
        return "\n".join(f"[Source: {c['source']} | Page: {c['page']}] {c['text']}" for c in chunks)
    except Exception as e:
        return f"Error reading database: {str(e)}"
    
tool_lists = [search_corpus]

# Ai brain

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_bind_tools = llm.bind_tools(tool_lists)

def agent_node(state:MessagesState):
    response = llm_bind_tools.invoke(state["messages"])
    return {"messages":[response]}

# build graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent",agent_node)
workflow.add_node("tools",ToolNode(tool_lists))
workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent",tools_condition)
workflow.add_edge("tools","agent")

# compile
app = workflow.compile()

def agentic_answer(question:str)->dict:
    """
    Runs the LangGraph agent to answer a question.
    Returns a dictionary with the final answer and the tools it decided to use.
    """

    print(f"\n Agentic Mode Activated: '{question}'")

    # run graph
    final_state = app.invoke({"messages":[("user",question)]})
    messages = final_state["messages"]

    # extract clean text from final message
    final_message = messages[-1]
    if isinstance(final_message.content, list):
        final_answer = final_message.content[0]['text']
    else:
        final_answer = final_message.content
        
    # Trace back through the memory to see which tools it actually used
    tools_used = []
    for msg in messages:
        # Check if the message has a 'tool_calls' attribute and if it's not empty
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['name'] not in tools_used:
                    tools_used.append(tc['name'])
                    
    return {
        "answer": final_answer,
        "tools_used": tools_used
    }


#  (Only runs if you execute this file directly)

if __name__ == "__main__":
    # Test 1: In-Scope (Should use search_corpus and cite sources)
    polity_question = "Which Article number guarantees the Right to Constitutional Remedies?"
    result_1 = agentic_answer(polity_question)
    print("\n--- TEST 1 RESULT ---")
    print(f"Tools Used: {result_1['tools_used']}")
    print(f"Answer: {result_1['answer']}\n")
    
    print("-" * 50)
    
    # Test 2: Out-of-Scope (Should NOT use tools, answers directly)
    random_question = "What is the capital of France?"
    result_2 = agentic_answer(random_question)
    print("\n--- TEST 2 RESULT ---")
    print(f"Tools Used: {result_2['tools_used']}")
    print(f"Answer: {result_2['answer']}\n")
          
