import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()

# the tool(the action)
@tool
def search_corpus(query:str)->str:
    """Search the NCERT textbooks for relevant passages regarding Indian Polity and Constitution.
    """
    print(f" [ACTION] Searching Database for :{query}")
    from retrieve_multi import retrieve
    try:
        chunks = retrieve(query,k=5,collection_name="govprep_v2")
        return "\n".join(f"[Source: {c['source']} | Page: {c['page']}] {c['text']}" for c in chunks)
    except Exception as e:
        return f"Error reading database: {str(e)}"
    
@tool
def calculate(expression:str)->str:
    """Evaluate a basic mathematical expression string."""
    print(f"   [ACTION] Calculating:{expression}")
    try:
        return str(eval(expression))
    except Exception:
        return "Could not calculate expression."
    
tools_list = [search_corpus,calculate]

# AI BRAIN(calling/connecting to gemini model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
llm_with_tools =  llm.bind_tools(tools_list)

# agent node
def agent_node(state:MessagesState):
    print("     [Thinking] Agent is deciding the next steps...")
    # Pass the entire conversation history to the model
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

# build the graph or workflow
workflow = StateGraph(MessagesState)

# add node
workflow.add_node("agent",agent_node)
workflow.add_node("tools",ToolNode(tools_list))

# make edges
workflow.add_edge(START,"agent")

# [agent node] --(wants tool?)--> [tool node] OR END
# tools_condition is LangGraph's prebuilt edge that checks if the LLM asked for a tool.
workflow.add_conditional_edges("agent",tools_condition)
workflow.add_edge("tools","agent")

# compile the graph
app =workflow.compile()

complex_prompt = "How many Fundamental Rights are explicitly listed in the constitution, and what is that number multiplied by 10?"

print(f"\nUser asks: {complex_prompt}\n")

# Run the entire loop silently and just get the final state
final_state = app.invoke({"messages": [("user", complex_prompt)]})
final_message = final_state["messages"][-1]
print("\n FINAL ANSWER:")
# If Google returned a list of data blocks, grab the text from the first block
if isinstance(final_message.content, list):
    print(final_message.content[0]['text'])
# Otherwise, just print the string normally
else:
    print(final_message.content)


graph_image_bytes = app.get_graph().draw_mermaid_png()
    # Save the bytes to a file in your current folder
with open("graph_diagram.png", "wb") as f:
        f.write(graph_image_bytes)
