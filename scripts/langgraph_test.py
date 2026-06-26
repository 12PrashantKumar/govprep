import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class State(TypedDict):
    messages : Annotated[list,add_messages]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",temperature=0.0)

def chatbot_node(state:State):
    print("   [NODE] Chatbot is thinking...")
    response = llm.invoke(state["messages"])
    return {"messages" : [response]}

graph = StateGraph(State)               #build graph

graph.add_node("chatbot",chatbot_node)

graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

app = graph.compile()           # Compile it into a runnable application

# generating image to look at graph
print(" Generating graph diagram...")
try:
    # Get the raw PNG bytes of the graph drawing
    graph_image_bytes = app.get_graph().draw_mermaid_png()
    
    # Save the bytes to a file in your current folder
    with open("graph_diagram.png", "wb") as f:
        f.write(graph_image_bytes)
        
    print(" Success! Open 'graph_diagram.png' in VS Code folder to see it.")
except Exception as e:
    print(f" Could not generate image. Error: {e}")



#  Run it!

print(" Starting LangGraph...")
user_input = "Hello! What is your name?"
print(f"User: {user_input}")

# Stream the output so we can see the graph working
for event in app.stream({"messages":[["user",user_input]]}):
    for value in event.values():
        print(f"Agent: {value['messages'][-1].content}")





