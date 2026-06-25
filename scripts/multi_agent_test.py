import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# create first tool
def search_corpus(query:str)->str:
    """ Search the NCERT textbooks for relevant passages regarding Indian Polity,History,Geography and constitution
    Args:
    query: The specific topic or keyword to search for in the database
    """
    print("\n[LOCAL DATABASE SEARCH]")

    from retrieve_multi import retrieve
    try:
        chunks = retrieve(query,k=3,collection_name="govprep_v2")
        formatted_result  = "n".join(
            f"Source:{c['source']} | Page:{c['page']}] {c['text']}"  for c in chunks 
        )
        return formatted_result
    except Exception as e:
        return f"Error reading database: {str(e)}"
    
# create tool 2
def calculate(expression:str)->str:
    """Evaluate a basic mathematical expression string.
    
    Args:
        expression: A clean mathematical string to compute (e.g., '6 * 5' or '12 + 24').
    """
    print(f"   [🔧 TOOL EXECUTION] calculate called for: '{expression}'")
    try:
        return str(eval(expression))
    except Exception:
        return "Could not calculate expression."
    
# list of available tools
available_tools = [search_corpus,calculate]

# THE REASON-ACT-OBSERVE AGENT LOOP
def run_agent(question:str):
    print(f"\n🚀 Starting Agent for query: '{question}'")

    message = [
        types.Content(
            role ="user",
            parts = [types.Part.from_text(text=question)]
        )
    ]

    MAX_STEPS = 5                       #Safety filter to prevent runaway infinite loops!

    for step in range(MAX_STEPS):
        print(f"\n--- 🧠 Loop Step {step + 1}/{MAX_STEPS}: Reason Stage ---")

        # Call the model with our historical conversation trail (declare tool)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents = message,
            config = types.GenerateContentConfig(
                tools=available_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.0
            )
        )

        # decide
        if response.function_calls:
            message.append(response.candidates[0].content)

            tool_parts = []

            for tool_call in response.function_calls:
                print(f"Gemini Decided to ACT: Call '{tool_call.name}' with args {tool_call.args}")

                if tool_call.name == "search_corpus":
                    arg_query = tool_call.args.get("query")
                    result_text = search_corpus(query=arg_query)

                elif tool_call.name == "calculate":
                    arg_expr = tool_call.args.get("expression")
                    result_text = calculate(expression=arg_expr)

                else:
                    result_text = "Unknown tool requested"

                print(f"   [🔍 OBSERVE] Tool output achieved.")

                # Format the tool response back into a standard SDK Part object
                tool_parts.append(
                    types.Part.from_function_response(
                        name  = tool_call.name,
                        response = {"result" : result_text} 

                    )
                )

            # Append our execution results to the history list so the loop loops smoothly
            message.append(types.Content(role="tool", parts=tool_parts))
        else:
            print("\n✅ --- FINAL AGENT ANSWER ACHIEVED ---")
            return response.text
    
    return "Agent aborted: Hit MAX_STEPS safety limit without reaching a conclusion."

complex_prompt = "ow many Fundamental Rights are explicitly listed in the constitution, and what is that number multiplied by 10?"
final_output = run_agent(complex_prompt)
print(final_output)





