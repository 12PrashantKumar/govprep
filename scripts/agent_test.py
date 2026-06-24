import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client  = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

# make tool
def search_corpus(query:str)->str:
    """ Search the NCERT textbooks for relevant passages regarding Indian Polity,History,Geography and constitution
    Args:
    query: The specific topic or keyword to search for in the database
    """
    print(f"\n[LOCAL DATABASE SEARCH]")

    from retrieve_multi import retrieve
    
    chunks = retrieve(query, k=3, collection_name="govprep_v2")

    formatted_result= "\n".join(
        f"[Source:{c['source']} | Page:{c['page']} {c['text']}" for c in chunks 
    )

    return formatted_result

# dynamic cycle
# Test with an IN-SCOPE question
user_question = "What are Fundamental Rights?"
print(f"Sending question to Gemini: '{user_question}'...")

# declare tool
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = user_question,
    config = types.GenerateContentConfig(
        tools = [search_corpus],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0
    )
)

# decide
if response.function_calls:
    for tool_call in response.function_calls:
        print(f"Function called: {tool_call.name}")
        print(f"function argument: {tool_call.args}")

        # execute
        if tool_call.name == "search_corpus":
            search_query = tool_call.args.get("query")
            # Execute your real ChromaDB search local code
            db_result = search_corpus(query=search_query)

            # continue and also mainting conversation history
            final_result = client.models.generate_content(
                model="gemini-2.5-flash",
                contents= [
                    user_question,
                    response.candidates[0].content,
                    types.Content(
                        role = "tool",
                        parts = [types.Part.from_function_response(
                            name="search_corpus",
                            response = {
                                "result" : db_result
                            }
                            )
                        ]
                    )

                ]

            )
            print("\n✅ --- FINAL GROUNDED ANSWER ---")
            print(final_result.text)
else:
    print("\nGemini answered directly without using the database:")
    print(response.text)







