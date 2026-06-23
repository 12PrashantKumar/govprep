import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# the tool
def get_weather(city:str)-> str:
    """Get the current weather of given city.
    Args:
        city:the name of the city to get weather for
    """
    print(f"\n[LOCAL EXECUTION]Runnnig  Python function for:{city}")
    return f"The waether in{city} is 28\N{DEGREE SIGN}C and Sunny."

# now the cycle
print("Semding question to Gemini: 'What's the weather like in Delhi?'...")

# step 1 : Declare tool
response =client.models.generate_content(
    model='gemini-2.5-flash',
    contents = "What's the weather like in Delhi?",
    config = types.GenerateContentConfig(
        tools= [get_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0
    )
)
# Step 2: The Model Decides
if response.function_calls:                     #Returns the list of function calls in the response.
    for tool_call in response.function_calls:
        print("\n GEMINI PAUSED TEXT GENERATION! ")
        print(f"Gemini requests function  : {tool_call.name}")
        print(f"Gemini provided arguments : {tool_call.args}")

        # Step 3: You Execute
        if tool_call.name == "get_weather":
            target_city = tool_call.args.get("city")

            tool_result = get_weather(city=target_city)

            # Step 4: Model Continues
            print(f"\n[📤 SENDING TO GEMINI] Feeding result back: {tool_result}")

            final_response = client.models.generate_content(
                model='gemini-2.5-flash',
                # We pass the history so it remembers the conversation
                contents =[
                    "What's the weather like in Delhi?",
                    response.candidates[0].content,                         #The tool call request
                    types.Content(
                        role = "tool",
                        parts = [
                            types.Part.from_function_response(
                                name = "get_weather",
                                response = {"result":tool_result}
                            )
                        ]
                    )
                ]
            )
            print("\n✅ --- FINAL GEMINI ANSWER ---")
            print(final_response.text)

else:
    print("Gemini answered directly without tools:")
    print(response.text)

        





