import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

client  =  genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# The prompt matrix that instructs Gemini to act as a linguistic translator
REWRITE_PROMPT = """Given the conversation history and a follow-up \
question, rewrite the follow-up into a standalone question that makes \
sense without the history. Resolve all pronouns (it, they, their, that, he, she, this). \
If the question is already standalone, return it unchanged.

Return ONLY the rewritten question string, nothing else. Do not add introductions or explanations.

Conversation history:
{history}

Follow-up question: {question}
Standalone question:"""

def rewrite_followup(question, memory):
    """
    Evaluates history and context. If a pronoun or vague reference is used, 
    calls Gemini to reshape the question into a direct database-searchable string.
    """
    if not memory.turns:
        return question
    
    prompt = REWRITE_PROMPT.format(
        history = memory.as_text(),
        question = question

    )
 
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents = prompt

    )

    return response.text.strip()

if __name__ == "__main__":
    from memory import ConversationMemory

    mem = ConversationMemory()
    mem.add_turn("Who is the President of India?", "The President is the head of state.")

    print("="*60)
    print("RUNNING CONTEXT REWRITE EVALUATION")
    print("="*60)

    # Test Scenario A: Highly ambiguous pronoun follow-up
    q1 = "what are thier powers?"
    print(f"Original question: {q1}")
    print(f"Rewritten question:{rewrite_followup(q1,mem)}")

    # Update memory state to simulate a deeper conversation branch
    mem.add_turn(q1, "The President commands the armed forces and has executive control...")

    # Test Scenario B: Vague location follow-up dependent on historical context
    q2 = "where do they live?"
    print(f"Original Q2:  '{q2}'")
    print(f"Rewritten Q2: '{rewrite_followup(q2, mem)}'\n")

    # Test Scenario C: Already self-contained question (Should pass through unchanged)
    q3 = "What is the capital of France?"
    print(f"Original Q3:  '{q3}'")
    print(f"Rewritten Q3: '{rewrite_followup(q3, mem)}'")


