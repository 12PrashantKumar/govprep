import os
from pathlib import Path
import time
from dotenv import load_dotenv
from google import genai


from retrieve_multi import retrieve
from rewrite import rewrite_followup

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """"You are a UPSC/CDS exam prep assistant answering from \
NCERT textbooks. Rules:
1. Answer ONLY from the passages below.
2. If the passages don't contain the answer, say "I don't have \
   that in my sources."
3. Cite the source of each fact like [polity] or [history] or [geography].
4. Use the conversation history for context, but base facts on passages.
5. Be concise - 2-4 sentences."""

def build_prompt(question, chunks, memory):

    """Compile history, explicitly marked metadata passages, and query into one matrix."""
    passages = ""
    for c in chunks:
        passages += f"\n[{c['source']} p{c['page']}] {c['text']}\n"
    return f"""{SYSTEM_PROMPT}

Conversation so far:
{memory.as_text()}

Passages:
{passages}

Current question: {question}
Answer:"""

def answer(question, memory):
    """
    Executes the entire v1 execution tree:
    1. Rewrite ambiguous query via context memory.
    2. Retrieve top-k semantic matches with metadata mapping.
    3. Construct highly constrained augmented prompt.
    4. Fire generation call to Gemini.
    5. Cache step data back into local state memory.
    """
    # Rewrite using history
    standalone = rewrite_followup(question, memory)
    # Retrieve across all sources
    chunks = retrieve(standalone,k=3, collection_name="govprep_v2")
    
    # build prompt with history + chunks
    prompt = build_prompt(standalone, chunks, memory)

    # generate response using model
    response = client.models.generate_content(
        model= "gemini-2.5-flash",
        contents = prompt
    )

    ans = response.text.strip()

    memory.add_turn(question, ans)
    return {"answer": ans, "rewritten": standalone, "chunks": chunks}

if __name__ == "__main__":
    from memory import ConversationMemory
    mem = ConversationMemory()

    # A progressive, multi-turn dialogue thread
    conversation_flow =[
        "What are fundamental rights?",
        "How many are there?",        
        "Give me an example of one"
    ]

    print("="*60)    
    print("STARTING FULL V1 PIPELINE EVALUATION SUITE")
    print("="*60)

    for i,q in enumerate(conversation_flow):
        print(f"\nUser Query: {q}")

        try:
            r = answer(q, mem)
            print(f"Internal Rewriter Output: '{r['rewritten']}'")
            print(f"AI Response:\n{r['answer']}")
        except Exception as e:
            print(f"Error during processing: {e}")
        
        # Pacing throttle to shield free tier quotas from spike drops
        if i < len(conversation_flow) - 1:
            print("\nPacing pipeline... Waiting 5 seconds before subsequent prompt submission...")
            time.sleep(5)


