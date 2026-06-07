import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Get the exact folder where govprep_v1.py lives
current_folder = os.path.dirname(os.path.abspath(__file__))

# 2. Tell Python exactly where the 'scripts' folder is
scripts_folder = os.path.join(current_folder, "scripts")

# 3. Force Python to look inside 'scripts' before it does anything else
sys.path.insert(0, scripts_folder)

from memory import ConversationMemory
try:
    from generate_v1 import answer
except ImportError as e:
    print(f"Initialization Error: {e}")
    print("Ensure 'generate_v1.py' is inside your 'scripts' directory.")
    sys.exit(1)

def banner():
    print("="*60)
    print("  govprep_v1 - government exam Assistant (Multi-Doc + Memory)")
    print("  Sources: NCERT Polity, History, Geography")
    print("  Powered by Gemini 2.0 Flash + ChromaDB")
    print("=" * 60)
    print("Commands: /sources, /reset, /history, quit\n")

def main():
    banner()
    # Initialize the memory block with our 10-turn safety limit
    memory = ConversationMemory()
    last = None

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit"):
                print("Exiting govprep. Goodbye!")
                break
            if q == "/reset":
                memory = ConversationMemory()
                print("Memory reset. Starting fresh.")
                continue
            if q == "/history":
                print("Conversation History:")
                print("="*40)
                print(memory.as_text())
                print("="*40+ "\n")
            if q == "/sources":
                if last and last.get("chunks"):
                    print("\n📚 Sources used for last answer:")
                    for c in last["chunks"]:
                        print(f"  [{c['source'].upper()} - Page {c['page']}] (Match dist: {c['distance']:.3f})")
                        print(f"  Text: {c['text'][:120].strip()}...\n")
                else:
                    print("No sources to display yet. Ask a question first!")
                continue
            # Process the actual question
            print("🤖 Thinking (rewriting & searching)...")
            last = answer(q, memory)
            print(f"\nAI: {last['answer']}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Type 'quit' to exit cleanly.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Load environment variables safely
    load_dotenv(dotenv_path=os.path.join(current_folder, ".env"))

    if  not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)
    main()







