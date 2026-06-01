
# Roadmap
'''
- [x] v0 - single-document RAG over one ncert polity chapter with basic prompt (week 1)
- [ ] v1 - multi-document support + conversation memory (week 2)
- [ ] v2 - smarter chunking + web UI (week 3-4)
- [ ] v3 - deployed public URL with FastAPI backend (week 5-6)
'''

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent
sys.path.insert(0,str(base_dir/"scripts"))

try:
    from generate import answer_with_rag
except ImportError as e:
    print(f"Initialization error: could not find generation script. {e}")
    print("Ensure 'generate.py' and 'retrieve.py' are inside your 'scripts' directory.")
    sys.exit()

def print_banner():
    print("="*50)
    print("Welcome to GovPrep v0 - Your AI Study Buddy for Polity!")
    print("Ask me anything about the NCERT Polity chapter, and I'll do my best to help you out.")
    print("  Powered by Gemini 2.0 Flash + ChromaDB")
    print("="*50)
    print("Type your question, /help for commands, or 'quit' to exit.\n")

def print_help():
    print("\nAvailable commands:")
    print("  /help          - Show this menu")
    print("  /sources       - Show textbook chunks used in the last answer")
    print("  /k <number>    - Change retrieval context chunk count (e.g., /k 5)")
    print("  quit           - Exit the application cleanly")
    print()


def main():
    print_banner()
    last_chunks = []
    k=3
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye! Happy studying!")
                break

            if user_input in ("/help", "/?"):
                print_help()
                continue
            
            if user_input == "/sources":
                if not last_chunks:
                    print("No sources available. Ask a question first to retrieve relevant chunks.")
                else:
                    print(f"\n Sources used for last answer (top {len(last_chunks)} chunks):")
                    for i, chunk in enumerate(last_chunks):
                        print(f"\n-- [Passage {i+1}] (Distance Match Score: {chunk['distance']:.4f}) --")
                        print(chunk['text'].strip()[:300] + "...")
                    print()
                continue
            
            if user_input.startswith("/k "):
                try:
                    requested_k = int(user_input.split()[1])
                    if requested_k <= 0:
                        print("Please enter a positive integer for k.")
                        continue
                    k = requested_k
                    print(f"\n🎯 Context updated. Now retrieving top-{k} chunks for future questions.\n")
                except (IndexError, ValueError):
                    print("Invalid command format. Use /k <number> (e.g., /k 5).")
                continue

            print("\n🤖 Thinking...")
            result = answer_with_rag(user_input, k =k)
            last_chunks = result['chunks_used']

            print(f"\n Answer:\n{result['answer']}\n")
            print("-"*50)
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Type 'quit' to exit cleanly.")
            break
        except Exception as e:
             print(f"\n❌ Pipeline Error: {e}\n")

if __name__ == "__main__":
    load_dotenv(base_dir / ".env")

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ CRITICAL ERROR: GEMINI_API_KEY not detected.")
        print("Please check that your .env file exists in the root directory and contains your key.")
        sys.exit(1)
    main()