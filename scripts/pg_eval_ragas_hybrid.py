import os, json, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pg_hybrid_retriever import hybrid_retriever

load_dotenv()
judge = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

with open("eval/gold_set.json", encoding="utf-8") as f:
    gold = json.load(f)[:20]   # 20 questions = solid, fast

scores = []
for item in gold:
    q = item["question"]
    docs = hybrid_retriever(q, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    answer = judge.invoke(
        f"Answer using ONLY this context.\nContext: {context}\nQuestion: {q}\nAnswer:"
    ).content.strip()

    verdict = judge.invoke(
        "You are grading a RAG answer for FAITHFULNESS.\n"
        f"Context: {context}\nAnswer: {answer}\n"
        "Is every claim in the answer supported by the context? "
        "Reply with ONLY a number 1-5 (5 = fully grounded, 1 = mostly made up)."
    ).content.strip()

    try:
        s = int("".join(c for c in verdict if c.isdigit())[:1])
    except:
        s = 0
    scores.append(s)
    print(f"[{s}/5] {q[:60]}")
    time.sleep(2)

valid = [s for s in scores if s > 0]
print(f"\n✅ Faithfulness: {sum(valid)/len(valid):.2f}/5 over {len(valid)} questions")