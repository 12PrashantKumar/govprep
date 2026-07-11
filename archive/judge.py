import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

judge_llm  = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.0)

def judge_faithfullness(question:str, context:str, answer:str) -> str:
    """
    Evaluate whether the generated answer is compeletly grounded in the retrieved context.
    Returns a score from 1-5 and a brief reason.

    """

    JUDGE_PROMPT = f""" You are an elite quality assurance judge evaluating a RAG(Retrieval-Augmented Generation) system.
    Your task is to determine if the generated answer is fully supported by the retrieved textbook context.
    
    Question :{question}
    Retrieved context : {context}
    Answer given : {answer}

    Rules for Grading:
    -Look only at the provided 'Retrieved context'. Do not use your own external Knowledge.
    -Rate 5: Every single claim in the answer is fully supported and tracebale to the context.
    -Rate 1: The answer completely makes things up or contains facts missing from the context.

    You must strictly respond in this exact  format:
    SCORE : <number between 1 and 5>
    REASON : <one short sentence explaining your score>
    """

    try:
        response = judge_llm.invoke([HumanMessage(content=JUDGE_PROMPT)])
        return response.content
    except Exception as e:
        return f"SCORE: Error\nREASON: Could not run judge - {str(e)}"
    

# =====================================================================
# RUNNING THE EVALUATION ON YOUR GOLD SET
# =====================================================================
if __name__ == "__main__":
    from agent import answer_agentic
    
    gold_set_path = "eval/gold_set.json" 
    
    # Fallback template if  gold_set.json is located in a different directory
    if not os.path.exists(gold_set_path):
        print(f" {gold_set_path} not found in current directory. Using sample evaluation array...")
        gold_set = [
            {"question": "What are the fundamental rights in the Indian Constitution?"},
            {"question": "Explain the difference between fundamental rights and directive principles."},
            {"question": "What is the minimum age required to become the President of India?"}
        ]
    else:
        with open(gold_set_path, "r") as f:
            gold_set = json.load(f)

    print(f"🚀 Starting Evaluation on {len(gold_set)} questions...\n")

    # Run the loop
    for i, item in enumerate(gold_set, 1):
        question = item["question"]
        print(f"==================================================")
        print(f"📊 [TEST {i}/{len(gold_set)}] Question: {question}")
        print(f"==================================================")
        
        # 1. Run the live GovPrep agent
        agent_output = answer_agentic(question)
        answer = agent_output["answer"]
        context = agent_output["context"]
        
        # 2. Run the custom judge evaluation
        print("\n⚖️ [JUDGING GENERATION QUALITY...]")
        judgment = judge_faithfullness(question, context, answer)
        
        print(f"\n{judgment}\n")
        
    print("✅ Gold Set Evaluation Complete.")