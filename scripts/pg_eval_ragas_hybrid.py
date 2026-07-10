import os
import sys
import json
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import types

# 1. DEPENDENCY INTERCEPTOR
from langchain_google_vertexai import ChatVertexAI
fake_module = types.ModuleType('langchain_community.chat_models.vertexai')
fake_module.ChatVertexAI = ChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = fake_module

# 2. MODERN RAGAS IMPORTS (v0.2+)
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pg_hybrid_retriever import hybrid_retriever

load_dotenv()

def run_evaluation():
    print("🚀 Booting up GovPrep Evaluation Pipeline (RAGAS v0.2 Edition)...")

    with open("eval/gold_set.json", 'r', encoding='utf-8') as file:
        gold_set_data = json.load(file)

    test_questions = [item["question"] for item in gold_set_data]
    ground_truths = [item["required_keyword"] for item in gold_set_data]
    
    # --- 3. INIT MODELS ---
    raw_judge_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant", 
        temperature=0
    )
    raw_judge_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    # --- 4. WRAP MODELS FOR RAGAS v0.2 ---
    ragas_llm = LangchainLLMWrapper(raw_judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(raw_judge_embeddings)

    answers = []
    contexts = []

    print("🔍 Running queries through the Hybrid Search engine...")
    for query in test_questions:
        print(f" -> Evaluating: {query}")
        
        retrieved_docs = hybrid_retriever(query, k=3)
        context_list = [doc.page_content for doc in retrieved_docs]
        contexts.append(context_list)
        
        context_string = "\n\n".join(context_list)
        prompt = f"Answer using ONLY the context.\nContext: {context_string}\nQuestion: {query}\nAnswer:"
        
        try:
            response = raw_judge_llm.invoke(prompt)
            answers.append(response.content.strip())
        except Exception as e:
            print(f"    ⚠️ Error generating answer: {e}")
            answers.append("Error generating answer.")
        
        time.sleep(1) 

    # Prepare RAGAS dataset
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\n📊 Passing data to RAGAS LLM-as-a-Judge...")
    
    safe_config = RunConfig(max_workers=1, max_retries=10)

    # --- 5. INJECT WRAPPED MODELS INTO METRICS ---
    cp = ContextPrecision(llm=ragas_llm)
    cr = ContextRecall(llm=ragas_llm)
    faith = Faithfulness(llm=ragas_llm)
    ar = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[cp, cr, faith, ar],
        run_config=safe_config,
        raise_exceptions=False
    )

    print("\n✅ Evaluation Complete! Final Scorecard:")
    df = result.to_pandas()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[['question', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']])
    
    output_csv = "evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed results saved to {output_csv}")

if __name__ == "__main__":
    run_evaluation()