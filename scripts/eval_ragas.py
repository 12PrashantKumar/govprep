import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, _answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agent import answer_agentic

load_dotenv()

# Initialize Gemini for RAGAS (It needs both an LLM to judge, and an embedding model to calculate relevancy)
ragas_llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.0)
ragas_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

def run_ragas_evaluation(gold_set_path = "eval/gold_set.json"):
    if not os.path.exists(gold_set_path):
        print(f" {gold_set_path} not found. Ensure you are in the root directory")
        return
    
    with open(gold_set_path,"r") as f:
        gold_set = json.load(f)


    # We will just test the first 3 questions to save time and API tokens during testing
    test_subset = gold_set[:3]

    data_samples = {
        "question" :[],
        "answer" :[],
        "context":[]
    }

    print(f" Generating answers for {len(test_subset)} questions...")
    for item in test_subset:
        question  = item["question"]

        # run ReAct agent
        agent_output = answer_agentic(question)
        answer = agent_output["answer"]
        context_string = agent_output["context"]

        # RAGAS requires contexts as a list of strings
        context_list = [c.strip for c in context_string.split("[Source:") if c.strip()]
        context_list = [f"[Source: {c}" for c in context_list]   #Add the prefox back

        # If no context was found, provide an empty list to avoid RAGAS crashing
        if not context_list or "No context retrieved" in context_string:
            context_list = [""]

        data_samples["question"].append(question)
        data_samples["answer"].append(answer)
        data_samples["context"].append(context_list)

    # Convert to HuggingFAce Dataset
    dataset = Dataset.from_dict(data_samples)

    print("\n Running RAGAS Evaluation (Faithfulness & Answer Relevancy)...")

    # RAGAS Evaluation
    score = evaluate(
        dataset,
        metrics = [faithfulness,_answer_relevancy],
        llm = ragas_llm,
        embeddings=ragas_embeddings
    )

    # Print final Pandas DataFrame results
    print("\n✅ Evaluation Complete! Here are the scores:")
    print(score.to_pandas()[['question', 'faithfulness', 'answer_relevancy']])


if __name__ == "__main__":
    run_ragas_evaluation()