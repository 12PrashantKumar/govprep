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
    

if __name__ == "__main__":
    # Mocking a quick test run to see the mechanism work
    mock_question = "What is the age limit for the President?"
    mock_context = "[Source: NCERT Ch 3] Article 58 states that a citizen must be at least 35 years old to contest the Presidential election."
    
    # Test 1: Perfect Answer (Should be a 5)
    print("--- Testing Perfect Answer ---")
    good_answer = "According to Article 58, a person must be at least 35 years old to become the President."
    print(judge_faithfullness(mock_question, mock_context, good_answer))
    
    # Test 2: Hallucinated Answer (Should be a 1 or 2)
    print("\n--- Testing Hallucinated Answer ---")
    bad_answer = "The minimum age is 35 years old, and they must also have a degree from Delhi University."
    print(judge_faithfullness(mock_question, mock_context, bad_answer))