import os, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors
base = Path(__file__).resolve().parent.parent
load_dotenv(base / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def call_llm(prompt, model="gemini-2.5-flash", max_retries=4):
    for attempt in range(max_retries):
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            return r.text
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (attempt + 1) * 20
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Still rate-limited after retries.")