import os, psycopg
from dotenv import load_dotenv
load_dotenv()

c = psycopg.connect(os.getenv("DATABASE_URL"))
cur = c.cursor()

cur.execute("SELECT subject, count(*) FROM chunks GROUP BY subject ORDER BY subject")
print("--- BY SUBJECT ---")
for subject, n in cur.fetchall():
    print(f"{n:>5}  {subject}")

cur.execute("SELECT source, count(*) FROM chunks GROUP BY source ORDER BY source")
print("\n--- BY SOURCE ---")
for source, n in cur.fetchall():
    print(f"{n:>5}  {source}")

cur.execute("SELECT count(*) FROM chunks")
print("\nTOTAL:", cur.fetchone()[0])