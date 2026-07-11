import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

conn_info = "postgresql://postgres:dev@localhost:5433/postgres"

def test_db_plumbing():
    print("Connecting to Postgres...")

    with psycopg.connect(conn_info) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            # 1. Insert a hand-made row with a dummy 768-dimension vector
            dummy_vector = [0.1]*768

            print("INSERTING test chunks...")
            cur.execute("""
                        INSERT INTO chunks( content ,subject, source, embedding)
                        VALUES (%s,%s,%s,%s)
                        RETURNING id;
            """, ("The President of India is the head of state.", "Polity", "test.txt", dummy_vector))

            inserted_id = cur.fetchone()[0]
            print(f"Success! Inserted row with ID: {inserted_id}")

            # 2. Run a Vector Search
            print("Running vector search...")
            search_vector = [0.1] * 768

            cur.execute(""" 
                SELECT id, content, subject FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
            """,(search_vector,))

            result= cur.fetchone()
            print(f"Search Result: {result}")
            
            # Rollback so we don't pollute our real database with fake data
            conn.rollback() 
            print("Test complete. Database plumbing is working perfectly.")

if __name__ == "__main__":
    test_db_plumbing()