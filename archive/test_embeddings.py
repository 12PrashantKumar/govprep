from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# load the local model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Test-1. single sentence embedding

sentence = "The President of India is the head of state."
embedding = model.encode(sentence)
print(f"Sentence: {sentence}")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values of the embedding: {embedding[:10]}")
print(f"type of embedding: {type(embedding)}")
print(f"\n" + "-"*50 + "\n")

# Test-2. show similarity between  sentences
sentences = [
    "Who is the prime minister of India?",
    "India's PM",
    "The PM of India",
    "I love eating mangoes",
    "Mango is my favorite fruit",
    "The capital of France is Paris",
]
embeddings = model.encode(sentences)

# Calculate cosine similarity between the first sentence and the others
similarity_matrix = cos_sim(embeddings,embeddings)

print("Similarity matrix (1.0 = identical meaning, 0.0 = unrelated):\n")
print(f"{'' :<40}",end = "")
for i in range(len(sentences)):
    print(f"S{i}        ",end="")
print()

for i, sentnce in enumerate(sentences):
    print(f"S{i}: {sentnce[:36]:<40}",end="")
    for j in range(len(sentences)):
        print(f"{similarity_matrix[i][j]:.3f}      ",end=" ")
    print()

