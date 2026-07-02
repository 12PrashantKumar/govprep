import os
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self,chunks):
        """
        Initialize the keyword search engine.
        'chunks' must be a list of dictionaries, e.g:[{'text':'Article 21 states...', 'sorce':'polity','page':45},...]
        """
        self.chunks = chunks
        print("Tokenizing corpus and building BM25 index...")

        # BM25 needs the text broken down into individual words (tokens)
        self.tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f" Successfully indexed {len(self.chunks)} chunks for keyword matching.")

    def _tokenize(self, text:str)->list:
        """
        A basic tokenizer: converts text to lowercase and splits by whitespace.

        """
        return text.lower().split(" ")
    
    def search(self, query:str, k:int =5)-> list:
        """
        Takes a user query, scores all chunks for exact keyword matches, and returns the top k results.
        """
        tokenized_query = self._tokenize(query)

        # Calculate BM25 scores for every chunk in the database
        doc_scores = self.bm25.get_scores(tokenized_query)


        # Zip the chunks and their scores together, then sort them highest to lowest
        ranked_chunks = sorted(
            zip(self.chunks, doc_scores),
            key = lambda x:x[1],
            reverse=True
        )

        # Extract just the chunk dictionaries for the top k results (ignoring scores of 0)

        top_results = [chunk for chunk,score in ranked_chunks[:k] if score >0]
        return top_results
    


# testing

if __name__ == "__main__":
    mock_corpus = [
        {"text": "The fundamental rights are enshrined in Part III of the Constitution.", "source": "polity", "page": 12},
        {"text": "Article 21 provides the right to life and personal liberty.", "source": "polity", "page": 45},
        {"text": "The Directive Principles of State Policy are non-justiciable.", "source": "polity", "page": 50},
        {"text": "Article 21A guarantees the right to education for children.", "source": "polity", "page": 46}
    ]

    # initialize the engine
    keyword_engine = BM25Retriever(chunks=mock_corpus)

    # test an exact-match keyword search
    test_query = "Article 21"
    print(f"\n🔍 Searching for: '{test_query}'")

    results = keyword_engine.search(test_query, k=3)

    for i , res in enumerate(results,1):
        print(f"\n[{i}] Source: {res['source']} | Page: {res['page']}")
        print(f"    Text: {res['text']}")