import sys

from src.search import RAGSearch

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What happens in As You Like It?"
    rag_search = RAGSearch()
    print("Summary:", rag_search.search_and_summarize(query, top_k=3))
