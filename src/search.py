import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    """Retrieves relevant chunks from the FAISS store and summarizes them with an OpenAI LLM."""

    def __init__(self, persist_dir: str = "faiss_store", data_dir: str = "data",
                 embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-6-luna"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        if self.vectorstore.exists():
            self.vectorstore.load()
        else:
            self.vectorstore.build_from_documents(load_all_documents(data_dir))

        llm_model = os.getenv("OPENAI_MODEL", llm_model)
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=llm_model, max_tokens=1024)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        context = "\n\n".join(r["metadata"]["text"] for r in results)
        if not context:
            return "No relevant documents found."

        prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
        return self.llm.invoke(prompt).content


if __name__ == "__main__":
    rag_search = RAGSearch()
    print("Summary:", rag_search.search_and_summarize("What is machine learning?", top_k=3))
