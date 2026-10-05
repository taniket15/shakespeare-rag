import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.data_loader import load_all_documents
from src.retriever import HybridRetriever
from src.vectorstore import FaissVectorStore

load_dotenv()

TOP_K = 8  # passages per answer; 8 beat 5 on completeness and faithfulness in evals/answer_eval.py
HISTORY_MESSAGES = 6  # last 3 question/answer pairs are enough to resolve follow-ups

SYSTEM_PROMPT = """You answer questions about Shakespeare's plays and poems using only the numbered passages provided.

- Base every statement on the passages. Do not add facts from memory, even if you know them.
- Cite the passages you used with their numbers, like [1] or [2][3], right after the statement they support.
- Each passage starts with the work and section it comes from (e.g. "Macbeth, Act 1, Scene 7"); use this to name where things happen.
- Answer completely. Combine the relevant details from all the passages: who did what, how, why, and what
  happened as a result. Include specific names, objects and events (e.g. a letter, a casket, a poison).
- If the passages don't contain the answer, say so plainly and briefly describe what they do cover. Don't guess.
- Length: 2 to 4 sentences for a focused question; up to two short paragraphs for a plot summary."""

CONDENSE_PROMPT = """Rewrite the user's latest question as a standalone question about Shakespeare's works.

Use the conversation to resolve references like "she", "that play", "what about her cousin?" or "and in Act 2?",
naming the work and characters explicitly. If the question is already standalone, return it unchanged.
Keep any quoted lines exactly as written. Return only the rewritten question."""


class RAGSearch:
    """Retrieves relevant chunks (hybrid vector + keyword search) and answers from them with an OpenAI LLM."""

    def __init__(self, persist_dir: str = "faiss_store", data_dir: str = "data",
                 embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-6-luna"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        if self.vectorstore.exists():
            self.vectorstore.load()
        else:
            self.vectorstore.build_from_documents(load_all_documents(data_dir))
        self.retriever = HybridRetriever(self.vectorstore)

        llm_model = os.getenv("OPENAI_MODEL", llm_model)
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def condense(self, query: str, history: list[dict]) -> str:
        """Turn a follow-up question into a standalone one using the recent conversation."""
        if not history:
            return query
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_MESSAGES:])
        messages = [
            ("system", CONDENSE_PROMPT),
            ("human", f"Conversation:\n{conversation}\n\nLatest question: {query}"),
        ]
        return self.llm.invoke(messages).content.strip()

    def answer(self, query: str, top_k: int = TOP_K, history: list[dict] | None = None) -> tuple[str, list[dict], str]:
        """Answer from the top_k retrieved passages only.

        history is the earlier chat as [{"role": "user" | "assistant", "content": ...}]; follow-ups are
        rewritten into standalone questions before retrieval. Returns (answer, sources, standalone question).
        """
        query = self.condense(query, history or [])
        results = self.retriever.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found.", [], query

        passages = "\n\n".join(f"[{i}] {r['metadata']['text']}" for i, r in enumerate(results, 1))
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Passages:\n{passages}\n\nQuestion: {query}"),
        ]
        return self.llm.invoke(messages).content, results, query

    def search_and_summarize(self, query: str, top_k: int = TOP_K) -> str:
        return self.answer(query, top_k)[0]


if __name__ == "__main__":
    rag_search = RAGSearch()
    print("Summary:", rag_search.search_and_summarize("What happens in As You Like It?"))
