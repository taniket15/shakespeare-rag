import os
import pickle
from typing import Any

import faiss

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    """FAISS index of document chunks, persisted to disk with the chunk texts."""

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "faiss.index")
        self.meta_path = os.path.join(persist_dir, "metadata.pkl")
        self.pipeline = EmbeddingPipeline(embedding_model, chunk_size, chunk_overlap)
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)

    def build_from_documents(self, documents: list[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")
        chunks = self.pipeline.chunk_documents(documents)
        embeddings = self.pipeline.embed_chunks(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = [{"text": chunk.page_content, **chunk.metadata} for chunk in chunks]
        self.save()

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved {self.index.ntotal} vectors to {self.persist_dir}")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded {self.index.ntotal} vectors from {self.persist_dir}")

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k closest chunks (smaller distance = more similar)."""
        query_emb = self.pipeline.embed([query_text])
        distances, indices = self.index.search(query_emb, top_k)
        return [
            {"index": int(idx), "distance": float(dist), "metadata": self.metadata[idx]}
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1  # FAISS pads with -1 when the index has fewer than top_k vectors
        ]


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    store = FaissVectorStore()
    store.build_from_documents(load_all_documents("data"))
    for result in store.query("What happens in As You Like It?", top_k=3):
        print(result["distance"], result["metadata"]["text"][:200])
