import os
import pickle
from typing import Any

import faiss

from src.embedding import EmbeddingPipeline


def _overlap(text: str, chunk: str, max_chars: int = 400) -> int:
    """Length of the text's ending that chunk starts with: the splitter's chunk_overlap.

    Matches only whole words at both ends, so a chunk starting "then" doesn't overlap a text ending "the".
    """
    for k in range(min(len(text), len(chunk), max_chars), 0, -1):
        whole_words = (k == len(chunk) or chunk[k].isspace()) and (k == len(text) or text[-k - 1].isspace())
        if whole_words and text.endswith(chunk[:k]):
            return k
    return 0


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

    def section_text(self, index: int) -> tuple[str, int, int]:
        """The whole section (scene, sonnet, synopsis...) that chunk index belongs to, rebuilt from its chunks.

        Returns (text, start, end): the text without the heading line, and where the chunk sits in it.
        A chunk without a section (generic loaders) is returned on its own.
        """
        meta = self.metadata[index]

        def same(i: int) -> bool:
            m = self.metadata[i]
            return m.get("source") == meta.get("source") and m.get("section") == meta.get("section")

        first = last = index
        if meta.get("section") is not None:  # a section's chunks are stored next to each other, in order
            while first > 0 and same(first - 1):
                first -= 1
            while last + 1 < len(self.metadata) and same(last + 1):
                last += 1

        text, start, end = "", 0, 0
        for i in range(first, last + 1):
            chunk = self.metadata[i]["text"]
            if self.metadata[i].get("heading"):
                chunk = chunk.split("\n", 1)[-1]
            chunk = chunk.strip()
            overlap = _overlap(text, chunk)
            if overlap:  # the chunk repeats the end of the text so far
                chunk_start, text = len(text) - overlap, text + chunk[overlap:]
            else:
                text = f"{text}\n{chunk}" if text else chunk
                chunk_start = len(text) - len(chunk)
            if i == index:
                start, end = chunk_start, len(text)
        return text, start, end

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
