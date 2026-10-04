from typing import Any

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    """Splits documents into chunks and embeds them with a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: list[Any]) -> list[Any]:
        chunks = self.splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        """Embed texts as float32, the dtype FAISS expects."""
        return self.model.encode(texts, show_progress_bar=show_progress_bar).astype("float32")

    def embed_chunks(self, chunks: list[Any]) -> np.ndarray:
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embed([chunk.page_content for chunk in chunks], show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(load_all_documents("data"))
    embeddings = pipeline.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) else None)
