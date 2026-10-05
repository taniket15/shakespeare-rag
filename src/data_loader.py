from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from src.folger_loader import is_folger_pdf, load_folger_pdf

# File extension -> function that builds a loader for that file
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".json": lambda path: JSONLoader(path, jq_schema=".", text_content=False),
}


def load_all_documents(data_dir: str) -> list[Any]:
    """Load every supported file under data_dir (recursively) as LangChain documents."""
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Loading documents from {data_path}")
    documents = []

    for ext, make_loader in LOADERS.items():
        files = sorted(data_path.glob(f"**/*{ext}"))
        if files:
            print(f"[INFO] Found {len(files)} {ext} files")
        for file in files:
            try:
                if is_folger_pdf(file):
                    loaded = load_folger_pdf(str(file))
                else:
                    loaded = make_loader(str(file)).load()
                documents.extend(loaded)
                print(f"[INFO] Loaded {len(loaded)} docs from {file.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {file.name}: {e}")

    print(f"[INFO] Total loaded documents: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print("Example document:", docs[0] if docs else None)
