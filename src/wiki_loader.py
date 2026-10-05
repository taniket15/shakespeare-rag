import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

BODY_SIZE = 9.5  # body text is 10pt; captions (8pt) and notes (6pt) are smaller
SECTION_SIZE = 14  # "Life", "Plays", ... (subsections are 12pt)
END_SECTIONS = {"Notes", "References", "External links", "Article Sources and Contributors"}
CITATION = re.compile(r"\[\d+\]")


def is_wikipedia_pdf(path: Path) -> bool:
    """Wikipedia's 'Download as PDF' export, recognisable by the mwlib renderer in its metadata."""
    if path.suffix.lower() != ".pdf":
        return False
    with pymupdf.open(path) as doc:
        return "mwlib" in (doc.metadata.get("keywords") or "")


def load_wikipedia_pdf(path: str) -> list[Document]:
    """Load a Wikipedia PDF export as one document per section, without notes, references,
    contributor lists, image captions, running headers or [n] citation markers."""
    path = Path(path)
    doc = pymupdf.open(path)
    article = None
    sections: list[tuple[str, int, list[str]]] = []  # (section, first page, text parts)
    section = subsection = None

    for page_no, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                span = line["spans"][0]
                size, bold = span["size"], "Bold" in span["font"]
                if not text or line["bbox"][1] < 50:
                    continue  # running header: article title and page number
                if size >= 20 and article is None:
                    article = text
                    continue
                if bold and size >= SECTION_SIZE - 0.5:
                    if text in END_SECTIONS:
                        return _documents(path, article, sections)
                    section, subsection = text, None
                elif bold and size >= 11.5 and text != article:  # the infobox repeats the title
                    subsection = text
                elif size >= BODY_SIZE:
                    name = " › ".join(p for p in (section, subsection) if p) or "Introduction"
                    if not sections or sections[-1][0] != name:
                        sections.append((name, page_no + 1, []))
                    sections[-1][2].append(CITATION.sub("", text))

    return _documents(path, article, sections)


def _documents(path: Path, article: str | None, sections) -> list[Document]:
    title = f"{article or path.stem} (biography)"
    documents = []
    for name, first_page, parts in sections:
        # Lines wrap mid-sentence; rejoin them, mending words hyphenated across lines
        content = re.sub(r"-\s+(?=[a-z])", "", " ".join(parts))
        content = re.sub(r"\s+([.,;:])", r"\1", content).strip()
        if len(content) < 40:
            continue
        documents.append(Document(
            page_content=content,
            metadata={"source": str(path), "title": title, "section": name, "page": first_page,
                      "heading": f"{title}, {name}"},
        ))
    return documents
