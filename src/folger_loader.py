import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

FOLGER_SUFFIX = "_PDF_FolgerShakespeare.pdf"

# Slugs whose titles can't be rebuilt from the file name alone (apostrophes)
TITLE_OVERRIDES = {
    "alls-well-that-ends-well": "All's Well That Ends Well",
    "a-midsummer-nights-dream": "A Midsummer Night's Dream",
    "loves-labors-lost": "Love's Labor's Lost",
    "shakespeares-sonnets": "Shakespeare's Sonnets",
    "the-winters-tale": "The Winter's Tale",
}
SMALL_WORDS = {"a", "and", "as", "for", "in", "of", "that", "the", "to"}
ROMAN = {"ii", "iii", "iv", "v", "vi", "viii"}

RUNNING_HEADER = re.compile(r"^ACT (\d+)\. SC\. (\d+)$")
ACT_HEADING = re.compile(r"^ACT (\d+)$")
SCENE_HEADING = re.compile(r"^Scene (\d+)$")
OTHER_HEADINGS = {"PROLOGUE": "Prologue", "THE PROLOGUE": "Prologue", "EPILOGUE": "Epilogue", "INDUCTION": "Induction"}
FRONT_MATTER_KEPT = ("Synopsis", "Characters in the Play")


def is_folger_pdf(path: Path) -> bool:
    return path.name.endswith(FOLGER_SUFFIX)


def title_from_path(path: Path) -> str:
    slug = path.name.removesuffix(FOLGER_SUFFIX)
    if slug in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[slug]
    slug = re.sub(r"-part-(\d)$", r" part \1", slug)
    words = slug.replace("-", " ").split()
    out = []
    for i, w in enumerate(words):
        if w in ROMAN:
            out.append(w.upper())
        elif i and w in SMALL_WORDS:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out).replace(" Part ", ", Part ")


def _page_lines(page) -> list[dict]:
    """Text lines on a page with position, size and italic flag, in reading order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            span = line["spans"][0]
            lines.append({
                "x": line["bbox"][0],
                "y": line["bbox"][1],
                "size": round(span["size"], 1),
                "italic": "It" in span["font"],
                "text": text,
            })
    # Speaker names sit either just above their first line (verse) or on it, ~2pt lower (prose),
    # so sort them as if 4pt higher to always land right before the line they introduce
    return sorted(lines, key=lambda ln: (round(ln["y"] - (4 if _is_speaker(ln) else 0)), ln["x"]))


def _is_speaker(line: dict) -> bool:
    return line["size"] == 10 and line["text"].upper() == line["text"] and line["x"] < 110


def _body_start(doc) -> int:
    """Index of the first page after the Folger ad, contents, director's note and textual introduction.

    In every Folger PDF the textual introduction runs 2 pages, followed by the Synopsis (plays)
    or the poem itself.
    """
    for i in range(min(15, doc.page_count)):
        if any(ln["size"] >= 14 and ln["text"] == "Textual Introduction" for ln in _page_lines(doc[i])):
            return i + 2
    return 0


def load_folger_pdf(path: str) -> list[Document]:
    """Load a Folger Shakespeare PDF as one document per scene, sonnet or front-matter section.

    Drops Folger boilerplate, FTLN markers, line numbers and running headers; attaches speaker
    names to their lines and marks stage directions with [brackets].
    """
    path = Path(path)
    title = title_from_path(path)
    is_sonnets = path.name.startswith("shakespeares-sonnets")
    doc = pymupdf.open(path)

    sections: list[tuple[str, int, list[str]]] = []  # (section, first page, text parts)
    act = None

    def start_section(name: str, page_no: int):
        if not sections or sections[-1][0] != name:
            sections.append((name, page_no, []))

    for page_no in range(_body_start(doc), doc.page_count):
        lines = _page_lines(doc[page_no])
        pending_speaker = None

        for line in lines:
            text, size, x = line["text"], line["size"], line["x"]

            if size >= 14 and text in FRONT_MATTER_KEPT:
                start_section(text, page_no + 1)
                continue
            if RUNNING_HEADER.match(text):
                continue  # names the scene starting on this page, not the one the page starts in
            if m := ACT_HEADING.match(text):
                act = m.group(1)
                continue
            if m := SCENE_HEADING.match(text):
                part = f"Act {act}" if act else "Induction"  # The Taming of the Shrew opens with an induction
                start_section(f"{part}, Scene {m.group(1)}", page_no + 1)
                continue
            if size >= 13 and x > 120 and text in OTHER_HEADINGS:  # 10pt PROLOGUE at the margin is a character
                start_section(OTHER_HEADINGS[text], page_no + 1)
                continue
            if size <= 8 or text.startswith("FTLN"):
                continue  # FTLN line markers
            if is_sonnets and text.isdigit() and 150 < x < 400:
                start_section(f"Sonnet {text}", page_no + 1)
                continue
            if text.isdigit():
                continue  # page and line numbers
            if line["y"] < 80 and line["italic"] and size >= 14:
                continue  # running header title

            if not sections:
                start_section(title, page_no + 1)  # poems without headings

            parts = sections[-1][2]
            if sections[-1][0] == "Characters in the Play":
                parts.append(f"\n{text}" if text[:1].isupper() else text)
                continue
            if sections[-1][0] in FRONT_MATTER_KEPT:
                parts.append(text)
                continue
            if _is_speaker(line):
                pending_speaker = text
                continue
            if line["italic"] and pending_speaker and text.startswith(","):
                # "LYSANDER, to Theseus": a direction attached to the speaker's name
                pending_speaker = f"{pending_speaker} [{text.lstrip(', ')}]"
                continue
            if line["italic"]:
                text = f"[{text}]"
            if pending_speaker:
                parts.append(f"\n{pending_speaker}: {text}")
                pending_speaker = None
            else:
                parts.append(text)

    documents = []
    for section, first_page, parts in sections:
        content = re.sub(r"\] \[", " ", " ".join(parts).replace(" \n", "\n")).strip()
        if not content:
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "source": str(path),
                "title": title,
                "section": section,
                "page": first_page,
                "heading": section if section == title else f"{title}, {section}",
            },
        ))
    return documents
