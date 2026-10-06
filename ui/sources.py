"""The cited-sources panel under each answer."""
import html
import re
from collections.abc import Callable
from itertools import pairwise

import streamlit as st

EXCERPT_CHARS = 300
# A speech's speaker at the start of a line, with an optional direction: "CELIA [as Aliena]:"
SPEAKER = re.compile(r"^([A-Z][A-Z0-9 .,'’&-]*?)(\s\[[^\]]*\])?:")

# chunk index -> (section text, start, end of the chunk in it), from FaissVectorStore.section_text
SectionText = Callable[[int], tuple[str, int, int]]


def to_sources(results: list[dict]) -> list[dict]:
    """What the panel shows for each retrieved passage (kept in the chat history)."""
    sources = []
    for r in results:
        # Drop the heading line every chunk starts with; it's already shown above the excerpt
        text = r["metadata"]["text"].split("\n", 1)[-1].strip()
        sources.append({
            "index": r["index"],
            "heading": r["metadata"].get("heading", r["metadata"].get("source", "")),
            "page": r["metadata"].get("page", "?"),
            "excerpt": text[:EXCERPT_CHARS].strip(),
        })
    return sources


def section_html(text: str, start: int, end: int) -> str:
    """One paragraph per speech, speakers in bold, and the cited passage (text[start:end]) highlighted."""
    paragraphs, offset = [], 0
    for line in text.split("\n"):
        line_start, line_end = offset, offset + len(line)
        offset = line_end + 1
        speaker = SPEAKER.match(line)
        # Cut the line where the speaker name ends and where the highlight starts and ends,
        # then wrap each piece on its own so the tags always nest
        cuts = sorted({0, len(line), *(c - line_start for c in (start, end) if line_start < c < line_end),
                       *([speaker.end(1)] if speaker else [])})
        pieces = []
        for a, b in pairwise(cuts):
            piece = html.escape(line[a:b])
            if speaker and b <= speaker.end(1):
                piece = f"<b>{piece}</b>"
            if start <= line_start + a and line_start + b <= end:
                piece = f"<mark>{piece}</mark>"
            pieces.append(piece)
        paragraphs.append(f"<p>{''.join(pieces)}</p>")
    return "".join(paragraphs)


@st.dialog("Read in context", width="large")
def show_section(number: int, source: dict, section_text: SectionText):
    """The whole scene (or sonnet, synopsis, biography section) with the cited passage highlighted."""
    text, start, end = section_text(source["index"])
    st.markdown(f"**[{number}] {source['heading']}** · page {source['page']}")
    where = "highlighted below" if start > 0 else "highlighted"
    st.caption(f"The whole section. The passage the answer cites is {where}.")
    st.markdown(f'<div class="section-text">{section_html(text, start, end)}</div>', unsafe_allow_html=True)


def show_sources(sources: list[dict], answer: str, key: str, section_text: SectionText):
    """List the passages the answer cites, keeping the [n] numbers used in the answer.

    key identifies the answer, so each passage's "Read more" button has a unique key.
    """
    if not sources:
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    shown = [(i, src) for i, src in enumerate(sources, 1) if i in cited]
    if shown:
        label = f"Sources ({len(shown)} cited)"
    else:  # e.g. the passages didn't answer the question: show what was searched
        shown, label = list(enumerate(sources, 1)), f"Passages searched ({len(sources)})"
    with st.expander(label):
        for i, src in shown:
            st.markdown(f"**[{i}] {src['heading']}** · page {src['page']}")
            more = "index" in src and len(section_text(src["index"])[0]) > len(src["excerpt"])
            # Excerpt and "…Read more" share one line of text (laid out inline by styles.css)
            with st.container(key=f"excerpt_{key}_{i}"):
                st.caption(f"{src['excerpt']}…" if more else src["excerpt"])
                if more and st.button("Read more", key=f"read_more_{key}_{i}", type="tertiary"):
                    show_section(i, src, section_text)
        if len(shown) < len(sources):
            st.caption(f"{len(sources) - len(shown)} other passages were searched but not cited.")
