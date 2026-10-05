"""The cited-sources panel under each answer."""
import re

import streamlit as st


def to_sources(results: list[dict]) -> list[dict]:
    """What the panel shows for each retrieved passage (kept in the chat history)."""
    return [
        {
            "heading": r["metadata"].get("heading", r["metadata"].get("source", "")),
            "page": r["metadata"].get("page", "?"),
            # Drop the heading line every chunk starts with; it's already shown above the excerpt
            "excerpt": r["metadata"]["text"].split("\n", 1)[-1][:300].strip() + "…",
        }
        for r in results
    ]


def show_sources(sources: list[dict], answer: str):
    """List the passages the answer cites, keeping the [n] numbers used in the answer."""
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
            st.caption(src["excerpt"])
        if len(shown) < len(sources):
            st.caption(f"{len(sources) - len(shown)} other passages were searched but not cited.")
