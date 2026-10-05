import os
import threading
from datetime import date

import streamlit as st

from src.search import RAGSearch

WORKS = {
    "Comedies": [
        "All's Well That Ends Well", "As You Like It", "The Comedy of Errors", "Love's Labor's Lost",
        "Measure for Measure", "The Merchant of Venice", "The Merry Wives of Windsor",
        "A Midsummer Night's Dream", "Much Ado About Nothing", "The Taming of the Shrew",
        "Twelfth Night", "The Two Gentlemen of Verona",
    ],
    "Tragedies": [
        "Antony and Cleopatra", "Coriolanus", "Hamlet", "Julius Caesar", "King Lear", "Macbeth",
        "Othello", "Romeo and Juliet", "Timon of Athens", "Titus Andronicus", "Troilus and Cressida",
    ],
    "Histories": [
        "Henry IV, Part 1", "Henry IV, Part 2", "Henry V", "Henry VI, Part 1", "Henry VI, Part 2",
        "Henry VI, Part 3", "Henry VIII", "King John", "Richard II", "Richard III",
    ],
    "Romances": ["Cymbeline", "Pericles", "The Tempest", "The Winter's Tale", "The Two Noble Kinsmen"],
    "Poems": ["Shakespeare's Sonnets", "Venus and Adonis", "Lucrece", "The Phoenix and Turtle"],
}

EXAMPLES = [
    "What happens in As You Like It?",
    "Why does Macbeth murder King Duncan?",
    "What bond does Shylock demand from Antonio?",
    "How does Puck cause trouble in the forest?",
    "Who is Caliban in The Tempest?",
    "How does Romeo and Juliet end?",
    "Who was Shakespeare's wife?",
]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English+SC&family=IM+Fell+English:ital@0;1&family=UnifrakturMaguntia&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stChatInput"] textarea {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.08rem;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at center, #faf2dc 0%, #f1e2bd 65%, #dcc28c 100%);
}
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ecdcb2 0%, #e2cc98 100%);
    border-right: 3px double #a8814a;
}
h1, h2, h3, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'IM Fell English SC', Georgia, serif !important;
    font-weight: 400 !important;
    color: #7a1f1f;
}
[data-testid="stSidebar"] h2::before {
    content: "❧ ";
}

.bard-header {
    position: relative;
    text-align: center;
    padding: 1.4rem 1rem 1.1rem;
    margin: 0.5rem 0 1.4rem;
    border: 4px double #7a1f1f;
    outline: 1px solid #c9a96e;
    outline-offset: 5px;
    background: rgba(251, 244, 226, 0.7);
}
.bard-header .corner {
    position: absolute;
    color: #7a1f1f;
    font-size: 1.3rem;
    line-height: 1;
}
.bard-header .tl { top: 0.3rem; left: 0.45rem; }
.bard-header .tr { top: 0.3rem; right: 0.45rem; }
.bard-header .bl { bottom: 0.3rem; left: 0.45rem; }
.bard-header .br { bottom: 0.3rem; right: 0.45rem; }
.bard-title {
    font-family: 'IM Fell English SC', Georgia, serif;
    font-size: clamp(2.6rem, 8.5vw, 4.6rem) !important;
    line-height: 1.05;
    color: #7a1f1f;
    text-shadow: 1px 1px 0 #e6d3a3;
}
.bard-subtitle {
    font-family: 'IM Fell English', Georgia, serif;
    font-style: italic;
    font-size: 1.35rem !important;
    margin-top: 0.3rem;
}
.bard-rule {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    color: #7a1f1f;
    font-size: 1.4rem;
    margin: 0.6rem auto 0;
    max-width: 22rem;
}
.bard-rule::before, .bard-rule::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7a1f1f, transparent);
}
.bard-epigraph {
    text-align: center;
    font-family: 'IM Fell English', Georgia, serif;
    font-style: italic;
    font-size: 1.2rem;
    margin: 0 auto 0.3rem;
    max-width: 32rem;
}
.bard-epigraph-source {
    text-align: center;
    font-variant: small-caps;
    letter-spacing: 0.05em;
    color: #6b4a2b;
}
.bard-hint {
    text-align: center;
    margin-top: 1.2rem;
    color: #6b4a2b;
}

[data-testid="stChatMessage"] {
    background: #fbf4e2;
    border: 3px double #c9a96e;
    border-radius: 2px;
    box-shadow: 2px 3px 8px rgba(90, 58, 34, 0.15);
}
[class*="st-key-answer"] [data-testid="stMarkdownContainer"] > p:first-child::first-letter {
    float: left;
    font-family: 'UnifrakturMaguntia', 'IM Fell English', serif;
    font-size: 3.3em;
    line-height: 0.85;
    padding: 0.05em 0.12em 0 0;
    color: #7a1f1f;
}
.stButton > button {
    font-family: 'IM Fell English', Georgia, serif;
    text-align: left;
    width: 100%;
    background: #fbf4e2;
    border: 1px solid #c9a96e;
}
.usage-note {
    text-align: center;
    font-family: 'IM Fell English', Georgia, serif;
    font-style: italic;
    color: #6b4a2b;
    margin-top: 0.4rem;
}
.stButton > button:hover {
    border-color: #7a1f1f;
    color: #7a1f1f;
}
</style>
"""

HEADER = """
<div class="bard-header">
  <span class="corner tl">✥</span><span class="corner tr">✥</span>
  <span class="corner bl">✥</span><span class="corner br">✥</span>
  <div class="bard-title">Shakespeare RAG</div>
  <div class="bard-subtitle">Ask anything about Shakespeare's plays and poems</div>
  <div class="bard-rule">❦</div>
</div>
"""

EPIGRAPH = """
<div class="bard-epigraph">“All the world’s a stage, and all the men and women merely players.”</div>
<div class="bard-epigraph-source">As You Like It, Act 2, Scene 7</div>
<div class="bard-hint">Ask a question below, or pick an example from the sidebar.</div>
"""

st.set_page_config(page_title="Shakespeare RAG", page_icon="🪶", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)
st.logo("assets/logo.svg", size="large")


@st.cache_resource(show_spinner="Loading the plays...")
def load_rag() -> RAGSearch:
    return RAGSearch()


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.questions_asked = 0

# Every question costs OpenAI credits, so cap usage of the public demo (override via env vars / Streamlit secrets).
# Set IS_LOCAL=true locally to skip them.
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION") or 20)
MAX_QUESTIONS_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY") or 300)


@st.cache_resource
def daily_usage() -> dict:
    """Questions asked today across all visitors (resets daily and when the app restarts)."""
    return {"day": date.today(), "count": 0, "lock": threading.Lock()}


def running_locally() -> bool:
    """IS_LOCAL=true (e.g. in your local .env) turns usage limits off; unset or anything else keeps them on."""
    return os.getenv("IS_LOCAL", "false").strip().lower() in {"true", "1", "yes"}


def limit_reached() -> str | None:
    if running_locally():
        return None
    usage = daily_usage()
    with usage["lock"]:
        if usage["day"] != date.today():
            usage["day"], usage["count"] = date.today(), 0
        if usage["count"] >= MAX_QUESTIONS_PER_DAY:
            return "The demo has reached its question limit for today. Please come back tomorrow."
    if st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
        return f"You've asked {MAX_QUESTIONS_PER_SESSION} questions, the limit for one visit to this demo. Thanks for trying it!"
    return None


def usage_note() -> str:
    """Questions this visitor has left, shown above the input."""
    if running_locally():
        return "Running locally: no question limit."
    session_left = max(MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked, 0)
    day_left = max(MAX_QUESTIONS_PER_DAY - daily_usage()["count"], 0)
    if day_left < session_left:
        return f"{day_left} question{'s' if day_left != 1 else ''} left today for all visitors to this demo."
    return f"{session_left} of {MAX_QUESTIONS_PER_SESSION} questions left in this visit."


def count_question():
    if running_locally():
        return
    st.session_state.questions_asked += 1
    usage = daily_usage()
    with usage["lock"]:
        usage["count"] += 1


def ask(question: str):
    st.session_state.pending = question


def show_sources(sources: list[dict]):
    """List the passages an answer was drawn from, numbered to match its [n] citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} passages)"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**[{i}] {src['heading']}** · page {src['page']}")
            st.caption(src["excerpt"])


with st.sidebar:
    st.header("The collection")
    st.markdown(
        f"This app searches **38 plays** and **4 books of poems** by William Shakespeare, "
        f"from the Folger Shakespeare Library editions, plus a biography of Shakespeare's life "
        f"and career (from Wikipedia), split into "
        f"**{rag.vectorstore.index.ntotal:,} passages** for searching."
    )
    for genre, titles in WORKS.items():
        with st.expander(f"{genre} ({len(titles)})"):
            st.markdown("\n".join(f"- {title}" for title in titles))

    st.header("What you can ask")
    st.markdown(
        "Ask about **plots, characters, scenes and speeches**. Each answer comes only from "
        "the passages that best match your question, with numbered sources you can check."
    )
    st.caption(
        "Follow-up questions like \"who is her cousin?\" use the conversation so far. "
        "Questions that need the whole canon at once, like counting every death, work poorly."
    )
    for example in EXAMPLES:
        st.button(example, on_click=ask, args=(example,))

st.markdown(HEADER, unsafe_allow_html=True)

limit = limit_reached()
query = st.chat_input("Ask about a play, character or scene", disabled=bool(limit)) or st.session_state.pop("pending", None)
if limit and query:
    query = None  # an example button was clicked after the limit

if not st.session_state.messages and not query:
    st.markdown(EPIGRAPH, unsafe_allow_html=True)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar=message["avatar"]):
        with st.container(key=f"{'answer' if message['role'] == 'assistant' else 'question'}_{i}"):
            st.markdown(message["content"])
        if message.get("searched"):
            st.caption(f"Searched for: {message['searched']}")
        show_sources(message.get("sources", []))

if query:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "avatar": "🪶", "content": query})
    with st.chat_message("user", avatar="🪶"):
        st.markdown(query)

    count_question()
    with st.chat_message("assistant", avatar="🎭"):
        with st.spinner("Searching the plays..."):
            question, results = rag.retrieve(query, history)
        with st.container(key=f"answer_{len(st.session_state.messages)}"):
            answer = st.write_stream(rag.stream(question, results))
        searched = question if question != query else None  # only show when a follow-up was rewritten
        sources = [
            {
                "heading": r["metadata"].get("heading", r["metadata"].get("source", "")),
                "page": r["metadata"].get("page", "?"),
                # Drop the heading line every chunk starts with; it's already shown above the excerpt
                "excerpt": r["metadata"]["text"].split("\n", 1)[-1][:300].strip() + "…",
            }
            for r in results
        ]
        if searched:
            st.caption(f"Searched for: {searched}")
        show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "avatar": "🎭", "content": answer, "sources": sources, "searched": searched}
    )
    if limit_reached():
        st.rerun()  # this question used up the limit: redraw now with the input disabled and the notice shown

if limit:
    st.info(limit, icon="📜")
else:
    st.markdown(f'<div class="usage-note">❦ {usage_note()}</div>', unsafe_allow_html=True)

if st.session_state.messages:
    with st.sidebar:
        st.divider()
        st.button("Clear chat", on_click=st.session_state.messages.clear)
