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


def ask(question: str):
    st.session_state.pending = question


with st.sidebar:
    st.header("The collection")
    st.markdown(
        f"This app searches **38 plays** and **4 books of poems** by William Shakespeare, "
        f"from the Folger Shakespeare Library editions, split into "
        f"**{rag.vectorstore.index.ntotal:,} passages** for searching."
    )
    for genre, titles in WORKS.items():
        with st.expander(f"{genre} ({len(titles)})"):
            st.markdown("\n".join(f"- {title}" for title in titles))

    st.header("What you can ask")
    st.markdown(
        "Ask about **plots, characters, scenes and speeches**. Each answer is summarized "
        "from the passages that best match your question."
    )
    st.caption(
        "Each question stands alone, so follow-ups don't remember earlier ones. "
        "Questions that need the whole canon at once, like counting every death, work poorly."
    )
    for example in EXAMPLES:
        st.button(example, on_click=ask, args=(example,))

st.markdown(HEADER, unsafe_allow_html=True)

query = st.chat_input("Ask about a play, character or scene") or st.session_state.pop("pending", None)

if not st.session_state.messages and not query:
    st.markdown(EPIGRAPH, unsafe_allow_html=True)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar=message["avatar"]):
        with st.container(key=f"{'answer' if message['role'] == 'assistant' else 'question'}_{i}"):
            st.markdown(message["content"])

if query:
    st.session_state.messages.append({"role": "user", "avatar": "🪶", "content": query})
    with st.chat_message("user", avatar="🪶"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🎭"):
        with st.spinner("Searching the plays..."):
            answer = rag.search_and_summarize(query, top_k=3)
        with st.container(key=f"answer_{len(st.session_state.messages)}"):
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "avatar": "🎭", "content": answer})

if st.session_state.messages:
    with st.sidebar:
        st.divider()
        st.button("Clear chat", on_click=st.session_state.messages.clear)
