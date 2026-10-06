"""Streamlit chat UI for Shakespeare RAG. Run with: streamlit run chat.py"""
import uuid

import langsmith
import streamlit as st

from src.search import RAGSearch
from ui.content import EPIGRAPH, EXAMPLES, HEADER, WORKS, styles
from ui.sources import show_sources, to_sources
from ui.usage import count_question, limit_reached, usage_note

st.set_page_config(page_title="Shakespeare RAG", page_icon="🪶", layout="centered")
st.markdown(styles(), unsafe_allow_html=True)
st.logo("assets/logo.svg", size="large")


@st.cache_resource(show_spinner="Loading the plays...")
def load_rag() -> RAGSearch:
    return RAGSearch()


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.questions_asked = 0
    st.session_state.thread_id = str(uuid.uuid4())  # groups a conversation's traces into one LangSmith thread


def ask(question: str):
    st.session_state.pending = question


def new_conversation():
    st.session_state.messages.clear()
    st.session_state.thread_id = str(uuid.uuid4())


with st.sidebar:
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
        show_sources(message.get("sources", []), message["content"])

if query:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "avatar": "🪶", "content": query})
    with st.chat_message("user", avatar="🪶"):
        with st.container(key=f"question_{len(st.session_state.messages) - 1}"):
            st.markdown(query)

    count_question()
    # One trace per question, with retrieval and the streamed answer as its child runs
    with st.chat_message("assistant", avatar="🎭"), langsmith.trace(
        "chat_turn", inputs={"question": query}, metadata={"thread_id": st.session_state.thread_id},
    ) as trace:
        with st.spinner("Searching the plays..."):
            question, results = rag.retrieve(query, history)
        with st.container(key=f"answer_{len(st.session_state.messages)}"):
            answer = st.write_stream(rag.stream(question, results))
        trace.end(outputs={"answer": answer, "question": question})
        searched = question if question != query else None  # only show when a follow-up was rewritten
        sources = to_sources(results)
        if searched:
            st.caption(f"Searched for: {searched}")
        show_sources(sources, answer)
    st.session_state.messages.append(
        {"role": "assistant", "avatar": "🎭", "content": answer, "sources": sources, "searched": searched}
    )
    if limit_reached():
        st.rerun()  # this question used up the limit: redraw now with the input disabled and the notice shown

if limit:
    st.info(limit, icon="📜")

# Footer row above the input: questions left, and a way to start over once there's a conversation
with st.container(key="chat_footer", horizontal=True, horizontal_alignment="center", vertical_alignment="center"):
    if not limit:
        st.markdown(f'<div class="usage-note">❦ {usage_note()}</div>', unsafe_allow_html=True)
    if st.session_state.messages:
        st.button("↺ New conversation", key="new_conversation", type="tertiary",
                  on_click=new_conversation, help="Clear this conversation and start fresh")
