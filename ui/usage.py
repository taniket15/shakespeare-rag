"""Usage limits for the public demo: questions per visit and per day, plus the counter shown to users."""
import os
import threading
from datetime import date

import streamlit as st

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
