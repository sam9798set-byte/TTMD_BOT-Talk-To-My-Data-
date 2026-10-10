"""
TTMD Bot: AI Business Data Copilot (Streamlit interface)

Run from the project folder:
    python -m streamlit run app.py

All Text-to-SQL logic lives in ttmd_bot.py. This file only draws the interface.
"""

try:
    import torch  # noqa: F401  (loaded first; avoids a Windows DLL problem seen with PyTorch)
except Exception:
    pass

import pandas as pd
import streamlit as st

# Must be the first Streamlit call.
st.set_page_config(page_title="TTMD Bot", page_icon="🤖", layout="wide")

try:
    from ttmd_bot import MAX_ROWS, TTMDError, answer_question, initialize_backend, redact
except ImportError as import_error:
    st.title("TTMD Bot")
    st.error(
        f"A required Python package is missing: **{import_error.name or 'unknown'}**. "
        "Install the dependencies and restart the app."
    )
    st.code(
        "python -m pip install streamlit groq sqlglot sqlalchemy pymysql cryptography "
        "langchain-core langchain-community langchain-chroma langchain-huggingface "
        "chromadb sentence-transformers",
        language="powershell",
    )
    st.stop()

EXAMPLE_QUESTIONS = [
    "How many devices are stale for more than 30 days?",
    "Which 5 device types have the most stale devices?",
    "What percentage of devices were active in the last 24 hours?",
    "Compare active and stale devices for each device type.",
]

ERROR_HINTS = {
    "Groq error": "The AI service had a problem. Wait a moment and try again.",
    "SQL generation error": "The AI did not produce a usable query. Try rephrasing your question.",
    "SQL blocked by validation": "For safety, only read-only SELECT queries are allowed. Try rephrasing your question.",
    "SQL execution error": "The database could not run the generated query. Try rephrasing, or check the SQL below.",
    "Unexpected error": "Something unexpected happened. Try again or rephrase your question.",
}


# ---------------------------------------------------------------------
# Backend (loaded once and shared; conversation history is NOT cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_backend():
    """MySQL engine, schema, embeddings, ChromaDB, example selector and Groq client."""
    return initialize_backend()


# ---------------------------------------------------------------------
# Session state (separate for every browser session)
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------
def unique_columns(columns) -> list:
    """Streamlit cannot show a table with duplicate column names."""
    seen, result = {}, []
    for col in columns:
        name = str(col)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        result.append(name)
    return result


def render_table(result) -> None:
    count = len(result.rows)
    if result.truncated:
        st.warning(
            f"Showing the first {count} rows. More rows matched, "
            f"but results are limited to {MAX_ROWS}."
        )
    else:
        st.caption(f"{count} row{'s' if count != 1 else ''} returned")

    df = pd.DataFrame(result.rows, columns=unique_columns(result.columns))
    try:
        st.dataframe(df, hide_index=True)
    except Exception:
        st.dataframe(df.astype(str), hide_index=True)


def render_result(result) -> None:
    if result.status == "error":
        st.error(f"**{result.error_type}**\n\n{result.error_message}")
        hint = ERROR_HINTS.get(result.error_type)
        if hint:
            st.caption(hint)
    elif result.status == "unanswerable":
        st.warning(result.answer)
    else:
        st.markdown(result.answer)

    if result.executed:
        st.caption(f"Query executed (attempt {result.attempts})")
    elif result.sql:
        st.caption("This query was NOT executed.")

    if result.sql:
        label = "Generated SQL" if result.executed else "Generated SQL (not executed)"
        with st.expander(label):
            st.code(result.sql, language="sql")

    if result.executed and result.rows:
        render_table(result)
    elif result.executed and result.status == "success":
        st.info("The query ran successfully but returned no rows.")

    if result.examples:
        with st.expander("Similar examples used for this question"):
            for question in result.examples:
                st.write(f"- {question}")


def render_example_buttons(prefix: str) -> None:
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        st.button(question, key=f"{prefix}_{i}", on_click=ask_example, args=(question,))


# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------
st.title("TTMD Bot")
st.caption("AI Business Data Copilot")

# ---------------------------------------------------------------------
# Load backend (cached), with friendly errors
# ---------------------------------------------------------------------
try:
    with st.spinner("Starting TTMD Bot... the first start loads the embedding model and can take a minute."):
        backend = load_backend()
except TTMDError as startup_error:
    st.error(f"**{startup_error.label}**\n\n{startup_error}")
    st.button("Retry")
    st.stop()
except Exception as startup_error:
    st.error(f"**Could not start TTMD Bot**\n\n{redact(type(startup_error).__name__ + ': ' + str(startup_error))[:300]}")
    st.button("Retry")
    st.stop()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Try an example")
    render_example_buttons("side")
    st.divider()
    st.button("Clear conversation", on_click=clear_conversation)
    st.divider()
    st.caption(f"Few-shot examples available: {backend.example_count}")
    st.caption("Each question is answered on its own. Follow-up questions do not remember earlier ones.")
    for warning in backend.startup_warnings:
        st.warning(warning)

# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------
pending = st.session_state.pending_question
st.session_state.pending_question = None
typed = st.chat_input("Ask a question about your data...")
question = pending or typed

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_result(message["result"])

if not st.session_state.messages and not question:
    st.info("Ask a question in plain English, or pick an example below.")
    render_example_buttons("main")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... writing SQL and querying the database"):
            query_result = answer_question(backend, question)
        render_result(query_result)

    st.session_state.messages.append({"role": "assistant", "result": query_result})