from __future__ import annotations

import os
from typing import Dict, List

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HR Resource Query Chatbot", page_icon="🤖", layout="wide")

st.title("HR Resource Query Chatbot")

with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", BACKEND_URL)
    st.caption("Expected to point to FastAPI service, e.g., http://127.0.0.1:8000")

    st.divider()
    st.subheader("Quick Search")
    query = st.text_input("Query", placeholder="Find Python developers with 3+ years experience")
    min_exp = st.number_input("Min years", min_value=0, max_value=30, value=0, step=1)
    skills = st.text_input("Skills (comma-separated)")
    available_only = st.checkbox("Available only", value=False)
    if st.button("Run Search"):
        params = {
            "query": query or None,
            "min_experience": int(min_exp) if min_exp else None,
            "skills": skills or None,
            "available_only": available_only,
        }
        params = {k: v for k, v in params.items() if v is not None}
        try:
            resp = requests.get(f"{backend_url}/employees/search", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Found {data.get('total', 0)} candidates")
            for c in data.get("candidates", []):
                emp = c["employee"]
                with st.container(border=True):
                    st.markdown(
                        f"**{emp['name']}** — {emp['experience_years']} yrs | skills: {', '.join(emp['skills'])} | availability: {emp['availability']}"
                    )
                    st.caption(f"Projects: {', '.join(emp['projects'])}")
                    if c.get("reasons"):
                        st.write(
                            "Reasons: " + ", ".join(c.get("reasons", []))
                        )
        except Exception as e:
            st.error(str(e))

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask for people, e.g., 'Who has worked on healthcare projects?' ")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        resp = requests.post(f"{backend_url}/chat", json={"message": prompt}, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "")
        candidates = data.get("candidates", [])

        with st.chat_message("assistant"):
            st.markdown(reply)
            with st.expander("View top candidates", expanded=True):
                for c in candidates:
                    emp = c["employee"]
                    with st.container(border=True):
                        st.markdown(
                            f"**{emp['name']}** — {emp['experience_years']} yrs | skills: {', '.join(emp['skills'])} | availability: {emp['availability']}"
                        )
                        st.caption(f"Projects: {', '.join(emp['projects'])}")
                        if c.get("reasons"):
                            st.write("Reasons: " + ", ".join(c.get("reasons", [])))

        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        error_text = f"Error contacting backend: {e}"
        with st.chat_message("assistant"):
            st.error(error_text)
        st.session_state.messages.append({"role": "assistant", "content": error_text}) 