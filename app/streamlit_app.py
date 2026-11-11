"""Day 11: chat UI. Talks to the FastAPI backend.

Run:  streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXAMPLES = [
    "Is flood damage covered?",
    "What is the deductible?",
    "What is excluded from coverage?",
]

st.set_page_config(page_title="PolicyLens", page_icon="📄")
st.title("PolicyLens")
st.caption("Answers come only from the policy text, with page citations.")


@st.cache_data(ttl=60)
def get_policies():
    try:
        return requests.get(f"{API_URL}/policies", timeout=10).json()["policies"]
    except Exception:
        return []


policies = get_policies()
policy = st.selectbox("Policy", ["All policies"] + policies)

cols = st.columns(len(EXAMPLES))
clicked = None
for col, example in zip(cols, EXAMPLES):
    if col.button(example):
        clicked = example

question = st.text_input("Ask a question", value=clicked or "")

if question:
    with st.spinner("Searching the policy..."):
        payload = {"question": question, "policy": None if policy == "All policies" else policy}
        try:
            resp = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

    st.markdown("### Answer")
    st.write(data["answer"])
    st.caption(f"Answered in {data['latency_ms']} ms")

    if data["sources"]:
        st.markdown("### Sources")
        for s in data["sources"]:
            with st.expander(f"{s['policy']} — page {s['page']}"):
                st.write(s["snippet"] + "…")
