"""Streamlit demo for support ticket category and priority prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import predict_ticket  # noqa: E402


st.set_page_config(page_title="Support Ticket Classifier", layout="centered")

st.title("Support Ticket Classification")
st.write("Classify incoming customer support tickets and apply business escalation rules.")

ticket_text = st.text_area(
    "Ticket text",
    height=180,
    placeholder="Example: Payment deducted twice and refund not received.",
)

if st.button("Predict", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a support ticket.")
    else:
        try:
            result = predict_ticket(ticket_text)
            col1, col2, col3 = st.columns(3)
            col1.metric("Category", result["category"])
            col2.metric("ML Priority", result["priority"])
            col3.metric("Final Priority", result["boosted_priority"])

            st.subheader("Escalation Keywords")
            if result["detected_keywords"]:
                st.write(", ".join(result["detected_keywords"]))
            else:
                st.write("No escalation keywords detected.")

            st.subheader("Top TF-IDF Terms")
            st.write(", ".join(result["top_terms"]) if result["top_terms"] else "No strong terms found.")

            with st.expander("Confidence scores"):
                st.write(
                    {
                        "category_confidence": result["category_confidence"],
                        "priority_confidence": result["priority_confidence"],
                    }
                )
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Run `python src/train.py` first to create model artifacts.")
