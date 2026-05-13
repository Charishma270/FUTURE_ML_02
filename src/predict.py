"""Reusable prediction pipeline for new support tickets."""

from __future__ import annotations

from typing import Any

import numpy as np

from preprocessing import clean_text
from utils import MODELS_DIR, load_joblib


CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.pkl"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

ESCALATION_KEYWORDS = [
    "refund",
    "hacked",
    "urgent",
    "payment failed",
    "server down",
    "cannot access",
    "breach",
    "security",
    "fraud",
    "double charged",
    "charged twice",
]

PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def detect_escalation_keywords(ticket_text: str) -> list[str]:
    """Find business-critical escalation keywords in raw ticket text."""
    normalized = ticket_text.lower()
    return [keyword for keyword in ESCALATION_KEYWORDS if keyword in normalized]


def boost_priority(predicted_priority: str, detected_keywords: list[str]) -> str:
    """Apply hybrid business logic to increase priority when critical keywords appear."""
    if not detected_keywords:
        return predicted_priority

    if predicted_priority not in PRIORITY_ORDER:
        return "High"

    current_index = PRIORITY_ORDER.index(predicted_priority)
    boosted_index = max(current_index, PRIORITY_ORDER.index("High"))
    return PRIORITY_ORDER[boosted_index]


def _confidence(model: Any, vectorized_text) -> float | None:
    """Return a best-effort confidence score when the model supports it."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_text)[0]
        return float(np.max(probabilities))

    if hasattr(model, "decision_function"):
        scores = model.decision_function(vectorized_text)
        scores = np.ravel(scores)
        if scores.size == 1:
            return float(1 / (1 + np.exp(-scores[0])))
        exp_scores = np.exp(scores - np.max(scores))
        return float(np.max(exp_scores / exp_scores.sum()))

    return None


def top_tfidf_terms(ticket_text: str, vectorizer, top_n: int = 8) -> list[str]:
    """Show the strongest TF-IDF terms present in a ticket."""
    cleaned = clean_text(ticket_text)
    vectorized = vectorizer.transform([cleaned])
    if vectorized.nnz == 0:
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    row = vectorized.toarray()[0]
    top_indices = row.argsort()[::-1][:top_n]
    return [feature_names[index] for index in top_indices if row[index] > 0]


def predict_ticket(ticket_text: str) -> dict:
    """Predict category and priority for one raw support ticket."""
    category_model = load_joblib(CATEGORY_MODEL_PATH)
    priority_model = load_joblib(PRIORITY_MODEL_PATH)
    vectorizer = load_joblib(VECTORIZER_PATH)

    cleaned = clean_text(ticket_text)
    vectorized = vectorizer.transform([cleaned])

    category = category_model.predict(vectorized)[0]
    priority = priority_model.predict(vectorized)[0]
    detected_keywords = detect_escalation_keywords(ticket_text)
    boosted_priority = boost_priority(priority, detected_keywords)

    return {
        "category": category,
        "priority": priority,
        "boosted_priority": boosted_priority,
        "detected_keywords": detected_keywords,
        "top_terms": top_tfidf_terms(ticket_text, vectorizer),
        "category_confidence": _confidence(category_model, vectorized),
        "priority_confidence": _confidence(priority_model, vectorized),
    }


if __name__ == "__main__":
    sample = "Payment deducted twice and refund not received"
    print(predict_ticket(sample))
