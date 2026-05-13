"""Train category and priority classifiers for support tickets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import add_cleaned_ticket_column, clean_text
from utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_FILE,
    ensure_project_dirs,
    evaluate_predictions,
    load_raw_data,
    plot_confusion_matrix,
    plot_target_distribution,
    plot_ticket_length_histogram,
    prepare_ticket_dataframe,
    save_joblib,
    save_metrics,
)


RANDOM_STATE = 42
MAX_FEATURES = 120000
MIN_DF = 2
TEST_SIZE = 0.2

CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.pkl"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


def build_vectorizer() -> TfidfVectorizer:
    """Create the shared TF-IDF vectorizer used by both models."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 3),
        min_df=MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def get_candidate_models(include_naive_bayes: bool = True) -> dict:
    """Return candidate estimators for comparison."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }
    if include_naive_bayes:
        models["Multinomial Naive Bayes"] = MultinomialNB()
    return models


def train_and_compare_models(X_train, X_test, y_train, y_test, models: dict, task_name: str) -> tuple[str, object, dict]:
    """Train candidate models and return the best model by macro F1-score."""
    results = {}
    best_name = ""
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        results[name] = metrics

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name
            best_model = model

    save_metrics(results, OUTPUTS_DIR / f"{task_name}_model_comparison.json")
    return best_name, best_model, results


def train_models() -> dict:
    """Run the complete training pipeline and save all artifacts."""
    ensure_project_dirs()
    raw_df = load_raw_data()
    df = prepare_ticket_dataframe(raw_df)
    df = add_cleaned_ticket_column(df)
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    plot_target_distribution(df, "category", "Ticket Category Distribution", OUTPUTS_DIR / "category_distribution.png")
    plot_target_distribution(df, "priority", "Ticket Priority Distribution", OUTPUTS_DIR / "priority_distribution.png")
    plot_ticket_length_histogram(df, OUTPUTS_DIR / "ticket_length_distribution.png")

    X_train_text, X_test_text, y_category_train, y_category_test = train_test_split(
        df["cleaned_ticket"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    vectorizer = build_vectorizer()
    X_category_train = vectorizer.fit_transform(X_train_text)
    X_category_test = vectorizer.transform(X_test_text)

    category_name, category_model, category_results = train_and_compare_models(
        X_category_train,
        X_category_test,
        y_category_train,
        y_category_test,
        get_candidate_models(include_naive_bayes=True),
        "category",
    )
    category_predictions = category_model.predict(X_category_test)
    plot_confusion_matrix(
        y_category_test,
        category_predictions,
        f"Category Confusion Matrix - {category_name}",
        OUTPUTS_DIR / "category_confusion_matrix.png",
    )

    X_priority_train_text, X_priority_test_text, y_priority_train, y_priority_test = train_test_split(
        df["cleaned_ticket"],
        df["priority"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["priority"],
    )
    X_priority_train = vectorizer.transform(X_priority_train_text)
    X_priority_test = vectorizer.transform(X_priority_test_text)

    priority_name, priority_model, priority_results = train_and_compare_models(
        X_priority_train,
        X_priority_test,
        y_priority_train,
        y_priority_test,
        get_candidate_models(include_naive_bayes=False),
        "priority",
    )
    priority_predictions = priority_model.predict(X_priority_test)
    plot_confusion_matrix(
        y_priority_test,
        priority_predictions,
        f"Priority Confusion Matrix - {priority_name}",
        OUTPUTS_DIR / "priority_confusion_matrix.png",
    )

    save_joblib(category_model, CATEGORY_MODEL_PATH)
    save_joblib(priority_model, PRIORITY_MODEL_PATH)
    save_joblib(vectorizer, VECTORIZER_PATH)

    summary = {
        "category_best_model": category_name,
        "priority_best_model": priority_name,
        "category_macro_f1": category_results[category_name]["f1_macro"],
        "priority_macro_f1": priority_results[priority_name]["f1_macro"],
        "processed_rows": int(len(df)),
    }
    save_metrics(summary, OUTPUTS_DIR / "training_summary.json")
    return summary


def train_pipeline_model(target: str) -> Pipeline:
    """Convenience helper for experiments that need a single sklearn pipeline."""
    df = add_cleaned_ticket_column(prepare_ticket_dataframe(load_raw_data()))
    if target not in {"category", "priority"}:
        raise ValueError("target must be 'category' or 'priority'")

    model = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = Pipeline(
        steps=[
            (
                "clean_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=MAX_FEATURES,
                    ngram_range=(1, 3),
                    min_df=MIN_DF,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("model", model),
        ]
    )
    pipeline.fit(df["ticket_text"], df[target])
    return pipeline


if __name__ == "__main__":
    print(pd.Series(train_models()).to_string())
