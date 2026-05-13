"""Shared helpers for data loading, evaluation, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATASET_CANDIDATES = [
    DATA_RAW_DIR / "dataset-tickets-german_normalized_50_5_2.csv",
    DATA_RAW_DIR / "dataset-tickets-german_normalized.csv",
    DATA_RAW_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    DATA_RAW_DIR / "dataset-tickets-multi-lang-4-20k.csv",
    DATA_RAW_DIR / "dataset-tickets-multi-lang3-4k.csv",
    DATA_RAW_DIR / "customer_support_tickets.csv",
]
RAW_DATA_FILE = DATASET_CANDIDATES[0]
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_tickets.csv"

TEXT_COLUMN_OPTIONS = [
    ("subject", "body"),
    ("Ticket Subject", "Ticket Description"),
]
CATEGORY_COLUMN_OPTIONS = ["queue", "Ticket Type", "type"]
PRIORITY_COLUMN_OPTIONS = ["priority", "Ticket Priority"]


def ensure_project_dirs() -> None:
    """Create directories used by training and evaluation scripts."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the Kaggle customer support ticket dataset."""
    path = Path(path)
    if not path.exists():
        for candidate in DATASET_CANDIDATES:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "No supported dataset found in data/raw/. Download a ticket dataset CSV and place it there."
            )
    return pd.read_csv(path)


def prepare_ticket_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modeling columns used by the project."""
    text_columns = next((cols for cols in TEXT_COLUMN_OPTIONS if all(col in df.columns for col in cols)), None)
    category_column = next((col for col in CATEGORY_COLUMN_OPTIONS if col in df.columns), None)
    priority_column = next((col for col in PRIORITY_COLUMN_OPTIONS if col in df.columns), None)

    if text_columns is None or category_column is None or priority_column is None:
        raise ValueError(
            "Dataset must include text columns plus category and priority labels. "
            f"Available columns: {list(df.columns)}"
        )

    prepared = df.copy()
    prepared["ticket_text"] = (
        prepared[text_columns[0]].fillna("").astype(str)
        + " "
        + prepared[text_columns[1]].fillna("").astype(str)
    )
    prepared["category"] = prepared[category_column].fillna("Unknown").astype(str)
    prepared["priority"] = prepared[priority_column].fillna("Unknown").astype(str)
    prepared["ticket_length"] = prepared["ticket_text"].str.split().str.len()
    return prepared


def save_joblib(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def evaluate_predictions(y_true, y_pred, labels: Iterable[str] | None = None) -> dict:
    """Return common classification metrics as a dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics(metrics: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def plot_target_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path | str) -> None:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y=column, order=df[column].value_counts().index, color="#4c78a8")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ticket_length_histogram(df: pd.DataFrame, output_path: Path | str) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df["ticket_length"], bins=30, color="#59a14f")
    plt.title("Ticket Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Tickets")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path | str) -> None:
    labels = sorted(pd.Series(y_true).unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
