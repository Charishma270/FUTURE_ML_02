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

RAW_DATA_FILE = DATA_RAW_DIR / "customer_support_tickets.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_tickets.csv"

TEXT_COLUMNS = ["Ticket Subject", "Ticket Description"]
CATEGORY_COLUMN = "Ticket Type"
PRIORITY_COLUMN = "Ticket Priority"


def ensure_project_dirs() -> None:
    """Create directories used by training and evaluation scripts."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the Kaggle customer support ticket dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place the CSV in data/raw/."
        )
    return pd.read_csv(path)


def prepare_ticket_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modeling columns used by the project."""
    missing = [col for col in [*TEXT_COLUMNS, CATEGORY_COLUMN, PRIORITY_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = df.copy()
    prepared["ticket_text"] = (
        prepared["Ticket Subject"].fillna("").astype(str)
        + " "
        + prepared["Ticket Description"].fillna("").astype(str)
    )
    prepared["category"] = prepared[CATEGORY_COLUMN].fillna("Unknown").astype(str)
    prepared["priority"] = prepared[PRIORITY_COLUMN].fillna("Unknown").astype(str)
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
