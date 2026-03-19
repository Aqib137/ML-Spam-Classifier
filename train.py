"""Train a spam classifier (SMS spam) and save a model artifact.

This script will:
  1) Download + extract the SMS Spam Collection dataset (if needed)
  2) Load and clean the text
  3) Vectorize text using TF-IDF
  4) Train a binary classifier (Logistic Regression)
  5) Evaluate and print metrics (accuracy + F1)
  6) Persist a pipeline to `model/model.pkl`

Run:
  python train.py

Output:
  - ml_spam_classifier/model/model.pkl
  - ml_spam_classifier/data/spam.csv

"""

from __future__ import annotations

import argparse
import os
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SPAM_CSV_PATH = os.path.join(DATA_DIR, "spam.csv")


def download_dataset_if_missing() -> None:
    """Download and prepare the dataset if it isn't already present."""

    if os.path.exists(SPAM_CSV_PATH):
        return

    # Dataset source: UCI SMS Spam Collection (same as the Kaggle dataset)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    )
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")

    print("Downloading dataset...")
    try:
        import requests

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        raise RuntimeError(
            "Unable to download dataset. Please ensure you have internet access "
            "or manually download the file and place it at ml_spam_classifier/data/spam.csv."
        ) from exc

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target = None
        for name in names:
            if name.lower().endswith("smsspamcollection") or name.lower().endswith("smsspamcollection.txt"):
                target = name
                break

        if target is None:
            raise RuntimeError("Expected SMS Spam Collection file inside zip not found")

        with zf.open(target) as f:
            lines = [line.decode("utf-8", errors="replace") for line in f.readlines()]

    # Parse TSV-like text: label\tmessage
    records = []
    for i, line in enumerate(lines):
        parts = line.rstrip("\n\r").split("\t", 1)
        if len(parts) != 2:
            continue
        label, msg = parts
        records.append({"label": label.strip(), "text": msg.strip()})

    df = pd.DataFrame(records)
    df.to_csv(SPAM_CSV_PATH, index=False)
    print(f"Wrote {len(df)} rows to {SPAM_CSV_PATH}")


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove non-text tokens."""

    text = text or ""
    text = text.strip().lower()

    # Replace URLs and email-like tokens with placeholders
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)

    # Remove non-alphanumeric characters, keep spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data() -> pd.DataFrame:
    """Load dataset (download if missing) and return a DataFrame."""

    os.makedirs(DATA_DIR, exist_ok=True)
    download_dataset_if_missing()

    # Some versions of the SMS spam dataset include non-UTF8 characters.
    # Use a permissive encoding and skip malformed rows.
    df = pd.read_csv(
        SPAM_CSV_PATH,
        encoding="latin-1",
        on_bad_lines="skip",
        engine="python",
    )
    # The official SMS Spam Collection dataset uses v1/v2 column names.
    if "text" not in df.columns or "label" not in df.columns:
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "text"})
        else:
            raise RuntimeError("Expected columns 'text' and 'label' in data")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)

    # Filter out empty strings after cleaning
    df = df[df["text"].str.strip().astype(bool)]

    return df


def build_pipeline() -> Pipeline:
    """Build a text classification pipeline."""

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )
    return pipeline


def train(args: argparse.Namespace) -> None:
    df = load_data()
    X = df["text"].values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=args.test_size,
        stratify=y_encoded,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    print("Evaluation")
    print("----------")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (spam class): {f1:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")

    joblib.dump({"pipeline": pipeline, "label_encoder": encoder}, model_path)
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spam classifier")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data to hold out for evaluation (default: 0.25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
