"""Load a trained spam classifier and run predictions.

Usage examples:
  python predict.py --text "Free entry in 2 a wkly comp to win FA Cup final tkts"
  python predict.py --file examples.txt

The model is expected at ml_spam_classifier/model/model.pkl
"""

from __future__ import annotations

import argparse
import os
from typing import List

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run `train.py` first to create the model."
        )

    obj = joblib.load(path)
    pipeline = obj["pipeline"]
    label_encoder = obj["label_encoder"]
    return pipeline, label_encoder


def predict_texts(texts: List[str]):
    pipeline, encoder = load_model()
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)

    # Do not assume a particular ordering of classes; map with encoder
    classes = encoder.inverse_transform([0, 1])

    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append(
            {
                "text": text,
                "prediction": encoder.inverse_transform([pred])[0],
                "probabilities": dict(zip(classes, prob.tolist())),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam/ham for raw text")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text string to classify")
    group.add_argument(
        "--file",
        type=str,
        help="Path to a file containing one text per line to classify",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]

    results = predict_texts(texts)

    for r in results:
        print("---")
        print(f"Text: {r['text']}")
        print(f"Prediction: {r['prediction']}")
        print("Probabilities:")
        for cls, p in r["probabilities"].items():
            print(f"  {cls}: {p:.4f}")


if __name__ == "__main__":
    main()
