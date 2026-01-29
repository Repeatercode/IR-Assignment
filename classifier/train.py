from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "news_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "model.joblib"
LOW_CONFIDENCE_PERCENTILE = 5

def main():
    if not DATASET_PATH.exists():
        print("Dataset not found. Run: python -m classifier.rss_collect --per-class 40")
        return

    df = pd.read_csv(DATASET_PATH)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    preds = classifier.predict(X_test_vec)

    train_probs = classifier.predict_proba(X_train_vec)
    max_probs = train_probs.max(axis=1)
    confidence_threshold = float(np.percentile(max_probs, LOW_CONFIDENCE_PERCENTILE))

    print(f"Low-confidence percentile: {LOW_CONFIDENCE_PERCENTILE}")
    print(f"Confidence threshold: {confidence_threshold:.4f}")
    print(classification_report(y_test, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "confidence_threshold": confidence_threshold,
            "low_confidence_percentile": LOW_CONFIDENCE_PERCENTILE,
        },
        MODEL_PATH,
    )
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
