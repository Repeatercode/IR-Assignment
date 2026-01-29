import argparse
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "data" / "model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        return None
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict) and "vectorizer" in obj and "classifier" in obj:
        return obj
    return None


def predict_cluster(text: str):
    bundle = load_model()
    if bundle is None:
        return None
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    vec = vectorizer.transform([text])
    probs = classifier.predict_proba(vec)[0]
    best_idx = int(probs.argmax())
    confidence = float(probs[best_idx])
    label = str(classifier.classes_[best_idx])
    return best_idx, label, confidence


def predict_label(text: str) -> str:
    result = predict_cluster(text)
    if result is None:
        return ""
    _, label, _ = result
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    model = load_model()
    if model is None:
        print("Model not found or incompatible. Train first: python -m classifier.train")
        return

    result = predict_cluster(args.text)
    if result is None:
        print("Model not found or incompatible. Train first: python -m classifier.train")
        return
    _, label, confidence = result
    print(f"{label} (confidence={confidence:.2f})")


if __name__ == "__main__":
    main()
