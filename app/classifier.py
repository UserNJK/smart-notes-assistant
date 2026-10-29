"""Note classifier: model definition (used by train.py) and loading/prediction (used by the app)."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"
LABELS = ["Study", "Money", "Travel", "Shopping"]


def build_pipeline(features: str = "char", C: float = 10.0) -> Pipeline:
    """TF-IDF + logistic regression.

    features="word": word unigrams+bigrams (first version).
    features="char": character 2-5-grams inside word boundaries (final version) - lets
    "examination" share features with "exam", "booked" with "booking", etc.
    """
    if features == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    elif features == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    else:
        raise ValueError(f"unknown features: {features}")
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=C, max_iter=2000))])


class NoteClassifier:
    """Wraps the saved pipeline and the uncertainty threshold chosen at training time."""

    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python train.py` first")
        bundle = joblib.load(path)
        self.pipeline: Pipeline = bundle["pipeline"]
        self.threshold: float = bundle["threshold"]

    def predict(self, text: str) -> dict:
        probs = self.pipeline.predict_proba([text])[0]
        best = int(probs.argmax())
        confidence = float(probs[best])
        return {
            "category": str(self.pipeline.classes_[best]),
            # Uncertain = the model's own top probability is low (out-of-scope or ambiguous note).
            "uncertain": confidence < self.threshold,
            "confidence": round(confidence, 4),
        }
