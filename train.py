"""Train and evaluate the note classifier.

    python train.py

Reads data/train.csv, compares baselines and two model versions, prints validation metrics
and saves models/classifier.joblib (the file the app and /classify load).
Validation labels are only used for reporting, never for fitting.
"""
import re
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.classifier import LABELS, MODEL_PATH, build_pipeline
from app.data import load_csv

SEED = 42
C = 10.0
# Chosen with scripts/eval_uncertainty.py using validation + out-of-scope notes (see RESULTS.md).
UNCERTAIN_THRESHOLD = 0.45

# Keyword baseline: words taken from CATEGORY_GUIDE.md definitions, not from the validation set.
KEYWORDS = {
    "Study": ["exam", "test", "assignment", "homework", "lecture", "study", "revise", "course",
              "quiz", "chapter", "project", "professor", "class", "tutorial", "reading", "lab"],
    "Money": ["pay", "bill", "rent", "bank", "savings", "transfer", "budget", "salary", "loan",
              "insurance", "invest", "deposit", "emi", "tax", "₹"],
    "Travel": ["flight", "train", "bus", "hotel", "trip", "travel", "ticket", "itinerary",
               "luggage", "station", "airport", "visa", "holiday", "booking"],
    "Shopping": ["buy", "order", "purchase", "return", "shop", "groceries", "supermarket",
                 "store", "compare", "deliver"],
}


def keyword_predict(text: str, fallback: str) -> str:
    tokens = re.findall(r"\w+|₹", text.lower())
    scores = {cat: sum(t in words for t in tokens) for cat, words in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else fallback


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    print(f"  {name:<34} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    return acc, f1


def main():
    X_train, y_train = load_csv("train.csv")
    X_val, y_val = load_csv("validation.csv")
    print(f"train rows={len(X_train)}  validation rows={len(X_val)}")
    print(f"train label counts: {dict(Counter(y_train))}\n")

    print("Validation results")
    majority = Counter(y_train).most_common(1)[0][0]
    report(f"Baseline: majority ({majority})", y_val, [majority] * len(X_val))
    report("Baseline: keyword rules", y_val, [keyword_predict(t, majority) for t in X_val])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {}
    for features in ["word", "char"]:
        pipe = build_pipeline(features, C)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()
        pipe.fit(X_train, y_train)
        report(f"TF-IDF({features}) + LogReg", y_val, pipe.predict(X_val))
        print(f"  {'':<34} (5-fold CV macro-F1 on train: {cv_f1:.3f})")
        models[features] = (pipe, cv_f1)

    # Pick the final feature set by cross-validation on TRAIN only.
    final_name = max(models, key=lambda k: models[k][1])
    final = models[final_name][0]
    print(f"\nFinal model (best train CV): TF-IDF({final_name}) + LogReg, C={C}")

    y_pred = final.predict(X_val)
    print(classification_report(y_val, y_pred, labels=LABELS, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):", LABELS)
    print(confusion_matrix(y_val, y_pred, labels=LABELS))

    conf = final.predict_proba(X_val).max(axis=1)
    flagged = conf < UNCERTAIN_THRESHOLD
    print(f"\nUncertain threshold={UNCERTAIN_THRESHOLD}: {flagged.mean():.0%} of validation notes flagged")
    wrong = np.array(y_pred) != np.array(y_val)
    print(f"  errors among flagged: {(wrong & flagged).sum()}/{flagged.sum()},"
          f" errors among confident: {(wrong & ~flagged).sum()}/{(~flagged).sum()}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": UNCERTAIN_THRESHOLD}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
