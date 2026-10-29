"""Evidence for the classifier's validation mistakes.

    python scripts/error_analysis.py

For every misclassified validation note (final model, and the first word-only version) prints
the class probabilities, the n-gram features that pushed hardest towards the predicted class
versus the true class, and the most similar training notes. Also explains the final model's
least-confident correct notes and out-of-scope notes it was too confident about.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classifier import NoteClassifier, build_pipeline  # noqa: E402
from app.data import load_csv  # noqa: E402

TOP = 6


def explain(pipe, X_train, y_train, text, true, pred, true_name="true"):
    """Print why `pipe` prefers `pred` over `true` for `text`."""
    vec, lr = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names, classes = vec.get_feature_names_out(), list(lr.classes_)
    x = vec.transform([text])
    probs = {str(c): round(float(p), 2) for c, p in zip(classes, pipe.predict_proba([text])[0])}
    print(f"\n{text!r}\n  {true_name}={true} predicted={pred} probs={probs}")
    # Contribution of each feature to the (pred - true) logit difference: tfidf value * weight difference.
    diff = lr.coef_[classes.index(pred)] - lr.coef_[classes.index(true)]
    idx = x.nonzero()[1]
    contrib = x.toarray().ravel()[idx] * diff[idx]
    order = np.argsort(contrib)
    fmt = lambda ids: ", ".join(f"{names[idx[j]]!r}({contrib[j]:+.2f})" for j in ids)
    print(f"  features pushing to {pred}: {fmt([j for j in order[::-1][:TOP] if contrib[j] > 0])}")
    print(f"  features pushing to {true}: {fmt([j for j in order[:TOP] if contrib[j] < 0])}")
    sims = cosine_similarity(x, vec.transform(X_train)).ravel()
    for j in np.argsort(-sims)[:3]:
        print(f"  similar train note ({sims[j]:.2f}, {y_train[j]}): {X_train[j]!r}")


def mistakes(pipe, X_train, y_train, X_val, y_val, title):
    preds = pipe.predict(X_val)
    wrong = [i for i, (p, y) in enumerate(zip(preds, y_val)) if p != y]
    print(f"=== {title}: {len(wrong)} mistakes out of {len(X_val)} ===")
    for i in wrong:
        explain(pipe, X_train, y_train, X_val[i], y_val[i], preds[i])
    print()


def main():
    X_train, y_train = load_csv("train.csv")
    X_val, y_val = load_csv("validation.csv")
    X_oos, _ = load_csv("out_of_scope_examples.csv")
    final = NoteClassifier()
    pipe = final.pipeline
    mistakes(pipe, X_train, y_train, X_val, y_val, "Final model TF-IDF(char) + LogReg")

    print("=== Final model: least confident CORRECT validation notes (true vs runner-up class) ===")
    probs = pipe.predict_proba(X_val)
    classes = list(pipe.classes_)
    correct = [i for i in range(len(X_val)) if classes[probs[i].argmax()] == y_val[i]]
    for i in sorted(correct, key=lambda i: probs[i].max())[:3]:
        runner_up = str(classes[np.argsort(probs[i])[-2]])
        explain(pipe, X_train, y_train, X_val[i], y_val[i], runner_up, "true (correctly predicted)")
    print()

    print(f"=== Final model: out-of-scope notes NOT flagged (confidence >= {final.threshold}) ===")
    for text, p in zip(X_oos, pipe.predict_proba(X_oos)):
        if p.max() >= final.threshold:
            # No true label: show what pushed the guess over the runner-up.
            order = np.argsort(p)
            explain(pipe, X_train, y_train, text, classes[order[-2]], classes[order[-1]], "runner-up")
    print()

    first = build_pipeline("word").fit(X_train, y_train)
    mistakes(first, X_train, y_train, X_val, y_val, "First version TF-IDF(word) + LogReg")


if __name__ == "__main__":
    main()
