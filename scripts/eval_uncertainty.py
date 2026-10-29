"""Uncertainty trade-off: share of out-of-scope vs validation notes flagged at several thresholds.

    python scripts/eval_uncertainty.py

Uses the saved model (run train.py first). The out-of-scope notes are never used for training.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classifier import NoteClassifier  # noqa: E402
from app.data import load_csv  # noqa: E402


def main():
    clf = NoteClassifier()
    X_val, y_val = load_csv("validation.csv")
    X_oos, _ = load_csv("out_of_scope_examples.csv")
    val_conf = clf.pipeline.predict_proba(X_val).max(axis=1)
    oos_conf = clf.pipeline.predict_proba(X_oos).max(axis=1)
    val_wrong = clf.pipeline.predict(X_val) != np.array(y_val)

    print(f"Saved model threshold: {clf.threshold}\n")
    print(f"{'threshold':>9} | {'OOS flagged':>12} | {'val flagged':>12} | val errors caught")
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        oos_f, val_f = oos_conf < t, val_conf < t
        mark = "  <- chosen" if abs(t - clf.threshold) < 1e-9 else ""
        print(f"{t:>9.2f} | {oos_f.sum():>3}/{len(X_oos)} = {oos_f.mean():.2f} | {val_f.sum():>3}/{len(X_val)} = {val_f.mean():.2f}"
              f" | {(val_wrong & val_f).sum()}/{val_wrong.sum()}{mark}")

    print("\nOut-of-scope notes (max probability, best guess):")
    for text, c, guess in sorted(zip(X_oos, oos_conf, clf.pipeline.predict(X_oos)), key=lambda x: -x[1]):
        print(f"  {c:.2f} {guess:<8} {'flagged' if c < clf.threshold else 'NOT flagged':<11} {text}")
    print("\nFlagged validation notes:")
    for text, c, y, wrong in zip(X_val, val_conf, y_val, val_wrong):
        if c < clf.threshold:
            print(f"  {c:.2f} true={y:<8} {'(misclassified)' if wrong else '':<15} {text}")


if __name__ == "__main__":
    main()
