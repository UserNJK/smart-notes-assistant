"""Re-run the validation set through the running POST /classify endpoint (as reviewers will).

    python scripts/eval_api.py --url http://127.0.0.1:8000/classify

Should print the same accuracy / macro-F1 as train.py, plus the out-of-scope flag rate.
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data import load_csv  # noqa: E402


def classify(url, text):
    req = Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=4) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/classify")
    url = p.parse_args().url
    X_val, y_val = load_csv("validation.csv")
    X_oos, _ = load_csv("out_of_scope_examples.csv")
    val = [classify(url, t) for t in X_val]
    oos = [classify(url, t) for t in X_oos]
    preds = [r["category"] for r in val]
    print(f"validation via API: accuracy={accuracy_score(y_val, preds):.3f}"
          f"  macro-F1={f1_score(y_val, preds, average='macro'):.3f}")
    print(f"uncertain flagged: validation {sum(r['uncertain'] for r in val)}/{len(val)},"
          f" out-of-scope {sum(r['uncertain'] for r in oos)}/{len(oos)}")


if __name__ == "__main__":
    main()
