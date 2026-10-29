"""Compare search approaches on data/search_queries.json over the 12 starter notes.

    python scripts/eval_search.py

Reports hit rate @3 (an expected note in the top 3 of the raw ranking) on queries with expected
notes, then what the app would actually display after its cut-offs (app.search.visible_results):
whether the expected note is shown, how many shown results are expected ones, and whether queries
with "expected": [] (nothing should match) correctly show nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data import DATA_DIR, load_starter_notes  # noqa: E402
from app.search import (RELATIVE_CUTOFF, EmbeddingSearch, HybridSearch, KeywordSearch, TfidfSearch,  # noqa: E402
                        tokenize, visible_results)

K = 3


def main():
    notes = load_starter_notes()
    by_id = {n["id"]: n for n in notes}
    queries = json.loads((DATA_DIR / "search_queries.json").read_text(encoding="utf-8"))
    positives = [q for q in queries if q["expected"]]
    negatives = [q for q in queries if not q["expected"]]

    # Verify (not just claim) which queries share no non-stopword with their target notes.
    no_overlap = [q["query"] for q in positives
                  if all(not set(tokenize(q["query"])) & set(tokenize(by_id[e]["text"])) for e in q["expected"])]
    print(f"{len(positives)} queries with expected notes, {len(no_overlap)} share no words with their target:")
    for q in no_overlap:
        print(f"   - {q}")
    print(f"{len(negatives)} queries that should match nothing\n")

    searchers = [KeywordSearch(), TfidfSearch(), EmbeddingSearch(), HybridSearch()]
    for s in searchers:
        hits, no_overlap_hits, failures, hit_scores = 0, 0, [], []
        for q in positives:
            # Ranking quality: top 3 among notes with any score > 0 (a score of 0 is not a match).
            top = [(n["id"], sc) for n, sc in s.rank(q["query"], notes) if sc > 0][:K]
            matched = [sc for nid, sc in top if nid in q["expected"]]
            if matched:
                hits += 1
                no_overlap_hits += q["query"] in no_overlap
                hit_scores.append(max(matched))
            else:
                failures.append((q["query"], q["expected"], [(nid, round(sc, 2)) for nid, sc in top]))
        # What the app would actually display (absolute + relative cut-off, top K).
        shown = {q["query"]: [n["id"] for n, _ in visible_results(s, s.rank(q["query"], notes), K)] for q in queries}
        empty_ok = sum(not shown[q["query"]] for q in negatives)
        shown_hit = sum(bool(set(shown[q["query"]]) & set(q["expected"])) for q in positives)
        shown_pos = [nid for q in positives for nid in shown[q["query"]]]
        relevant = sum(nid in q["expected"] for q in positives for nid in shown[q["query"]])
        neg_best = [max(sc for _, sc in s.rank(q["query"], notes)) for q in negatives]
        print(f"{s.name:<10} hit@{K} = {hits}/{len(positives)} = {hits / len(positives):.2f}"
              f"   (no-overlap queries: {no_overlap_hits}/{len(no_overlap)})")
        print(f"           as displayed (min_score={s.min_score}, relative {RELATIVE_CUTOFF}):"
              f" expected note shown {shown_hit}/{len(positives)},"
              f" shown results that are expected {relevant}/{len(shown_pos)},"
              f" no-match queries correctly empty {empty_ok}/{len(negatives)}")
        for q in negatives:
            if shown[q["query"]]:
                print(f"     no-match query {q['query']!r} still shows {shown[q['query']]}")
        if hit_scores:
            print(f"           score of correct hits: min {min(hit_scores):.2f}, median {sorted(hit_scores)[len(hit_scores) // 2]:.2f};"
                  f" best score on no-match queries: {', '.join(f'{b:.2f}' for b in neg_best)}")
        for query, expected, top in failures:
            print(f"     MISS {query!r} expected {expected}, top-3 {top}")
        print()


if __name__ == "__main__":
    main()
