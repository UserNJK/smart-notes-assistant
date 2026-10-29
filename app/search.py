"""Note search. Four approaches with the same interface, compared in scripts/eval_search.py.

Each searcher has rank(query, notes) -> [(note, score), ...] sorted best first, and a
min_score used by visible_results() to decide what the app shows.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Hide results scoring below this fraction of the best result (e.g. "hotel": N007 0.82 vs others ~0.26).
RELATIVE_CUTOFF = 0.5


def visible_results(searcher, ranked, k):
    """What the app shows: nothing if the best score is below min_score ("no good match"),
    otherwise the top k that pass both the absolute and the relative cut-off."""
    if not ranked or ranked[0][1] < searcher.min_score:
        return []
    floor = max(searcher.min_score, RELATIVE_CUTOFF * ranked[0][1])
    return [(n, s) for n, s in ranked[:k] if s >= floor]

STOPWORDS = {
    "a", "an", "the", "my", "our", "your", "i", "me", "we", "is", "are", "am", "was", "be", "do",
    "does", "did", "to", "of", "in", "on", "at", "for", "by", "and", "or", "it", "this", "that",
    "what", "when", "where", "which", "who", "how", "need", "next", "have", "has", "any", "from",
}


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class KeywordSearch:
    """Baseline: fraction of query words (minus stopwords) that appear in the note."""
    name = "keyword"
    min_score = 0.01

    def rank(self, query, notes):
        q = set(tokenize(query))
        if not q:
            return []
        scored = [(n, len(q & set(tokenize(n["text"]))) / len(q)) for n in notes]
        return sorted(scored, key=lambda x: -x[1])


class TfidfSearch:
    """Cosine similarity of character n-gram TF-IDF vectors fitted on the notes being searched."""
    name = "tfidf"
    min_score = 0.15

    def rank(self, query, notes):
        if not notes:
            return []
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        doc_m = vec.fit_transform([n["text"] for n in notes])
        sims = (doc_m @ vec.transform([query]).T).toarray().ravel()
        return sorted(zip(notes, sims.tolist()), key=lambda x: -x[1])


class EmbeddingSearch:
    """Cosine similarity of sentence embeddings (all-MiniLM-L6-v2, runs on CPU)."""
    name = "embedding"
    min_score = 0.25

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import, only when used
        self.model = SentenceTransformer(model_name, device="cpu")
        self._cache = {}  # note text -> embedding, so each note is embedded once

    def _embed(self, texts):
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, e in zip(missing, self.model.encode(missing, normalize_embeddings=True)):
                self._cache[t] = e
        return np.array([self._cache[t] for t in texts])

    def rank(self, query, notes):
        if not notes:
            return []
        doc_m = self._embed([n["text"] for n in notes])
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return sorted(zip(notes, (doc_m @ q).tolist()), key=lambda x: -x[1])


class HybridSearch(EmbeddingSearch):
    """Embedding similarity + a bonus for exact keyword overlap.

    Embeddings find notes by meaning but under-score exact tokens (a weekday, "buy"); the keyword
    bonus fixes that. Queries sharing no words with any note get no bonus, so the cut-off still
    hides unrelated notes.
    """
    name = "hybrid"
    keyword_weight = 0.3

    def rank(self, query, notes):
        keyword = dict((n["id"], s) for n, s in KeywordSearch().rank(query, notes))
        scored = [(n, s + self.keyword_weight * keyword.get(n["id"], 0.0)) for n, s in super().rank(query, notes)]
        return sorted(scored, key=lambda x: -x[1])


def default_searcher():
    """The app uses hybrid search (best hit rate @3, see RESULTS.md); falls back to TF-IDF if unavailable."""
    try:
        return HybridSearch()
    except Exception as e:  # e.g. sentence-transformers not installed / model not downloadable
        print(f"WARNING: embedding search unavailable ({e}); falling back to TF-IDF search")
        return TfidfSearch()
