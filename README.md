# Smart Notes Assistant

Save short notes, get them automatically classified as **Study / Money / Travel / Shopping** by a
model trained here (TF-IDF character n-grams + logistic regression), see an **Uncertain** badge when
the model is not confident (e.g. "Call grandma this weekend"), filter by category, and search notes
in plain language (sentence embeddings + keyword overlap).

One FastAPI process serves everything: the `POST /classify` API, note/search routes and a plain HTML UI.

## Setup (Python 3.11)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls CPU-only PyTorch (for sentence-transformers). The first start downloads the
~90 MB `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face (needs internet once; cached afterwards).
No API keys or paid services are used.

## Run

```bash
python train.py                          # trains, prints validation metrics, saves models/classifier.joblib
uvicorn app.main:app --port 8000         # starts API + UI
```

- UI: <http://127.0.0.1:8000/>
- Classify API: `POST http://127.0.0.1:8000/classify` with `{"text": "..."}` -> `{"category": "Money", "uncertain": false, "confidence": 0.76}`

```bash
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d "{\"text\":\"Pay my electricity bill tomorrow.\"}"
```

`models/classifier.joblib` is committed, so the server also starts without re-training; `python train.py` reproduces it exactly.

### Load the sample notes (N001–N012)

Either click **Load sample notes** in the UI, or (server stopped or running):

```bash
python load_starter.py
```

Running it twice adds nothing the second time (`INSERT OR IGNORE` on the note ID). Notes you add get IDs
`U0001`, `U0002`, ... so they can never reuse `N001`–`N012`.

Notes are stored in `notes.db` (SQLite) in the project folder; delete that file to start empty.
Set `NOTES_DB=/path/to/file.db` to use a different database.

## Other routes

| Route | Purpose |
| --- | --- |
| `POST /notes` `{"text": ...}` | classify + save a note (422 on empty text) |
| `GET /notes?category=Travel` / `?uncertain=true` | list / filter notes |
| `GET /search?q=what do I need to buy&k=5` | ranked search; returns `"message": "No notes match this query well."` when nothing scores above the relevance cut-off |
| `POST /notes/load-starter` | same as `load_starter.py` |

Interactive docs: <http://127.0.0.1:8000/docs>

## Tests and evaluations

```bash
pytest                                                    # 14 tests: /classify, invalid input, persistence/restart, starter IDs, filter, search
python examples/test_api.py --url http://127.0.0.1:8000/classify   # supplied smoke test (server must be running)
python scripts/eval_api.py                                 # validation accuracy/macro-F1 re-computed through /classify
python scripts/eval_search.py                              # hit rate @3 for keyword vs TF-IDF vs embedding vs hybrid search
python scripts/eval_uncertainty.py                         # out-of-scope vs validation flag rates per threshold
python scripts/error_analysis.py                           # evidence for each classification mistake
```

**Windows note:** if `examples/test_api.py` crashes with `UnicodeEncodeError ... '₹'`, that is the console
failing to *print* the `₹` sign, not the server. Run it as `python -X utf8 examples/test_api.py ...`.

All numbers are in [RESULTS.md](RESULTS.md). How AI tools were used is in [AI_WORKFLOW.md](AI_WORKFLOW.md).

## How it works

| File | Role |
| --- | --- |
| `train.py` | baselines (majority, keyword rules), word vs character TF-IDF + LogReg, picks the feature set by 5-fold CV on **train only**, saves pipeline + uncertainty threshold |
| `app/classifier.py` | model definition and `NoteClassifier.predict()` -> category, confidence (max probability), `uncertain = confidence < 0.45` |
| `app/store.py` | SQLite note storage and ID scheme |
| `app/search.py` | `KeywordSearch`, `TfidfSearch`, `EmbeddingSearch`, `HybridSearch` with one interface; the app uses hybrid; `visible_results()` applies the relevance cut-offs |
| `app/main.py` | FastAPI routes; model, DB and embedding model are loaded once at startup |
| `app/static/index.html` | the UI (plain HTML + JS) |
| `data/search_queries.json` | 23 search queries with expected note IDs + 5 queries that should match nothing |

## Limitations

- 156 training notes from a small number of templates: the model learns surface patterns. Words it has
  never seen (e.g. "railway", "cart") are handled only through character overlap, which can mislead it.
- `uncertain` is a single probability threshold. Out-of-scope notes shaped like in-scope ones
  ("Doctor's appointment on Tuesday at 4 PM." -> Study, 0.51) are not flagged, and some in-scope
  mistakes are confident ("I must reach the railway station by 10 AM." -> Study, 0.65).
- Search cut-offs (absolute 0.25, relative 0.5 of the best score) were chosen on only 23 + 5 queries;
  `water the plants` still shows a vegetables note. The hybrid search was designed after seeing failures on
  the same query set, so its perfect hit rate @3 is optimistic.
- No delete/edit of notes, no authentication; single-user local app.
- If sentence-transformers cannot load, search falls back to TF-IDF (a warning is printed) and quality drops.
