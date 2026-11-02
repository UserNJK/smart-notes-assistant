# Smart Notes Assistant

Save short notes, get them automatically classified as **Study / Money / Travel / Shopping** by a
model trained here (TF-IDF character n-grams + logistic regression), see an **Uncertain** badge when
the model is not confident (e.g. "Call grandma this weekend"), filter by category, and search notes
in plain language (sentence embeddings + keyword overlap).

One FastAPI process serves everything: the `POST /classify` API, note/search routes and a plain HTML UI.

## Quick start: run these in order

Requirements: **Python 3.11** and git. Internet is needed once, for `pip install` and the first server start.
No API keys or paid services are used.

Commands are for **Windows PowerShell**; macOS/Linux versions are shown where they differ.
You need **two terminals**:

| Terminal | Used for | Stays open? |
| --- | --- | --- |
| **Terminal 1** | setup (steps 1–5), then the running server (step 6) | Yes, the server runs here until `Ctrl+C` |
| **Terminal 2** | everything else while the server runs: sample notes, API calls, tests (steps 8–10) | Use as needed |
| Browser | the UI (step 7) | |

### Terminal 1: set up and start the server

**1. Get the code** (Terminal 1, one time)

```powershell
git clone https://github.com/UserNJK/smart-notes-assistant.git
cd smart-notes-assistant
```

**2. Create a Python 3.11 virtual environment** (Terminal 1, one time)

```powershell
py -3.11 -m venv .venv          # macOS/Linux: python3.11 -m venv .venv
```

Use `py -3.11` on Windows: plain `python` may be a different version.

**3. Activate it** (Terminal 1)

```powershell
.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. If PowerShell says running scripts is disabled, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal, then activate again.

**4. Install dependencies** (Terminal 1, one time, a few minutes: includes CPU-only PyTorch)

```powershell
pip install -r requirements.txt
```

**5. Train the model** (Terminal 1, about 10 seconds)

```powershell
python train.py
```

Prints the baseline and model validation metrics and saves `models/classifier.joblib`, the file the app loads.
That file is also committed, so this step is optional; re-running it reproduces it exactly.

**6. Start the app** (Terminal 1: API + UI in one process)

```powershell
uvicorn app.main:app --port 8000
```

Wait for `Application startup complete.` The first start also downloads the ~90 MB
`sentence-transformers/all-MiniLM-L6-v2` search model (cached afterwards).
**Leave Terminal 1 running.** Stop the server with `Ctrl+C` in Terminal 1. If port 8000 is busy, use
`--port 8001` and change the URLs below to match.

### Browser: use the app

**7. Open the UI:** <http://127.0.0.1:8000/>

- Click **Load sample notes** to load `data/starter_notes.json` (N001–N012). Loading twice adds nothing the second time.
- Add a note, filter by category, search. Notes you add get IDs `U0001`, `U0002`, ... and never reuse `N001`–`N012`.
- API docs: <http://127.0.0.1:8000/docs>

### Terminal 2: while the server keeps running in Terminal 1

**8. Open Terminal 2 in the project folder and activate the environment**

```powershell
cd smart-notes-assistant          # wherever you cloned it
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
```

**9. Optional: load the sample notes from the command line** (Terminal 2; same as the UI button)

```powershell
python load_starter.py
```

**10. Try the classify API** (Terminal 2)

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify -ContentType "application/json" -Body '{"text":"Pay my electricity bill tomorrow."}'
```

macOS/Linux (or Git Bash):

```bash
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d '{"text":"Pay my electricity bill tomorrow."}'
```

Response: `{"category":"Money","uncertain":false,"confidence":0.8696}`.
Tests and evaluations (next section) also run in Terminal 2.

**Next time:** Terminal 1: `cd` into the project, then steps 3 and 6. Terminal 2 (if needed): step 8.

### Where notes are stored

Notes are stored in `notes.db` (SQLite) in the project folder, so they survive restarts. Delete that file
(with the server stopped) to start empty. Set the `NOTES_DB` environment variable to use a different file.

## Other routes

| Route | Purpose |
| --- | --- |
| `POST /notes` `{"text": ...}` | classify + save a note (422 on empty text) |
| `GET /notes?category=Travel` / `?uncertain=true` | list / filter notes |
| `GET /search?q=what do I need to buy&k=5` | ranked search; returns `"message": "No notes match this query well."` when nothing scores above the relevance cut-off |
| `POST /notes/load-starter` | same as `load_starter.py` |

Interactive docs: <http://127.0.0.1:8000/docs>

## Tests and evaluations

Run these in **Terminal 2** (activated, step 8). `examples/test_api.py` and `scripts/eval_api.py` need the
server running in Terminal 1. `pytest` and the other `scripts/` evaluations don't: they start their own copy
of the app, so they also work with the server stopped.

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
