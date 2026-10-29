# AI_WORKFLOW

> DRAFT written with Claude Code from the session log. **Rewrite sections 2–3 in your own words
> before submitting**, and add or replace anything that does not match what you actually did.

## 1. Tools

- **Claude Code** (desktop app, Claude Opus 5.5): planning, writing the code, running the scripts and tests.
- Python 3.11, scikit-learn, FastAPI, sentence-transformers, pytest. No paid APIs at runtime.

Process: I gave Claude the assessment email and the candidate pack, asked for a plan first, reviewed it
(FastAPI + SQLite + TF-IDF/LogReg, the build order from START_HERE "if you are short on time"), then
let it build step by step while I read each script's output.

## 2. One delegated task (prompt as written)

> `first plan` (with the invitation email pasted), then after reviewing the plan: `Yes, start building`

What I kept: the single-process FastAPI design (avoids running Streamlit + a second API server), SQLite
storage, character n-gram model, embedding search.
What I changed / pushed back on: *(fill in: anything you changed or questioned)*

## 3. Problems caught and how the fix was proven

**a) Wrong note IDs (bug in AI-generated code).** The first version of `NoteStore.add()` derived user IDs
from SQLite's `AUTOINCREMENT` counter. Symptom: after loading the starter notes twice, the first user
note was saved as **`U0025`** instead of `U0001` (seen when testing manually with curl). Cause: SQLite
advances the AUTOINCREMENT counter even for the 12 duplicate starter inserts that `INSERT OR IGNORE` skipped.
IDs were still unique, but the numbering was wrong and depended on how often the starter notes were loaded.
Fix: compute `max(U-number) + 1` inside the INSERT statement itself (atomic). Proof: the regression test
`test_starter_notes_no_duplicates_and_ids_not_reused` loads the starter notes twice, adds a note and
asserts its ID is `U0001`. It passes (`pytest`: 14 passed).

**b) Unsupported claims in the write-up.** The first draft of RESULTS.md said "similarity to the nearest
training note would be a good second uncertainty signal; in-scope notes are much more similar" and
"~30 distinct templates in validation". Neither had been measured. Checked with a short script:
median nearest-neighbour similarity is 0.39 (validation) vs 0.36 (out-of-scope), so the signal is weak.
After removing weekdays and numbers there are 41 distinct validation notes. Both statements were corrected.

**c) Search looked fine in the numbers but not in the UI.** Embedding search scored 22/23 hit@3,
yet clicking through the UI, the task's own example `what do I need to buy?` showed only one of the two
"Buy ..." notes (N004 scored 0.21, under the 0.25 cut-off), and `hotel` showed two unrelated notes under
the real one. Hit@3 ignores what the cut-off hides. Fix: hybrid score (embedding + 0.3 × keyword overlap)
plus a relative cut-off, and `eval_search.py` now also measures what is *displayed* (expected note shown
21/23 -> 23/23, displayed results that are relevant 22/38 -> 26/35, no-match queries unchanged at 4/5).

**d) Smaller ones found while running things**
- The supplied `examples/test_api.py` crashed on Windows with `UnicodeEncodeError` for `₹`. The server
  response was fine: the cp1252 console could not *print* the character. `python -X utf8 ...` passes all checks; noted in README.
- The first `eval_search.py` counted a hit only if it was above the display cut-off. That hid TF-IDF's
  correct rank-1 answer for `train` (score 0.14 < 0.15) and mixed up two separate questions (ranking vs
  "is anything relevant"). Now hit@3 uses the raw ranking and the cut-off is reported separately.
- `error_analysis.py` listed features with positive contributions under "pushing to the true class",
  because it took the lowest values without checking their sign. Fixed by filtering on sign.

Approximate time: *(fill in)*
