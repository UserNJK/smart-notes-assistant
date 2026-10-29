# RESULTS

Every number below is printed by the command named next to it, and was re-checked on the final code.

## Dataset and training

- Training rows: 156 (Study 48, Money 36, Travel 39, Shopping 33)
- Validation rows: 50 (never used for fitting)
- Command: `python train.py`
- Baselines: (1) always predict the majority class (Study); (2) keyword rules: count words from the
  category definitions in `CATEGORY_GUIDE.md` (e.g. exam, bill, flight, buy), fall back to Study.
- Trained model: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), sublinear_tf=True)` +
  `LogisticRegression(C=10, max_iter=2000)`. First version used word 1–2-grams instead (see "fix" below).
- Python 3.11.4, scikit-learn 1.9.1, numpy 2.4.6. CV split seed 42; LogisticRegression (lbfgs) is deterministic.

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 | 5-fold CV macro-F1 on train |
| --- | ---: | ---: | ---: |
| Baseline: majority class | 0.300 | 0.115 | – |
| Baseline: keyword rules | 0.680 | 0.677 | – |
| First model: TF-IDF word 1–2-grams + LogReg | 0.640 | 0.666 | 0.960 |
| **Final model: TF-IDF char 2–5-grams + LogReg** | **0.920** | **0.924** | 0.973 |

The same 0.920 / 0.924 is reproduced through the live endpoint with `python scripts/eval_api.py`.

Final model per category:

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Study | 0.833 | 1.000 | 0.909 | 15 |
| Money | 1.000 | 1.000 | 1.000 | 11 |
| Travel | 0.909 | 0.769 | 0.833 | 13 |
| Shopping | 1.000 | 0.909 | 0.952 | 11 |

Confusion matrix (rows = true, columns = predicted; order Study, Money, Travel, Shopping):

```
[[15  0  0  0]
 [ 0 11  0  0]
 [ 3  0 10  0]
 [ 0  0  1 10]]
```

Notable: the first (word) model scored *below* the keyword baseline on validation even though its
train CV was 0.960. The training notes are built from a few templates ("Book a hotel in X for Y"), so
random K-fold puts near-identical sentences on both sides of the split and CV is inflated.
Validation uses different phrasings, which exposes that.

## Error analysis

Evidence from `python scripts/error_analysis.py`. It prints, for each note, the class probabilities,
the n-gram features with the largest contribution (tf-idf value × weight difference between the two
classes) and the three most similar training notes (cosine on the model's own TF-IDF vectors).
The final model makes 4 mistakes on 50 notes, but 3 of them are one template, so there are only
2 distinct mistakes; I add two difficult cases to make up the analysis.

| # | Note | Expected | Predicted (confidence) | Reason (evidence below) |
| --- | --- | --- | --- | --- |
| 1 | I must reach the railway station by 10 AM / 2 PM / 8 AM (3 notes, 1 mistake) | Travel | Study (0.61–0.65, **not** flagged) | "railway" and "reach" never occur in train; char n-grams `st`, `st `, `ati` from "must"/"station" match Study words (test, statistics, study) |
| 2 | Put snacks and cleaning items in the cart. | Shopping | Travel (0.37, flagged uncertain) | "snacks", "cart", "items" never occur in train; ` in ` pushes Travel ("Book a hotel in Chennai") |
| 3 | Our return journey from Jaipur is on Wednesday / Friday (difficult, correct) | Travel | Travel (0.35 / 0.39, flagged) | "return" only appears in train as "Return the damaged … to the store" (Shopping), so `turn`, `retur` push Shopping (0.31) |
| 4 | Doctor's appointment on Tuesday at 4 PM. (out-of-scope, not flagged) | none | Study (0.51) | ` at ` pushes Study (train: "Attend the … tutorial at 10 AM"); a date/time-shaped note looks in-scope |

**Evidence for mistake 1** (`I must reach the railway station by 10 AM.`):

```
true=Travel predicted=Study probs={'Money': 0.11, 'Shopping': 0.06, 'Study': 0.65, 'Travel': 0.17}
features pushing to Study:  'st'(+0.21), 'st '(+0.10), 'ati'(+0.08), 'io'(+0.07), 'h '(+0.07), 'ch '(+0.07)
features pushing to Travel: 'ai'(-0.16), 'i '(-0.14), 'mu'(-0.09), ' mu'(-0.09), ' i'(-0.06), 'ra'(-0.06)
similar train note (0.27, Travel): 'The boarding time for the plane is 10 AM.'
similar train note (0.26, Study):  'Prepare a presentation for the statistics seminar.'
similar train note (0.26, Travel): 'Find directions from the station to our hotel.'
```

`grep -ci "railway\|reach" data/train.csv` returns 0. The only Travel signal is "station", which
appears once in train, and is outweighed by many small Study n-grams. The nearest neighbours are a
tie between Travel and Study at 0.26–0.27, so nothing in train strongly resembles this note.
The old word model made the same mistake for a different reason: `by` (+0.40 towards Study, from
"Complete the economics homework by Saturday") outweighed `station`.

**Evidence for difficult case 3** (`Our return journey from Jaipur is on Wednesday.`):

```
true=Travel runner-up=Shopping probs={'Money': 0.2, 'Shopping': 0.31, 'Study': 0.14, 'Travel': 0.35}
features pushing to Shopping: 'urn'(+0.09), 'rn'(+0.07), 'et'(+0.06), 'ed'(+0.06), 'turn'(+0.05), 'retur'(+0.05)
```

In `train.csv` every note with "return" is Shopping (`Return the damaged notebook/backpack/keyboard/desk lamp to the store.`).

## The fix I tried: word n-grams -> character n-grams

The first version (word 1–2-grams) made 18 validation mistakes. `error_analysis.py` showed the
model had learned **function words** as category signals, because of the templated training notes:

```
'I need to study mathematics for next week.'   true=Study predicted=Travel
  features pushing to Travel: 'to'(+1.82), 'for'(+0.71)      pushing to Study: 'mathematics'(-1.45)
'The teacher moved our biology quiz to Tuesday.' true=Study predicted=Travel
  features pushing to Travel: 'to'(+1.31), 'our'(+1.02)      pushing to Study: 'biology'(-1.13)
```

("flight **to** Delhi", "**our** hotel" make `to`/`our` Travel words.) It also could not relate unseen
word forms ("coursework" vs "homework", "settled" vs "payment").

Reasoning: character n-grams inside word boundaries share features between related word forms and
spread the weight of a short function word over many small n-grams. I chose between the two feature sets
**by 5-fold cross-validation on the training set only** (word 0.960 vs char 0.973 macro-F1;
`train.py` selects the higher one automatically). C=10 was fixed: CV macro-F1 for char n-grams is flat
(0.973–0.974) for C between 1 and 30, and a larger C spreads the probabilities so the uncertainty
threshold separates in-scope from out-of-scope notes better (see next section).

Result on the unchanged validation set: accuracy 0.640 -> 0.920, macro-F1 0.666 -> 0.924; mistakes 18 -> 4.
Caveat: the CV improvement (+0.013) is much smaller than the validation improvement (+0.26), again
because train CV is inflated by near-duplicates. CV pointed in the right direction but it underestimated the gain.

## Search evaluation

- Query file: `data/search_queries.json`. Command: `python scripts/eval_search.py`
- 23 queries with expected note IDs over the 12 starter notes; **12 share no non-stopword with their
  target note**. The script checks this itself rather than relying on a hand label, e.g. `money transfer` -> N006,
  `groceries` -> N012, `plane journey` -> N003, `send back the broken earphones` -> N008, `bank` -> N006.
- Plus 5 queries that should match nothing (`dentist appointment`, `water the plants`, ...), used to tune
  the "no good match" cut-off.

| Search approach | Hit rate @3 | Hit @3, 12 no-overlap queries | Expected note displayed | Displayed results that are expected | No-match queries shown nothing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword baseline (fraction of query words in the note) | 11/23 = 0.48 | 0/12 | 11/23 | 13/16 | 5/5 |
| TF-IDF character 3–5-grams, cosine | 17/23 = 0.74 | 7/12 | 11/23 | 11/19 | 1/5 |
| Sentence embeddings (all-MiniLM-L6-v2), cosine | 22/23 = 0.96 | 12/12 | 21/23 | 22/38 | 4/5 |
| **Hybrid: embedding + 0.3 × keyword overlap** | **23/23 = 1.00** | **12/12** | **23/23** | **26/35** | **4/5** |

- *Hit rate @3* is measured on the raw ranking (top 3 notes with score > 0), as defined in the task.
- The last three columns measure what the user actually sees after the app's cut-offs
  (`visible_results()`: nothing if the best score < `min_score`; otherwise the top 3 scoring at least
  `max(min_score, 0.5 × best score)`).

How I got to the hybrid: embeddings alone missed `what is happening on Thursday?`. While checking
the UI I also saw that `what do I need to buy?`, the task's own example, displayed only N012:
N004 "Buy a new laptop stand" scored 0.21, under the 0.25 cut-off, even though it contains "buy".
Embeddings under-weight exact tokens (weekdays, short verbs). Adding the keyword baseline's score
×0.3 lifts exact matches (N004 -> 0.51). Queries that share no word with any note get no bonus, so the
no-match behaviour is unchanged (same 4/5). The weight 0.3 was set by reasoning (a full keyword match
should outweigh the typical 0.1–0.2 gap between neighbours) and not swept. Still, the hybrid was
designed after I had seen the embedding failures on this query set, so its 23/23 is optimistic.

The relative cut-off was added because `hotel` displayed N007 (0.82) *and* two unrelated notes scoring
0.27 / 0.25. Results under half the best score are now hidden.

**The app uses hybrid search**: best on every column, CPU only, note embeddings are cached, and a
search takes milliseconds once the model has loaded at startup. If sentence-transformers cannot load,
it falls back to TF-IDF.

**Queries that still fail:**
- `water the plants` (should match nothing) shows N012 "Buy rice, eggs, and vegetables from the
  supermarket." (score 0.28): the embedding links plants and vegetables. The no-match queries' best scores
  (0.20–0.28) overlap the weakest correct hits (min 0.29 hybrid, 0.24 embedding-only), so no single
  cut-off separates them on this set.
- `what is happening on Thursday?` is a hit @3 but ranks N008 second (0.37) behind N001 "My Python test
  is on Friday at 10 AM." (0.38): every note with a day/time looks alike to the embedding.
- Precision: 9 of the 35 displayed results across the 23 queries are not expected notes.

## Notes that fit no category

- Method: `uncertain = max predicted probability < 0.45`, using the trained model's `predict_proba`.
  No keyword list. Command: `python scripts/eval_uncertainty.py`
- I did **not** train on `out_of_scope_examples.csv`. All 12 notes are used only to choose the threshold.
  Since the same 12 notes and the validation set were used to choose the threshold, the flag rates
  below are optimistic for unseen notes.

| Threshold | Out-of-scope notes flagged | Validation notes flagged | Validation errors flagged |
| --- | ---: | ---: | ---: |
| 0.35 | 4/12 = 0.33 | 2/50 = 0.04 | 0/4 |
| 0.40 | 8/12 = 0.67 | 4/50 = 0.08 | 1/4 |
| **0.45 (chosen)** | **10/12 = 0.83** | **9/50 = 0.18** | 1/4 |
| 0.50 | 10/12 = 0.83 | 13/50 = 0.26 | 1/4 |
| 0.55 | 12/12 = 1.00 | 17/50 = 0.34 | 1/4 |

- Why 0.45: it is the knee of the curve. Going 0.40 -> 0.45 flags 2 more out-of-scope notes for 5 more
  validation notes, while 0.45 -> 0.50 flags no extra out-of-scope notes and costs 4 more validation notes.
  Catching all 12 needs 0.55, which flags a third of the in-scope notes, too many for the badge to mean anything.
- The flagged validation notes are reasonable "unsure" cases: `Our return journey from Jaipur ...`
  (see case 3), `Check the credit card balance on ...` (3 variants), `Look for a sale on a water bottle ...`
  and the misclassified `Put snacks and cleaning items in the cart.`
- Not flagged: `Take the trash out on Thursday night.` (Study 0.53) and `Doctor's appointment on Tuesday at 4 PM.` (Study 0.51).
- Weakness: the railway-station mistake is confidently wrong (0.61–0.65), so the flag catches only 1 of the 4 validation errors.
  Max probability measures distance from the decision boundary, not "have I seen anything like this".

## What I would improve next

- A better out-of-scope signal. I checked one candidate, similarity to the nearest training note, and it
  would *not* help much: median max-similarity is 0.39 for validation notes vs 0.36 for out-of-scope notes
  (character n-grams of short notes overlap with everything). More promising: an explicit "Other" class
  trained on half of the out-of-scope notes (evaluated on the other half), or more out-of-scope data.
- Search: a learned or swept combination weight, and a larger query set (the current one was also used to design the hybrid).
- Group-aware cross-validation (by sentence template) so train CV stops overstating accuracy.
- More varied training phrasing for Travel ("railway", "reach", "journey") and Shopping ("cart").

## Limitations

- The dataset is small and synthetic, so it cannot show real-world reliability.
- How much I trust the 0.92 validation score: only moderately. 50 notes is small (one note = 2 points),
  and validation is also templated: the 4 errors come from 2 templates, so one template decides 6 points.
  After removing weekdays and numbers only 41 of the 50 notes are distinct (fewer if cities/subjects are
  also ignored). A ±10-point swing on a fresh set would not surprise me.
- The uncertainty threshold and the search cut-off were tuned on the same small sets they are reported on.
