# RESULTS

All numbers in this file are printed by the committed code (`python train.py`,
`python experiments.py`, `python threshold_analysis.py`, `python eval_search.py`,
`python error_evidence.py`) on this dataset, not copied from a memo.

## Dataset and training

- Training rows: **156** (`data/train.csv`), validation rows: **50**
  (`data/validation.csv`), out-of-scope notes: **12**
  (`data/out_of_scope_examples.csv`, used for evaluation only, never trained on)
- Command that reproduces these numbers: `python train.py`
- Baselines: (1) most-frequent-training-label, (2) keyword rules scored from
  the `CATEGORY_GUIDE.md` definitions
- Trained model / features: TF-IDF, word n-grams 1–2, sublinear TF, custom
  stopword list (`app/textnorm.py`) + `LogisticRegression` (C=1.0,
  max_iter=2000, `random_state=42`), scikit-learn
- Library versions: scikit-learn 1.9.1, numpy 2.4.3, FastAPI 0.136.1,
  Python 3.13; seed fixed at 42

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Baseline: most-frequent label | 0.300 | 0.115 |
| Baseline: keyword rules | 0.680 | 0.677 |
| **Model: TF-IDF + LogReg** | **0.860** | **0.859** |

Confusion matrix (rows = true, cols = predicted):

|            | Study | Money | Travel | Shopping |
| ---        | ---: | ---: | ---: | ---: |
| Study      | 15 | 0 | 0 | 0 |
| Money      | 0 | 10 | 1 | 0 |
| Travel     | 2 | 0 | 9 | 2 |
| Shopping   | 2 | 0 | 0 | 9 |

Baselines were trained by `python train.py` in the same run (the keyword rules
are deterministic; the most-frequent label is computed from training counts).
The trained model beats the keyword baseline by +18 accuracy points.

## Error analysis (distinct mistakes)

Ten validation rows are misclassified; they reduce to **5 distinct mistakes**
after grouping notes that differ only in day/city/subject (all five came from
`python train.py`'s misclassified printout, evidence from `python
error_evidence.py`):

| # | Note (representative) | Expected | Predicted | Reason |
| --- | --- | --- | --- | --- |
| 1 | "I must reach the railway station by 10 AM." (×2: 10 AM / 2 PM) | Travel | Study | `railway`, `reach`, `railway station` never seen in training; the only positive signal for Travel is `station` (+0.19), while in the competing Study column `station` carries −0.08 and the bare token `10` ("at 10 AM" appears in Study training notes) pushes Study (+0.19). Nearest training neighbours span 3 classes at ≤0.36 similarity → genuinely ambiguous text (margin 0.047 → flagged uncertain) |
| 2 | "Our return journey from Jaipur is on Wednesday." (×3: Kochi/Sunday, Jaipur/Friday) | Travel | Shopping | `return` is a strong Shopping feature (+0.32, from "Return the damaged backpack/keyboard…"); `journey` is unseen. Only Travel neighbour is 0.27 similar |
| 3 | "Put snacks and cleaning items in the cart." | Shopping | Study | Every token/bigram is unseen in training (`put`, `snacks`, `cart`…): the feature vector is all zeros, so the prediction collapses to soft class priors |
| 4 | "Check the credit card balance on Sunday." | Money | Travel | `credit card` unseen (nearest neighbours all Money at 0.29 similarity); `check` (+0.15 Travel vs +0.10 Money, from "Check the train departure time…") and `sunday` (+0.11 Travel) outweigh `balance` (+0.15 Money); margin only 0.018 |
| 5 | "Need to exchange the size of a purchased item." | Shopping | Study | `exchange`, `purchased`, `item` unseen; only `size` survives (+0.20 Shopping from "Choose a replacement size…" but with −0.08 on Study) |

**Evidence (mistake #2):** `python error_evidence.py` prints the law of the
prediction from the trained model itself:

- top contribution to the predicted class: `+0.32 * 'return'` (Shopping),
- printed nearest training examples: "Return the damaged backpack to the
  store." (Shopping, 0.19), "Pack my luggage for the Jaipur trip." (Travel, 0.27),
- unseen-in-training tokens: `journey`, `return journey`, `journey jaipur`.

This shows a systematic weakness of lexical features: *return* means "go
back" here but the training data only used it for product returns.

**A useful honest note:** all five distinct mistakes were already flagged
`uncertain` by the margin rule, i.e. the app would not have presented any of
them as confident labels.

## Search evaluation

- Query file and command: `data/search_queries.json` / `python eval_search.py`
- 16 queries over the 12 starter notes; **6 share no words** with their target
  note (requirement: ≥10 queries, ≥3 no-overlap)

| Search approach | Hit rate @3 |
| --- | ---: |
| Keyword-overlap baseline | 9/16 = 0.5625 |
| TF-IDF cosine similarity (same features as classifier) | 9/16 = 0.5625 |
| **Sentence embeddings (all-MiniLM-L6-v2), used in the app** | **15/16 = 0.9375** |

- Which approach the app uses and why: **embeddings**, with automatic
  fallback to the TF-IDF version if `sentence-transformers` or the model
  download is unavailable. The bag-of-words approaches tie on keywords but
  both score **0** on all six no-word-overlap queries — for example
  `money transfer` has zero token/bigram overlap with "Move ₹2,000 into the
  savings account…" — which the embedding model matches semantically
  (0.37 top hit, in the top 3).
- A query that still fails: `top up my balance` (expected N006). The note is
  ranked but with all-scores ≤ 0.20 (banking phrasing "move ₹ into the
  savings account" is not what the embedding model considers a paraphrase of
  "top up"), and everything is below the 0.30 relevance cutoff, so the app
  explicitly answers "No good match found" instead of pretending. Fixing it
  would require either training-domain embeddings (not available offline) or
  hand-picking a synonym lexicon — which would overfit my own test set, so I
  left it failing and documented it.
- Anti-hallucination cutoff: gibberish queries return no result marked
  `relevant` (`xyzzy plugh`, `quantum velcro flood` → max similarity ≈0.11–0.18
  < cutoff 0.30).

## Notes that fit no category

- How `uncertain` is decided: purely from model output — flag when
  `P(top) − P(2nd class) < cutoff` with cutoff **0.08**. No keyword list is
  involved anywhere.
- Did I train on `out_of_scope_examples.csv`? **No.** They are used only to
  evaluate the flag; the classifier was fitted on `train.csv` only.

| Setting (rule, cutoff) | Out-of-scope flagged | Validation flagged |
| --- | ---: | ---: |
| maxprob < 0.35 (alternative) | 0.67 | 0.26 |
| **margin < 0.08 (chosen)** | **0.75** | **0.20** |
| margin < 0.12 (alternative) | 0.83 | 0.30 |

- Why this trade-off: the margin signal at 0.08 catches 75% of out-of-scope
  notes while flagging 20% of in-scope validation notes. Loosening to 0.12
  buys +8% out-of-scope recall but flags nearly a third of normal notes —
  the flag then loses meaning for users. Tightening to the maxprob rule
  misses more out-of-scope notes. The signal is intentionally conservative:
  an uncertain note still shows its best-guess category, just with a
  visible "Uncertain" badge.
- Assessed honestly: with only 156 training examples and a classifier
  regularised this heavily, probabilities are compressed (most validation
  max-probabilities sit in 0.3–0.6), so probability-based uncertainty is
  inherently coarse here. 20% of validation flagged is higher than I would
  like; margin beat the max-probability rule in every comparable setting.

## What I tried to fix, and whether it worked

The initial model (word 1–2 grams, no stopword handling) scored **0.64
accuracy / 0.666 macro-F1** and the diagnostic was unambiguous: the top
"Travel" features were *stop-words and function phrases* (`'to'`, `'for'`,
`'our'`, `'time for'`), so anything phrased with similar filler words got a
Travel bias.

1. **Fix: remove generic stopwords.** Chosen by 5-fold CV on training data
   only (`python experiments.py`): CV macro-F1 0.947 → 0.966. Validation
   accuracy rose 0.64 → 0.80.
2. **Bug found while checking feature evidence:** sklearn's built-in
   'english' stop list (Glasgow IR list) also deletes *task-relevant* words —
   `bill`, `due`, `call`, `put` — silently erasing the core Money/Travel
   signal ("The water bill payment…" vectorised without `bill` at all).
3. **Refined fix: custom stopword list** = sklearn's list minus those domain
   words (`app/textnorm.py`). Again decided by training-only CV: 0.966 →
   **0.973 CV macro-F1**. Final validation: **0.860 accuracy / 0.859 macro-F1**.

Everything was measured without fitting anything on validation labels; OOS
notes were never trained on either.

## What I would improve next

- More (and more varied) training data — especially paraphrases of Travel
  "reach/journey" and Shopping "exchange" language, which dominate the
  residual errors.
- Calibrated probabilities (e.g. temperature scaling fitted on training-only
  CV) to make the uncertainty cut-off more forgiving of in-scope notes.
- A small local cross-encoder or a domain-tuned embedding model to close the
  last search gap ("top up my balance").

## Limitations

- The dataset is synthetic and tiny (156 training rows). 5-fold CV macro-F1
  reaches 0.973 on training data, but validation (phrasings not in training)
  is 0.859 — a realistic proxy for how much I **don't** trust this score on
  unseen phrasings. Notes that differ from training templates in wording
  (not just dates/cities) are exactly where the model fails, as the error
  analysis shows.
- The uncertainty threshold is a modest safeguard, not an OOD detector: with
  heavy L2 regularisation the probabilities are compressed, and genuinely
  confident-yet-wrong predictions are possible (1 of 16 search-relevant
  semantics, and false-confident val mistakes exist outside the flagged set).
- Search hit-rate is measured on a hand-written 16-query set and the embedding
  model general-isation was not verified on out-of-domain corpora.
- The starter-note search evaluation uses the 12-note corpus only; scores
  will shift on a larger collection (relative re-ranking should be similar).
