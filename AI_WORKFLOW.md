# AI_WORKFLOW

## 1. Tools

Built with OpenCode running GLM (agent coding sessions), used throughout the
72-hour window: scaffolding the FastAPI/sklearn code, writing the evaluation
scripts, and drafting first passes of `RESULTS.md` and this document. All
outputs were verified by actually running every command (`python train.py`,
`python experiments.py`, `python eval_search.py`, `pytest`,
`examples/test_api.py`) and by reading contradictory results sceptically
rather than trusting generated code.

## 2. One delegated task

Prompt given to the agent (abbreviated; typed as a single message):

> "Look at train.py output — the TF-IDF+LogReg model (acc 0.64) scores worse
> than my keyword rules (0.68) and predicts Travel for almost everything.
> Print the top-weighted features per class from the fitted pipeline on
> models/classifier.joblib and diagnose why Travel dominates."

What came back was the diagnosis that stopword n-grams (`to`, `for`, `our`,
`time for`) carried the Travel boundary, plus a proposed experiment script
comparing stopword configurations by cross-validation on training data.
I kept the CV-by-training-data-only approach and the experiment file
(`experiments.py`), changed which configuration got shipped after seeing the
CV numbers, and rejected a suggestion in the same exchange to also tune
`min_df`/`C` (it did not help CV: 0.946 vs 0.966).

## 3. One problem caught (and how it was verified)

**Prediction-to-label mapping bug in `app/classifier.py`.** The API smoke test
(`examples/test_api.py`) passed every format check, but reading the actual
responses showed `"Check my bank balance."` classified as **Study** and
`"Book a train to Pune."` as **Shopping** — visibly wrong for notes of this
shape. Root cause: `predict_proba` columns follow `LogisticRegression.classes_`
(sorted alphabetically: Money, Shopping, Study, Travel), while the code mapped
the argmax index into my task-order label list (`Study, Money, Travel,
Shopping`) — silently scrambling predictions. The same bug existed in
`error_evidence.py` (it printed contradictory wrong predictions for the same
notes as `train.py`, which is what first drew attention).

Verification of the fix:

- re-ran `examples/test_api.py`: sensible categories (`Money` / `Travel`),
  same latency, still all format checks passing;
- added a regression test `test_classify_semantics_on_familiar_phrases`
  asserting four familiar phrases classify to their true categories;
- full suite re-run: 25/25 pytest tests pass.

## Other bugs caught while verifying AI-suggested code

- **Inverted threshold sweep:** the first uncertainty sweep printed
  "flagged" as `P(max proba) >= threshold`, i.e. it reported the *confident*
  share and then maximised it — backwards. Caught by checking that the
  "OOS flagged 1.00 at threshold 0.90" row made no sense. Rewritten as
  "uncertain when confidence/margin < cutoff" and re-selected the operating
  point (now margin < 0.08).
- **scikit-learn's built-in English stop list deletes domain words:**
  while tracing empty feature contributions, discovered `bill`, `due`,
  `call`, `put` are inside `ENGLISH_STOP_WORDS` — the vectoriser was removing
  `"bill"` from "The water bill payment…". Verified by tokenising a training
  sentence directly (`transform` output had no `bill`, `due`) and by the CV
  numbers after building a custom list (0.966 → 0.973).
- **Windows console crashed (cp1252 `UnicodeEncodeError`) printing the `₹`
  character** from evaluation scripts; fixed by reconfiguring stdout to
  UTF-8 in the CLI scripts.
- **pytest module-name collision:** `tests/test_api.py` collided with the
  provided `examples/test_api.py` during test collection; renamed to
  `tests/test_classifier_api.py`.

## Optional: time spent

Roughly 8 hours: ~2 h model/diagnostics, ~2 h app + tests, ~2 h search and
uncertainty evaluation, ~2 h docs/re-verification on a clean run.
