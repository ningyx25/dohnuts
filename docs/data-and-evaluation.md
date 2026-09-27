# Data and evaluation

Use public labels and programmatic scoring. The training mixture includes
public teacher-derived soft labels from typed-decisions; no new teacher calls
or LLM judges are required. Deterministic conversion means reproducible examples
and scores, not infallible annotations or known per-example true probabilities.

## Training sources

| Source | Decision tasks | Protocol |
| --- | --- | --- |
| [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive) | English/Chinese `choice` | All 60 intents; fixed human-readable label mapping |
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Text `noul` | Original passage, question, and Boolean answer |
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | Visual `choice`, `noul`, `score` | Attributes, existence, and ordered counts |
| [ScreenQA](https://github.com/google-research-datasets/screen_qa) | Screenshot `choice` and candidate-level `noul` | Original human annotations in `answers_and_bboxes` |
| AG News / emotion / BANKING77 | Text `choice` | Full label vocabularies and official labeled test sets |
| XNLI | English/Chinese `choice` | Three labels; aligned translations and duplicate premises stay grouped |
| A-OKVQA / ScienceQA | Image `choice` | Four-choice A-OKVQA; image-only ScienceQA subset with official test retained |
| VQAv2 yes/no | Image `noul` | Soft answer votes; image-grouped split of labeled official validation |
| typed-decisions | All three decision types | Public soft teacher distributions; state-grouped partitions |

CLEVR scene graphs and programs verify labels but never enter model inputs.
Use complete answer domains per question family rather than answer-dependent
candidate sampling. Count performance does not establish subjective scoring
ability. English visual benchmarks do not establish Chinese visual competence.

### ScreenQA conversion

Build candidate regions from the screenshot's Rico View Hierarchy using a fixed,
answer-independent rule. Present region IDs and geometry consistently; do not
feed annotator descriptions or ground-truth answers into the input.

Retain questions whose annotations agree on one answer element that maps
unambiguously to a candidate. Record exclusion counts for multi-element answers,
disagreement, missing assets, and unmapped regions. Do not inject a target region
to rescue an otherwise invalid example.

- `choice`: select the matching region from the candidate set.
- `noul`: determine whether a specified candidate is the answer region.

A negative candidate is not evidence that the answer is absent from the screen.
Publish the negative-sampling rule and class balance; calibration is specific to
that distribution. Report these as derived decision tasks, not official ScreenQA
question-answering scores. The inference contract requires candidate regions;
automatic UI detection is outside scope.

ScreenQA Short and ComplexQA are excluded from the data mix. The former
contains model-produced answers, and the latter uses model-generated questions
and answers with human validation.

## Splits and reproducibility

- MASSIVE and ScreenQA retain official train/dev/test partitions. Reserve a
  group-based calibration subset from train.
- BoolQ and CLEVR use public labeled dev/val as frozen final evaluation sets,
  not as official test results. Split train into fitting, development, and
  calibration partitions.
- Keep all questions from one image, all language variants of one source
  utterance, and questions sharing a passage in the same partition. Audit
  official splits for overlap and document any exclusions.
- Store source revisions and hashes, sample IDs, split assignments, templates,
  filters, candidate permutations, sampling weights, and seeds in manifests.
- Pin preprocessing and metric implementations. Report cross-split duplicates
  and possible backbone pretraining contamination.

Select checkpoints and hyperparameters using development data only. Fit
temperatures on calibration data only. Freeze the recipe before final evaluation.
Seeded runs improve repeatability but do not guarantee bitwise equivalence across
GPU kernels or software versions.

## Evaluation

Report results by dataset, language, decision type, and candidate count:

- Accuracy and macro-F1 for decision quality.
- NLL and explicitly normalized Brier score for distributions.
- Ranked probability score and MAE for ordered counts.
- Fixed-bin ECE and reliability diagrams for calibration.
- Candidate-order sensitivity and correct-image/missing-image/mismatched-image
  checks for vision dependence; these are quality checks, not competing trainers.
- End-to-end p50/p95 latency, throughput, and peak VRAM on the local GPU, including
  preprocessing and with image, sequence, candidate, and batch sizes disclosed.

Complete calibration, held-out evaluation, and benchmarks for seed 42 before
replication. Report variation only when multiple seeds have completed evaluation;
for one seed, leave sample standard deviation unmeasured. All probabilities
must be finite, in range, and normalized where appropriate. Report filtered
coverage alongside accuracy. Do not equate a hard target with a known conditional
probability, or a concentrated distribution with justified confidence.

## Release assets and terms

Release the recipe, manifests, inference configuration, calibration parameters,
metrics, and model card alongside any weights. Keep raw data and checkpoints out
of Git. Retain source attribution and audit data and image terms before release.

MASSIVE and CLEVR identify CC BY 4.0 terms; BoolQ identifies CC BY-SA 3.0 terms.
ScreenQA annotations identify CC BY 4.0, but Rico screenshot terms must also be
checked. Do not infer a weight license solely from an annotation or code license.
ScienceQA is included in the research mixture. Existing Laya Vision
weights remain benchmark references. See [upstream alignment](upstream-alignment.md)
for source protocols and comparison limits.

## Business decision sources

| Source | Decision supervision | Evidence and limits |
| --- | --- | --- |
| [Amazon ESCI](https://github.com/amazon-science/esci-data) | Query/product `choice`: exact, substitute, complement, irrelevant; English, Spanish and Japanese | Actual customer search queries with manual relevance judgments. Use explicit E/S/C/I labels. The categories are not assumed to be an ordinal scale. Apache-2.0 project release. |
| [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/) | Query/passage answerability `noul` | Real Bing queries, Wikipedia sentences and crowdsourced binary judgments. Explicit negative labels; never equate missing judgments with negatives. Microsoft Research Data License: research/technology development restrictions apply. |
| [ShARC](https://sharc-data.github.io/data.html) | Agent `choice`: yes, no, irrelevant, ask for missing information | Public rule documents with crowdsourced scenarios/conversations. Real policy tasks, not a log of actual customer interactions. Evidence annotations and the target follow-up wording never enter model state. CC BY-SA 3.0. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Full-contract `choice`: entailment, contradiction, not mentioned | Human annotations on real contracts. Keep the entire document; exclude over-budget documents instead of selecting gold evidence spans. CC BY 4.0. |
| [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/readme.html) | Email spam `noul`; known ham supplies negative phishing examples | Public collected messages. Includes hard ham. Sender copyright remains with senders; the software's Apache license is not a blanket data license. This is a research corpus. |
| [Nazario phishing corpus](https://monkey.org/~jose/phishing/README.txt) | Positive phishing `noul`, paired with separately identified ham | Hand-classified personal-inbox messages; 2023 for training/calibration, 2024 development, 2025 test. CC BY 4.0. Positive/negative source and age differ; report cross-corpus and temporal limitations. |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Message spam `noul` | Public collected SMS with spam/ham labels. Adds a short-message domain; it does not substitute for email evaluation. UCI release is CC BY 4.0. |

The mixture includes research-use sources, including ScienceQA. Dataset annotations, raw documents and mixed trained
weights have distinct rights; this document does not certify a commercial
checkpoint license.

## Conversion and isolation

`scripts/prepare_data.py` verifies every raw-file SHA-256 against
`data/manifests/enrichment-downloads.json` before conversion. It creates the
existing `{state, question, target}` schema. One-hot targets come from source
labels. No model generates pseudo-labels, confidence scores or explanations.

Source rules are fixed in the converter:

- ESCI uses the official challenging-query subset. Select 450 training queries
  and at most 60 queries per held-out partition per locale using stable hashes.
  Keep whole queries, with official test priority. Products may occur under
  multiple queries: this is unseen-query evaluation, not unseen-product evidence.
- ShARC groups by source page and snippet. ContractNLI groups by full contract.
  WikiQA joins query and source-document aliases before resolving split conflicts.
- Mail uses decoded bodies with the same plain-text/HTML conversion for both
  classes; labels, corpus names, spam-filter headers and acquisition timestamps
  are absent from state. URL/number-normalized body signatures group simple
  campaign variants. Attachments and links are never executed or fetched.
- Independent calibration comes from training groups. Original dev/test labels
  are never moved into training. Conflicting group partitions keep the higher
  priority partition and exclude the others.
- Frozen Laya/JevBench inputs, the sampled training inputs, and its
  held-out states are checked with punctuation-insensitive text hashes.
  Checks include Laya's 3,000-character body prefixes. Matching source groups are
  excluded. This does not prove absence of semantic or paraphrase duplicates.
- The fixed 2,048-token training budget is unchanged. Exclusions are counted.
  Serving still accepts 4,096 tokens. Results on the eligible ContractNLI subset
  must not be presented as scores on its complete official test set.

The training mixture combines 17 text/image groups and nine business task/language
groups. Original dev/calibration/test rows retain their partitions. The existing uniform group
sampler therefore allocates an expected 9/26 of updates to business tasks and 17/26
to the other text/image tasks. Its fixed 6,000-row training cap per group remains in effect.
No class rebalancing is hidden in calibration or evaluation. Phishing positives
and WikiQA positives are minorities; report recall, macro-F1, PR-AUC and class
counts alongside accuracy, NLL, Brier and ECE.


The fixed 26-group mixture lives in `data/processed/v1`. Its prepared partitions
contain 185,857 training, 64,843 development, 86,897 calibration and 180,031 test
rows. The per-group training cap yields 143,238 eligible training rows.
`manifest.json` records counts, exclusions and hashes. The frozen benchmark text
fingerprints in `data/manifests/evaluation-text-sha256.json` are exclusion inputs,
never training examples. `scripts/prepare_data.py` builds this mixture without
requiring a trained checkpoint or a previous training run.

## GUI step conversion

`scripts/prepare_gui_data.py` converts step-level GUI agent trajectories (user
query, completed-step history, screenshot, and the ground-truth tool call) into
decision rows. It needs no candidate list, element tree, or model output: every
row follows from the tool call alone.

One step yields two or three rows that share state and image, one question per
row:

| Question | Type | Candidates | Target |
| --- | --- | --- | --- |
| `action` | choice | click, long_press, swipe, type, answer, system_button, wait, terminate | the tool call's action |
| `button` | choice | Back, Home, Menu, Enter | the pressed button (system_button steps only) |
| `complete` | noul | false, true | whether the next action terminates the task |
| `swipe_dir` | choice | up, down, left, right | dominant axis of the swipe (swipe steps only) |

Fixed rules:

- `state` keeps the source `user_query` and `task_progress` verbatim; thought and
  action text are kept in `reference` for provenance and never enter model input.
  Records must hold exactly one user and one assistant message; anything else is
  excluded instead of guessed.
- Task ids strip the `_step<N>` suffix and any trailing batch marker after it
  (e.g. `__from0208_...`) and form the isolation group. Split
  buckets are `int(sha256("doh-gui-split-2026:" + group)[:8], 16) % 100`:
  calibration < 10, dev < 20, test < 30, train otherwise. Image bytes join groups
  before the split priority (`train < calibration < dev < test`) is resolved, and
  a merged group keeps the lexicographically smallest task name (aliases never
  become group names). Task groups never straddle splits.
- Dataset names split by question (`gui_action`, `gui_button`, `gui_complete`,
  `gui_swipe`) so macro-F1 stays within one fixed candidate vocabulary. The
  uniform dataset sampler therefore gives each question family roughly equal
  weight, which relatively upweights button and swipe rows.
- Swipes whose axes tie on absolute delta are excluded instead of guessed.
- The token budget is checked when `--model` names a local snapshot (skip with
  `--no-token-check`). The estimate mirrors the training collator: rendered text
  tokens plus the expanded image placeholders at `IMAGE_PIXELS`. Over-budget rows
  are excluded whole, before their screenshot is stored.
- Exclusions are audited, never silent. The parse stage drops whole records
  (`unparsable_state`, `multi_turn`, `missing_tool_call`, `unknown_tool`,
  `missing_id`, `unknown_action`, `invalid_button`, `invalid_swipe`,
  `multi_image`, `missing_image`, `token_budget`, `unexpected`); the isolate
  stage drops rows (`cross_split_group`, `duplicate_input`). Both land in
  `excluded.jsonl` with `{id, reason, detail, stage}` and in the manifest's
  `exclusions` counts (`parse:<reason>` and `<dataset>:<split>:<reason>`).
- Manifest `images` lists the screenshot files this run stored (bare file names
  under `<output>/images/`, content-deduplicated); an image can outlive rows that
  were later dropped by isolation, so it is not the set of images the dataset
  references. Re-running into the same `--output` reproduces the same four
  hashes; a different `--output` legitimately changes them because rows embed the
  stored image path.
- `excluded.jsonl` uses two id conventions: parse-stage entries carry the raw
  record id (which can be `null` when the record has none), isolate-stage entries
  carry the affected row id with its `:action`/`:button`/`:complete`/`:swipe_dir`
  suffix.

Coordinates and typed text are payloads for the orchestrator, not decisions:
this model answers what to do, which button to press, in which direction to
swipe, and whether to stop. Region detection stays outside the converter.

Known limits: only one screenshot per step; symbol links inside the input root
can still resolve outside it; tasks that share a screen with another task are not
detected as near-duplicates.

**Shared screenshots are resolved per record.** Screenshots with identical bytes
may only live in one split: for every screenshot the highest-priority split among
the records carrying it wins (`train < calibration < dev < test`), and the rows
carrying it in lower-priority splits are dropped as `cross_split_group`. The rest
of a task's rows stay in the task's own split, and task groups are never renamed.
A launcher screen, lock screen, or repeated initial state therefore costs only
the rows that literally repeat that screenshot — on a 27k-record trajectory set
this dropped 4,870 of 59,449 rows (8.2%), affecting 1,876 of 27,360 records
(6.9%), where merging whole task groups would have cost about half the records.
Read `manifest["exclusions"]`
(especially `cross_split_group:…`) and `excluded.jsonl` (`stage=isolate`) to see
exactly which rows went and why before trusting the split sizes.

```bash
# Run from the repository root. --input is your own directory of step-record
# *.json arrays; --model points at a local snapshot, or the token check is skipped.
pdm run python scripts/prepare_gui_data.py \
  --input data/raw/gui --output data/processed/gui-v1 --model Qwen/Qwen3.5-0.8B
```
