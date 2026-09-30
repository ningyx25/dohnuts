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

## Android Control conversion

`scripts/prepare_android_control_data.py` converts
[Android Control](https://console.cloud.google.com/storage/browser/gresearch/android_control)
episodes into decision rows whose prompt is the one android_world's
`ClientMobileJev` builds for the same screen and goal; the design, the parity
test and the measured coverage are in
`docs/superpowers/specs/2026-09-30-android-control-mobile-jev-alignment-design.md`.
An episode is one directory: `metadata_{episode_id}.json` lists its steps, and
every step points at a screenshot plus a `step_NNN_a11y.json` accessibility
forest. Episodes are discovered by an all-digit directory name in numeric order,
so anything else under `--input` is ignored. The converter reuses the gui-v1 split
seed, isolation and self-check. The parsed corpus holds 15,283 episodes and
99,131 steps, 83,848 of them with an action; output goes under the gitignored
`data/processed/` (the `ac-jev-v1` convention).

Rows are one question each, and the questions are the agent's own:

| Question | Dataset | Type | Candidates | Target | Rows |
| --- | --- | --- | --- | --- | --- |
| `operation` | `jev_operation` | choice | the operations the screen offers, in the agent's fixed order | the step's operation | every step |
| `tap_target` | `jev_tap_target` | choice | the TAP candidates, as `[i] label` | the element under the action's point | click, long_press |
| `text_value` | `jev_text_value` | choice | the goal's 1..8 word spans, plus `NONE` | the typed span | input_text |
| `app_target` | `jev_app_target` | choice | the offered app inventory | the app the corpus opened | open_app |

`scroll_target` is deliberately absent: Android Control records a scroll
direction but no coordinates, so which region moved is unknowable. The state and
the questions are produced by `dohnuts.mobile_jev_prompt`, a verbatim port of the
agent's policy, so a row's prompt is what the agent would have posted —
including the 771-character `RULES`, the `{'goal', 'rules'}` instruction object,
the nine state keys and the shared 1-based element numbering (TAP targets first,
then scroll-only regions).

Row ids are `android_control_{episode}_step{n}:{family}`, with `n` the step's
0-based position in the episode, and the group is
`task:android_control_{episode}`, so one episode never straddles splits. Buckets
are the shared `int(sha256("doh-gui-split-2026:" + group)[:8], 16) % 100`
(calibration < 10, dev < 20, test < 30, train otherwise), and the shared split
priority `train < calibration < dev < test` is resolved per record on screenshot
bytes: a tap_target step's rows carry both `image-bytes:` aliases (the raw frame
and its mark), and byte-identical screenshots never straddle splits.

**Operation mapping.** Android Control actions map onto the agent's operations:

| Android Control action | Operation | Target row |
| --- | --- | --- |
| click(x, y), long_press(x, y) | TAP | tap_target |
| scroll(direction) | SCROLL_{DIRECTION} | — |
| input_text(text) | TYPE_TEXT | text_value |
| open_app(app_name) | OPEN_APP | app_target |
| navigate_back, navigate_home | BACK, HOME | — |
| wait | WAIT | — |
| null action (the last step) | DONE | — |

- The scroll mapping is the **identity**, not the inversion gui-v1 uses: Android
  Control's `direction` is content-ward — `scroll down` reveals content below the
  fold, so the finger moves up — and the agent's `SCROLL_DOWN` is that same
  gesture. `SCROLL_TO_SWIPE_DIRECTION` still flips AC onto gui-v1's finger
  vocabulary inside `reference.tool_call`, and the two mappings must not be
  confused.
- A step is excluded whole when the screen does not offer its operation
  (`operation_not_offered`), so the operation row is always answerable.
- The last step of every episode carries a null action and becomes DONE. Episodes
  are treated as successful demonstrations; the parsed corpus no longer carries
  the source `goal_status` field that would confirm it.
- `reference` also carries `ac_action` (the raw source action, or null on the
  terminal step) and `element_positions` (every TAP candidate the action point
  touched, ascending, or `[]` outside that family); neither enters model input.

**Element questions.** TAP candidates are the visible nodes with a usable box
that are enabled and either clickable or editable, in window order then node
order, with the agent's shared numbering: tap targets are numbered `1..N` first
and scroll-only regions continue after them, so a node with no interactive flag
never consumes a number and nothing is deduplicated. The ground truth is a
distribution over the candidates whose box contains the action's point: every hit
weighs `1 / box area`, normalized over the hits, so the smaller — more specific —
box of a nested pair carries the larger share and a single hit is a one-hot. The
row is therefore a soft label, and the training loss already accepts one. A point
that hits no candidate drops the row (`no_target_element`) instead of guessing,
and `reference.element_positions` lists every hit. Fewer than 2 or more than 128
candidates also drop the row (`too_few_candidates`, `too_many_candidates`) — the
agent allows 255 options, a Dohnuts row does not.

**Text and app questions.** `text_value` offers the goal's 1..8 word n-grams plus
`NONE`, exactly what the agent offers, and a typed value that is not one of them
drops the row (`text_not_a_goal_span`); the corpus's typed text is a goal span
only about half the time. `app_target` offers the app inventory the first pass
collected from the corpus's own `open_app` names (the device's installed apps are
not recorded), narrowed per goal by the agent's whole-word rule and capped at
200; a narrowing that leaves one option drops the row.

**State.** `goal` is the episode goal verbatim; `app` is the foreground package
inferred from the accessibility windows (the largest `TYPE_APPLICATION` window,
keyboard excluded); `visibleText` is every kept element's text and description;
`elements` is the agent's entry list (`index`, `label`, `editable`, `scrollable`,
`operations`, plus `checked`/`selected` where they apply); `availableApps` mirrors
the app question; `recentActions` is the last eight executed decisions with the
agent's own labels and a `screenChanged` flag computed by comparing consecutive
screens' fingerprints. `task_progress` is gone: the agent's state has no such
key.

**Token budget.** The conversion no longer checks one. `MAX_LENGTH = 2048` was
defined before the aligned prompts existed, and a row is now measured rather than
excluded: `scripts/report_token_lengths.py` renders every row through the same
prompt template the collator uses and writes `token_lengths.jsonl` (one line per
row) plus `token_stats.json` (min/p50/p90/p99/max and `over_max_length` per
dataset and split). The budget is redefined from those numbers; until it is,
`DecisionCollator` still refuses an over-`MAX_LENGTH` batch.

**Exclusions and audit.** The parse stage drops whole steps
(`unparsable_metadata`, `missing_image`, `missing_a11y`, `unknown_action`,
`payload_too_large`, `operation_not_offered`, `unexpected`), and drops single rows
for their own question (`no_target_element`, `too_few_candidates`,
`too_many_candidates`, `text_not_a_goal_span`, `app_not_offered`). The isolate
stage drops rows (`cross_split_group`, `duplicate_input`). All of them land in
`excluded.jsonl` with `{id, reason, detail, stage}`, where a dropped family
carries `stage: "family"` and the row id it would have had, and in the manifest's
`exclusions` counts. `unparsable_metadata` also covers an episode whose directory
id and record id disagree, and its `detail` names the first field or step that
fails the schema (`step 1: missing key 'action'`). `missing_image` is decided by
a full decode of the screenshot, not by its container: a PNG whose CRCs are
consistent over a stream the decoder rejects fails on its own step instead of
surfacing at the end-of-run self-check, where it used to abort the whole batch.

**Manifest.** `family_stats` is a per-split, per-dataset map of row counts,
candidate min/mean/max, `soft_targets` (rows with more than one positive weight —
only the element family can have them) and `max_positive_weights`.
`family_coverage` is corpus-wide and names its `basis`: `steps_asking` counts the
parsed steps whose operation belongs to that question, `rows` counts what the run
minted before isolation, and `dropped` attributes each reason to the family that
asked. `app_inventory` records the inventory size, its 200 cap and where it came
from. `vocabularies` carries the ported prompt itself — `rules`,
`operation_descriptions`, `target_question_template`, `text_value_none`,
`text_value_instructions`, `state_keys`, `question_ids`, `limits` — plus
`prompt_rule`, `target_rule`, `marked_images` and `deviations`, so a consumer can
read what a row means without the source. `source.metadata_sha256` hashes only the
readable metadata files and `metadata_files_hashed` says how many those were.
`images` lists the PNGs this run wrote under `<output>/images/` (raw and marked,
content-addressed). The manifest goes to stdout; the inventory line, the
empty-split warning and the periodic progress lines go to stderr.

**Marked screenshots.** A tap_target row ships a set-of-mark rendering: every TAP
candidate is boxed in green and labelled with the criteria key it is offered
under, nothing else is drawn, and the ground truth is not highlighted. The label
size scales with the image height, and a candidate with an unusable box is
skipped rather than renumbering the rest. The PNG drops the source image's
metadata, so its bytes are a pure function of the pixels, the candidate list and
the Pillow version — which is why each manifest records `environment.python` and
`environment.pillow`.

**Metrics.** The four `jev_*` datasets are in `metrics.py`'s macro-F1 suppression
set: the operation question offers a per-screen subset of a fixed vocabulary and
the target questions have per-screen candidate lists, so index macro-F1 would
compare unrelated labels across rows. Per-row accuracy, `macro_accuracy` and
checkpoint selection still include them.

**Separate output directory.** AC rows are written to their own directory
(`data/processed/ac-jev-v1`) rather than into `gui-v1`, because the two pipelines
share no question and no state shape.

**Known limits.** The terminal-step mapping assumes every episode is a successful
demonstration. The foreground package is inferred, not recorded. The ground-truth
element is inferred from the recorded point rather than given, so a point that
lands in several boxes is answered with a distribution over all of them — a soft
label the loss accepts, and the honest answer when two nested boxes are both
plausible. A history entry's scroll region is the lowest-indexed scroll candidate
of that screen. Marked bytes depend on the Pillow version. Rows carry screenshots
even though the agent is text-only, and a row allows 2..128 options where the
agent allows 255, which costs roughly half the `app_target` rows and a third of
the `text_value` rows on a 60-episode smoke run (the full-run numbers are in the
manifest). As in the GUI converter, symbolic links inside the input root can
still resolve outside it.

```bash
# Run from the repository root. --input is the directory of {episode_id} episode
# directories. --workers converts that many episodes at a time (forked
# processes); results are merged in episode order, so the splits are byte
# identical to a serial run.
python scripts/prepare_android_control_data.py \
  --input example-data/android_control_parsered/parsered \
  --output data/processed/ac-jev-v1 --workers 32

# Measure the token length of every row afterwards; nothing was excluded for it.
python scripts/report_token_lengths.py \
  --input data/processed/ac-jev-v1 --model Qwen/Qwen3.5-4B-Base --workers 32
```
