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
episodes into decision rows; it is the sibling of the GUI converter above. An
episode is one directory: `metadata_{episode_id}.json` lists its steps, and every
step points at a screenshot plus a `step_NNN_a11y.json` accessibility forest.
Episodes are discovered by an all-digit directory name in numeric order, so
anything else under `--input` is ignored. The converter reuses the gui-v1 split
seed, vocabularies, isolation and self-check,
and adds one question family that needs the element tree: which element to act
on. The parsed corpus holds 15,283 episodes and 99,131 steps, 83,848 of them with
an action; output goes under the gitignored `data/processed/` (the `ac-v1`
convention).

Each step yields two or three rows that share state, group and image aliases, one
question per row:

| Question | Dataset | Type | Candidates | Target | Rows |
| --- | --- | --- | --- | --- | --- |
| `action` | `gui_action` | choice | the nine actions (gui-v1's eight plus `open_app` at index 8) | the step's action | every step |
| `complete` | `gui_complete` | noul | false, true | whether the episode ends here | every step |
| `element` | `screenshot_choice` | choice | the step's clickable elements | the element under the action's point | click, long_press |
| `swipe_dir` | `gui_swipe` | choice | up, down, left, right | the finger's direction | scroll |
| `button` | `gui_button` | choice | Back, Home, Menu, Enter | Back or Home | navigate_back, navigate_home |

Row ids are `android_control_{episode}_step{n}:{family}`, with `n` the step's
0-based position in the episode, and the group is
`task:android_control_{episode}`, so one episode never straddles splits. Buckets
are the shared `int(sha256("doh-gui-split-2026:" + group)[:8], 16) % 100`
(calibration < 10, dev < 20, test < 30, train otherwise), and the shared split
priority `train < calibration < dev < test` is resolved per record on screenshot
bytes: a click step's rows carry two `image-bytes:` aliases (the raw screenshot
and the marked copy), the other rows carry the raw one, and byte-identical
screenshots never straddle splits, so a repeated initial screen costs only the
rows that repeat it.

**Action mapping.** Android Control actions map onto the mobile_use vocabulary:

| Android Control action | mobile_use action | Extra row |
| --- | --- | --- |
| click(x, y), long_press(x, y) | click, long_press; the pixel point stays in `reference` | element |
| scroll(direction) | swipe, direction inverted | swipe_dir |
| input_text(text) | type; the text stays in `reference` | — |
| wait | wait | — |
| open_app(app_name) | open_app (index 8) | — |
| navigate_back, navigate_home | system_button (Back, Home) | button |
| null action (the last step) | terminate | complete target is `true` |

- `open_app` is appended after the eight gui-v1 actions as index 8, so indices
  0–7 keep their gui-v1 meaning; its criterion is "open an app by name". The
  vocabulary keeps all nine classes, so `answer` stays a criterion even though no
  Android Control action maps to it.
- Scroll directions are inverted deliberately. Android Control's `direction` is
  content-ward — `scroll down` reveals content below the fold, so the finger
  moves up — while gui-v1 derives `swipe_dir` from the finger's displacement.
  Inverting keeps `up`/`down` meaning the same finger motion in both datasets,
  and lives in one constant. Cross-correlating the screenshots before and after
  138 real vertical scroll steps puts content moving up in 28 coherent cases
  against 10 moving down for `scroll: down`, and the dataset's own step
  instructions agree ("Swipe up for Product details" on `scroll: down` steps,
  "Swipe down" on `scroll: up` steps).
- `reference` also carries `ac_action` (the raw source action, or null on the
  terminal step) and `element_positions` (every candidate the action point
  touched, ascending, or `[]` outside the element family); neither enters model
  input.
- The last step of every episode carries a null action and becomes `terminate`
  with `complete = true`. Episodes are treated as successful demonstrations; the
  parsed corpus no longer carries the source `goal_status` field that would
  confirm it.

**Element questions.** Candidates are the nodes that are clickable and visible
with a non-degenerate box on screen, in window order then node order. They are
numbered contiguously `r0..r{N-1}`: a node that is not clickable, or has an
unusable box, consumes no number, and nothing is deduplicated, so clickable
containers with children stay candidates and the keys cannot be read off the raw
node list. The ground truth is a distribution over the candidates whose box
contains the action's pixel point: every hit weighs `1 / box area`, normalized
over the hits, so the smaller — more specific — box of a nested pair carries the
larger share and a single hit is a one-hot. The element row is a soft label, and
the training loss already accepts one because it only one-hots a length-`n`
target (`rlcd.py`). A point that hits no candidate excludes the step
(`no_target_element`) instead of guessing, and the sizes a step was resolved
from, where they matter, are `reference.element_positions`. Fewer than 2 or more
than 128 candidates also exclude the step
(`too_few_candidates`, `too_many_candidates`). A criterion is
`"UI element {i}: {payload}"`, where the payload is JSON holding only a
non-empty `text` and/or `content_description`, in that key order, and `{}` when
the node has neither; node flags, class names and indices never reach the prompt.
The instruction is "Which action should be taken next to complete the user's
task?", taken verbatim from the hand-built element row in
`example-data/train.jsonl`.

**Reading element metrics.** A soft target changes what the per-row numbers
mean, so the element family is reported with two accuracies. `metrics.summarize`
takes `label = target.argmax()` and `pred = p.argmax()`, so `accuracy` is
"the model picked the **dominant** hit" — the largest-weight candidate, and on
equal weights the earliest one, exactly as `np.argmax` breaks the tie — while
`soft_accuracy` is `target[pred]`, the mass the model's pick carries, so it
credits **any** positive-weight hit and a uniform multi-hit row gives 1/`n` to
even a wrong pick. `macro_f1` is over label indices as before. `nll` and both
`brier` values read the whole distribution rather than the argmax, so they need
no reinterpretation. `reference.element_positions` lists every hit, which is
what a per-row error analysis needs to tell "picked the wrong box" from "picked
the other box the tap point was in".

**Marked screenshots.** The element row ships a set-of-mark rendering: every
candidate is boxed in green and numbered with a white chip in the same order as
the criteria, nothing else is drawn, and the ground truth is not highlighted. The
label size scales with the image height. The PNG drops the source image's
metadata, so its bytes are a pure function of the pixels, the candidate list, and
the Pillow version — content-addressed file names are only reproducible for a
fixed Pillow, which is why each manifest records `environment.python` and
`environment.pillow`. The element row points at the marked copy, every other row
of the step points at the raw copy, and all of them carry both aliases.

**State.** `user_query` is the episode goal verbatim. `task_progress` is a
deterministic template over the completed steps' instructions —
`"(You have done the following operation on the current device): Step 1: …; Step
2: …; ."` — and never the current step's instruction; the numbering closes over
the instructions that exist. The dataset's own narration was human-written and is
not reproducible, so this wording differs from it by design.

**Token budget.** The budget is checked when `--model` names a local snapshot
(skip with `--no-token-check`), before anything is written. The estimate mirrors
the training collator: rendered text tokens plus the expanded image placeholders
at `IMAGE_PIXELS`, against `MAX_LENGTH = 2048`. A step over budget is excluded
whole as `token_budget`, with the longest of its rows in `detail`, so a step is
never partially converted. The check is load-bearing: `DecisionCollator` raises
on an over-budget batch instead of truncating. Measured on the parsed corpus, 520
of 49,924 element rows (1.04%) exceed the budget; the worst is 38,642 tokens
(18.9 times the limit) because one accessibility node's `text` was an entire PDF.
The median row is 624 tokens and p99 is 2,070. Probing the whole corpus (99,131
steps) costs about six minutes.

**Exclusions and audit.** The parse stage drops whole steps
(`unparsable_metadata`, `missing_image`, `missing_a11y`, `unknown_action`,
`too_few_candidates`, `too_many_candidates`, `no_target_element`, `token_budget`,
`unexpected`); `unparsable_metadata` also covers an episode whose directory id
and record id disagree, and its `detail` names the first field or step that fails
the schema (`step 1: missing key 'action'`). The isolate stage drops rows
(`cross_split_group`, `duplicate_input`). Both land in `excluded.jsonl` with
`{id, reason, detail, stage}` and in the manifest's `exclusions` counts.
Parse-stage entries carry the step id or the episode directory name; isolate-stage
entries carry the row id with its family suffix. `missing_image` is decided by a
full decode of the screenshot, not by its container: a PNG whose CRCs are
consistent over a stream the decoder rejects fails on its own step, where it
costs one step, instead of surfacing at the end-of-run self-check, where it used
to abort the whole batch and write no splits at all. The self-check that closes
a run walks containers rather than pixels (`gui_data.validate_rows` calls
`Image.open` and `verify`, a real chunk-and-CRC walk for the PNGs stored here),
which is sound only because of that parse-time decode: what the self-check sees
is either a byte-for-byte copy of a source that already decoded, re-copied
whenever its digest stops matching its name, or bytes this process encoded from
an image it had just decoded.

**Manifest.** `element_stats` and `element_resolution` both describe the element
family and both carry a `basis` field, but they answer different questions and
have different shapes. `element_stats` (`basis: post_isolation`) is a per-split
map of the element rows that were actually written, with candidate min/mean/max,
`soft_targets` (element rows whose target has more than one positive weight),
`multi_hit_rate` (`soft_targets / rows`), `max_hits` (the largest hit count any
one row of the split carries), and `empty_target_payload_rate` (rows where **no**
hit candidate carries a payload, so the prompt names no correct answer at all);
splits without element rows are absent from it.
`element_resolution` (`basis: pre_isolation`) is a single corpus-wide record of
the steps whose element question resolved within budget before isolation, with
`element_rows`, the three refusal counts and `hit_rate` — a step that resolved
but was then dropped by the token budget counts in neither `element_rows` nor the
refusals.

`source.metadata_sha256` hashes only the readable metadata files, and
`metadata_files_hashed` says how many those were. `images` lists the PNG files
this run wrote under `<output>/images/` (raw and marked, content-addressed). Rows
are minted before any file is written, so a step that fails before its stores
writes nothing and the list never names an unreferenced file; a failure between
the raw and the marked store leaves the raw PNG on disk *and* in `images` while
its rows are dropped, and a rerun with a different `--input` into the same
`--output` never cleans the directory, so the list describes this run only. What
a rerun does clean is the temporary files of an interrupted run — `convert`
unlinks `.<name>.<pid>.<n>.tmp` under `images/` before it starts, the only files
a SIGKILL can strand there — so the directory-equals-manifest property holds
again for an operator re-running into the same `--output`.
`vocabularies` carries the nine actions, buttons, swipe directions,
instructions, complete criteria, and two plain-language entries — `element_rule`
and `marked_images` — that state the candidate and marking rules without the
source. `dataset_weighting` records that the five dataset names are drawn
uniformly, so `gui_button` and `gui_swipe` rows, which exist only on the steps
that press a button or scroll, are relatively upweighted. The manifest goes to
stdout; the empty-split warning and the periodic progress lines go to stderr.

**Metrics.** `screenshot_choice` is in `metrics.py`'s macro-F1 suppression set,
because its candidate labels are per-row (every row's `r0..r{N-1}` comes from its
own screen) and macro-F1 over cross-row label indices would be meaningless.
Per-row accuracy, `macro_accuracy` and checkpoint selection still include it.

**Separate output directory.** AC rows are written to their own directory
(`data/processed/ac-v1`) rather than into `gui-v1`, because `gui_action` here has
nine classes where gui-v1 has eight. Under one dataset name the by-dataset
aggregates would mix two label semantics, and both the temperature fit and the
collator pad to the widest target of a batch or decision type, so the two
vocabularies stay in separate trees. Indices 0–7 still mean the same action in
both pipelines.

**Known limits.** The terminal-step mapping assumes every episode is a successful
demonstration. `task_progress` is mechanically templated. Ground-truth element
payloads are frequently `{}` because many clickable nodes are unlabeled
containers: an independent corpus-wide probe found 21,507 of 49,924 resolvable
targets unlabeled (43.1%), a 150-episode sample put the per-split rate at 44–49%,
and small samples vary widely (60% over the 21-episode run). Those three numbers
were measured under the pre-soft-target definition — one target per row, the
single hit the old smallest-area rule chose — and a row is now counted as an
empty target only when **no** candidate the tap point touched carries a payload,
so the post-change rate is expected to be equal or slightly lower;
`element_stats.empty_target_payload_rate` is the authoritative per-run number and
`soft_targets`/`multi_hit_rate`/`max_hits` say how much of that run's family was
soft. The ground-truth element is inferred from the
recorded point rather than given, so a point that lands in several boxes is
answered with a distribution over all of them, weighted by inverse area — a
soft label the loss accepts, which is also the honest answer when two nested
boxes are both plausible targets. Marked bytes depend on
the Pillow version. The label numbering differs from the hand-built smoke row in
`example-data/train.jsonl`, which numbers dataset indices with gaps instead of
contiguous clickable-only candidates. As in the GUI converter, symbol links
inside the input root can still resolve outside it.

```bash
# Run from the repository root. --input is the directory of {episode_id} episode
# directories; --model points at a local snapshot, or the token check is skipped.
# --workers converts that many episodes at a time (forked processes); results
# are merged in episode order, so the splits are byte-identical to a serial run.
pdm run python scripts/prepare_android_control_data.py \
  --input example-data/android_control_parsered/parsered \
  --output data/processed/ac-v1 --model Qwen/Qwen3.5-0.8B --workers 8
```
