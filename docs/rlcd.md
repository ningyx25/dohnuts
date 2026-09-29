# RLCD implementation

The Dohnuts training recipe trains a two-projection candidate scorer and, for
every stage except `warmup`, language LoRA adapters (rank 8, alpha 16); the
Qwen3.5-0.8B vision encoder stays frozen in the `warmup` and `text` stages.
The pinned base is `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`.

The recipe's `method` selects one of three objectives over the same candidate
logits. `rlcd` is the Laya-aligned estimator described in this document and
stays the default. `sft` is a distribution cross-entropy, and `grpo` is
categorical GRPO with a fixed reference; both are described below.
`src/dohnuts/objectives.py` implements the dispatch, while `src/dohnuts/rlcd.py`
and its pinned reference fixtures are unchanged.

The numerical reference is [Laya Vision `vlm_loss`](https://github.com/r33drichards/laya-vision/blob/86ccec115ef3d72d1851168fc9b7194e8dbed35a/laya/vlm_train.py)
and [Laya's typed-decisions training notebook](https://github.com/NandhaKishorM/laya/blob/d113dca2512fb3eaca313534bc54c7162d87c1d4/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

Shared prefixes and parallel question suffixes produce candidate logits without
autoregressive generation. Draw four
Gaussian perturbations with sigma 0.3, center each over valid candidates, and
score their softmax distributions. The reward adds log score (floor −9.21) and
0.75 times spherical score; ordinal questions also subtract RPS divided by K−1.
Subtract the group mean reward for each question, then divide all advantages by
one sample standard deviation across groups and questions, plus 1e−6. Both the
sampled logits and rewards are detached in the policy-gradient calculation.

The training loss is this policy-gradient loss plus soft cross-entropy with
weight 1. This is one joint training path, following Laya; there is no separate
SFT warm-up, reward model, critic, or autoregressive rollout. Policy loss can be
negative and noisy, so cross-entropy, reward, and held-out accuracy are logged
separately. Clipping and this gradient estimator do not prove calibration.

Candidate-selection (`choice`) candidates are shuffled during training with their targets. Binary
false/true order and ordinal level order stay fixed. Soft labels are normalized
within tolerance; VQAv2 yes/no votes and typed-decisions distributions retain
uncertainty rather than being converted to hard labels.

The training recipe is defined in `src/dohnuts/recipe.py`: equal probability per
source/language/task group, deterministic sampling with replacement, at most
6,000 eligible source rows in each group's training pool, batch 8 with four
accumulation steps, a default budget of 3,600 updates, one learning rate per open
group — head 5e−4, LoRA 1e−4, vision merger 1e−5 (`joint`, `vision_top`) and
vision blocks 2e−6 (`vision_top`) — 72 warm-up updates, cosine decay to 10%
through update 2,400, then a fixed 10% floor through update 3,600; weight decay
0.01 and gradient clip 1.
The default run processes 115,200 sampled examples. Each run freezes the eligible
IDs and source hashes; sampling with replacement can revisit an example.

Every 400 updates, evaluate the same deterministic development subset of up to
256 examples per group. Select by unweighted group macro top-1 accuracy; test
labels never select a checkpoint. Save trainable parameters, optimizer, step,
RNG state, recipe, and selected score. Sampling depends on seed and absolute
microbatch index, so restarting does not restart the data stream.
Extending the total budget retains the original decay horizon and uses its final
learning-rate floor for the additional updates; optimizer moments are preserved.

Merge LoRA, then fit three temperatures, one per decision type, on the independent calibration
partition using LBFGS (100 iterations, LR 0.1, temperature range 0.1–10). Final
reports retain both raw and calibrated metrics. Raw predictions include IDs,
target distributions and logits so plots and metrics can be regenerated.

Only `sigma` and `ce_weight` configure `method: rlcd`; `--steps` controls the training budget.
The default values above match
the pinned upstream objective; the remaining estimator and reward settings are fixed.

## SFT

`method: sft` minimizes the soft cross-entropy of the candidate distribution.
Two optional terms share the same forward pass: `sft.rps_weight` weights the
ranked probability score, which only counts the ordinal (`score`) rows the
collator marks, and `sft.brier_weight` weights the sum of squared probability
errors. Both default to 0, so the default objective is the pure distribution
cross-entropy.

## GRPO

`method: grpo` samples `grpo.group_size` candidates per question with
replacement from the current distribution. The reward weights correctness — the
label mass on the sampled candidate, so multi-hit soft labels keep their
uncertainty — against a confidence error, and the advantages are normalized
within each question's sampled group. A group whose rewards are all equal gets
exactly zero advantage; the Brier term still provides a gradient there. The
update combines the clipped ratio, an exact categorical KL to the reference
model when `grpo.beta > 0`, and the Brier term.

`grpo.num_iterations` defaults to 1. With one iteration the rollout and the
update reuse the same forward, so the ratio is exactly one and clipping is
inert: the objective degrades to a policy gradient with KL and Brier. Set it to
2 or more to let the clipped objective act. With more than one iteration the
rollout is sampled once per accumulated group and every iteration performs its
own optimizer update from that frozen rollout, so the policy has moved by the
second iteration and the ratio differs from one. A recipe's `steps` still counts
accumulated groups, which means GRPO performs `steps × grpo.num_iterations`
optimizer updates. The remaining controls are `grpo.correctness_weight`,
`grpo.confidence_weight`, `grpo.brier_weight`, `grpo.beta`, `grpo.clip_epsilon`,
and `grpo.advantage_epsilon`; at least one reward weight must be positive.

`beta > 0` builds one frozen reference: a copy of the policy taken at the start
of the run, roughly 1.7 GB in BF16 for 0.8B. Resume restores that original
snapshot from the checkpoint's `training_state` and refuses a key set that does
not match the current trainable parameters — it never re-derives the reference
from the current weights. Because of the reference and the vision gradients, the
`grpo` method pairs best with the `warmup` and `text` stages.
