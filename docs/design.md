# Model architecture

```{toctree}
:hidden:

RLCD <rlcd>
Compute efficiency <compute-efficiency>
Data and evaluation <data-and-evaluation>
Reference protocols <upstream-alignment>
Benchmarking <local-benchmarks>
```

Dohnuts accepts a state and independent questions and returns temperature-scaled
probabilities over supplied candidates. The training and serving templates are
shared. A request reuses its causal prefix and computes question suffixes in
parallel. Frozen image features are encoded on a cache miss and shared. More
questions still require more suffix compute.

Each sequence contains the state, question, candidate descriptions, and a
reserved marker after each candidate. Two projections score every candidate
against a separate decision vector: the marker hidden state through
`W_candidate`, the token that ends the question stem through `W_decision`, and
`dot(W_candidate h_candidate, W_decision h_decision) / sqrt(P)` is the logit.
The parameter count is independent of the candidate count, and the readout stays
in FP32 under a BF16 backbone. The decision vector sits before the first
candidate, so it never reads a candidate description. A per-question softmax
gives the decision distribution. Nominal candidate order is shuffled during
training; binary and ordinal order are fixed. Candidate-order sensitivity is
measured on held-out data.

## Fixed recipe

Training opens parameters in four stages. `warmup` trains the decision head
alone; `text` adds rank-8 language LoRA over every language `nn.Linear`; `joint`
also opens the vision merger; `vision_top` also opens the top four vision
blocks. While the vision tower is open its features are recomputed with
gradients rather than read from the frozen-image cache. Each open group has its
own learning rate: head, LoRA, merger, and vision.

The training recipe fixes precision, image and sequence budgets, optimizer,
learning rates, sampling, checkpoint selection, and calibration. `recipe.py` is
the executable specification. Run metadata records all values and data hashes,
including values that callers cannot change.

## Data-parallel training

One machine, several GPUs, one process per GPU. Launched under
`torchrun --nproc_per_node=N`, each rank replicates the model on its own device
and `DistributedDataParallel` averages gradients across ranks. A step's mixture
is `accumulation * N` microbatches; rank `r` reads the contiguous slice
`[step*A*N + r*A, step*A*N + (r+1)*A)`, so the global batch scales with the GPU
count while `--steps` keeps meaning optimizer updates. The world size is recorded
in the run config, so a resume must reproduce it.

Gradient accumulation synchronizes once per update on the last microbatch: DDP's
`no_sync` avoids a partial sum being mixed into the reduced gradient. Stages that
open the vision tower declare `find_unused_parameters`, because a microbatch
without an image legitimately never touches the merger or vision blocks; head and
language parameters always do, so `warmup`/`text` keep the faster static path.

Rank 0 alone reads telemetry, evaluates development accuracy, and writes
checkpoints and metrics; every rank makes the same collective calls in the same
order, and metrics and consumed-sample counters are reduced before rank 0 logs
them. A plain `python` launch takes the same single-GPU path as before, with no
process group, so nothing changes for one device. Final calibration and
evaluation remain single-process.

`method` selects one of three objectives over the same candidate logits. `rlcd`
(the default) is the Laya-aligned estimator, where
`RLCDConfig(sigma=0.3, ce_weight=1.0)` is the complete configuration surface:
`sigma` controls exploration in logit space and `ce_weight` weights the joint
cross-entropy term. `sft` is distribution cross-entropy with optional RPS and
Brier terms, both zero by default. `grpo` samples candidates per question,
normalizes rewards within the sampled group, and combines a clipped ratio, an
exact categorical KL to a fixed reference, and a Brier term. See
[RLCD and the other objectives](rlcd.md). `--steps` controls the total update
budget.

The release path merges LoRA before calibration and final evaluation, and a
`warmup` run has no LoRA to merge. Loading the compact checkpoint reconstructs
that same merged model from the pinned base. The caller does not choose merge
state, precision, attention backend, or calibration mode.

## Model adapters

`Qwen35Adapter` contains the backbone-specific code. Another backbone supplies
an adapter object to `DecisionModel`, `DecisionCollator`, `train`, and
`final_evaluation`; `Predictor.from_checkpoint(..., adapter=adapter)` validates
its identity against the saved artifact.

| Adapter responsibility | Contract |
| --- | --- |
| `name`, `base_model`, `marker`, `image_prefix` | Stable identity, one reserved candidate token, image prompt syntax |
| `processor`, `load`, `hidden_size` | Load the pinned processor/backbone and size the two projections |
| `apply_stage`, `merge`, `freeze_vision` | Apply the training and deployment lifecycle |
| `batch_inputs`, `shared_image_inputs` | Preserve all candidates and process shared images once |
| `forward` | Return candidate-addressable hidden states and the shared-prefix position offset |

There is one supplied adapter. Supporting a different backbone requires its
implementation and the same checkpoint/quality acceptance; an arbitrary model
path alone does not establish compatibility. Model-specific tokenization and
kernels belong here, without branching the RLCD objective or public decision API.

The [inference reference](inference.md) describes question types, response fields,
and input limits. [Bub integration](bub-agent.md) exposes the same decision API to agents.

## Behavior and regression tests

`tests/acceptance.py` loads the exported checkpoint and exercises the real
prediction and Bub interfaces. It checks text and image decisions, candidate
probabilities, independent requests, reload, and input limits. The training
workflow runs this acceptance on its exported weights.

The CPU tests in `tests/test_*.py` cover reported metric meanings, temperature
calibration, the pinned RLCD objective and gradients, and the regression where
identical screenshots crossed splits under different source IDs. They also pin
the two-projection head numerics, the decision readout position, the
stage-owned parameter sets, and the SFT and GRPO closed forms. Run them when
changing the corresponding behavior:

```bash
make test
```

Tests assert observable behavior or a known failure case. Internal allocation,
helper structure, log keys, file counts and exact error wording are not contracts.
Quality evaluation separately measures held-out accuracy, calibration,
candidate-order sensitivity and image dependence.
