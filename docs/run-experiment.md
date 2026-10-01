# Train and evaluate Dohnuts

> **NVIDIA branch.** This branch replaces the recorded ROCm install with
> PyTorch 2.9.1 + CUDA 12.8. Recipe, data, and model code paths are unchanged;
> `data/manifests/environment-freeze.txt` and the model card describe the
> original RX 7900 XTX run.

Use Python 3.12 and a working PyTorch CUDA installation. The recorded training
environment is preserved in `data/manifests/environment-freeze.txt`. Follow the
[PDM setup](inference.md) first; the package sources and lock file select the
CUDA wheels.

```bash
pdm install --check --prod -G train -G agent
pdm run python scripts/run_experiment.py
```

The Qwen3.5 adapter limits PyTorch's caching allocator to 80% of visible VRAM,
leaving headroom for the desktop and GPU driver on a shared workstation. That
bound applies to single-GPU runs only; with data parallelism each process owns
its device outright.

The script downloads the pinned base and public datasets when absent, converts
the complete mixture, isolates related examples and identical images across
splits, and freezes token eligibility. It trains seed 42 for 3,600 optimizer updates,
resuming saved optimizer and sampling state after interruption.

Development accuracy selects the checkpoint within this run. LoRA is merged,
three temperatures are fitted on the calibration partition, and the complete
held-out partition is evaluated. Test scores never choose a model. The workflow
exports the selected checkpoint directly after calibration and evaluation, then
completes acceptance, quality diagnostics and benchmarks. Variation across seeds
is unmeasured.

The script then runs checkpoint/Bub acceptance, quality diagnostics, local
Laya/Laya Vision references, JevBench's frozen public tasks, and the complete
frozen Laya chart protocol. API benchmarks measure latency and resource use.
Every failed stage stops execution.
Completed stages are recorded; rerunning the command resumes the same recipe.
A different recipe or dataset requires a separate experiment directory. The
current method warms up for 72 updates, follows cosine decay through update
2,400, and uses the 10% learning-rate floor through update 3,600. This is the
schedule used by the selected Dohnuts 0.1.0 export.

`--steps` changes an experiment's total budget. Extending a completed run retains
optimizer and random state, sampling position, and the decay horizon. Development
selection includes its existing best checkpoint, followed by calibration,
evaluation, and acceptance. Checkpoint steps are experiment coordinates, not
additional product versions.

The training objective is `--method`, one of `rlcd` (the default), `sft`, or
`grpo`. `--sigma` and `--ce-weight` configure `rlcd` and default to 0.3 and 1.0.
`--stage` opens parameters in four steps: `warmup` trains the decision head
alone, `text` adds language LoRA (the default), `joint` also opens the vision
merger, and `vision_top` also opens the top four vision blocks. The two vision
stages trade the frozen-image cache for correct gradients, so they recompute
every image feature. `--projection-dim`, `--lora-rank`, and `--lora-alpha`
default to 256, 8, and 16. `--steps` sets the total update budget, and `--data`,
`--model`, and `--output` identify local assets and output locations.

Objective details beyond those flags, such as `grpo.group_size` or
`sft.brier_weight`, belong in the generated `recipe-seed-*.json`. The file is a
closed recipe: the trainer rejects any configuration that differs from
`training_recipe(...)`. Flags do not select architectures; different backbone
support uses the [adapter interface](design.md).

## Multiple GPUs

`--gpus N` (default 1) trains on one machine with `N` data-parallel processes,
launched through `torchrun --nproc_per_node=N`. Each process owns one visible
device (`CUDA_VISIBLE_DEVICES=0..N-1`) and the model is replicated, not sharded,
so a 0.8B model still needs a full copy per GPU. `--master-port` sets the
torchrun rendezvous port (default 29500).

```bash
pdm run python scripts/run_experiment.py --gpus 4 --output runs/v1-4gpu
```

The effective global batch is `batch_size * accumulation * N`, while `--steps`
still counts optimizer updates, so the same recipe trains longer per step as
GPUs are added. `batch_size` and `accumulation` are fixed in the recipe; scale
`--steps` or add GPUs, not the per-rank batch. Development evaluation, telemetry,
and checkpoint writing stay on rank 0, so metrics and selection are identical in
shape to a single-GPU run. Final calibration and evaluation run in a separate
single-process step and are never distributed.

The world size is recorded in the run's `config.json`; resuming requires the same
`--gpus`, because the world size determines how each step's batch is partitioned.
Checkpoints are ordinary `format_version: 2` exports and load the same way whether
they were trained on one GPU or several. `--gpus 1` is the original code path with
no process group, and a plain `python -m dohnuts.train` launch stays single-GPU.

Run the GPU plumbing check with the elastic launcher (add `--full` to also train
two real Qwen3.5 SFT steps on the pinned base):

```bash
pdm run torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py
```

Checkpoints written before the two-projection head change carry
`format_version: 1`. Both the resume path and `Predictor.from_checkpoint` refuse
them, because the decision-head parameter keys and the frozen recipe have
changed. Start a new run; use `--initialize-from` only with an export written by
the current recipe.

Base initialization and checkpoint initialization use the same 26-group mixture
and training workflow. To initialize from an exported checkpoint:

```bash
pdm run python scripts/run_experiment.py --initialize-from runs/v1/checkpoint --output runs/domain
```

The checkpoint supplies the starting parameters and a fresh optimizer/schedule.
The runner automatically resumes a saved update in its output directory, restoring
optimizer and sampling state. Both modes use the same fused kernels, differentiable
prefix sharing, frozen-image cache, training objective, development selection and
calibration. There is no separate incremental training program.

## Outputs

The default data directory is `data/processed/v1`; outputs go to `runs/v1`:

| Artifact | Contents |
| --- | --- |
| `checkpoint/` | Selected head/LoRA/vision weights, method, stage, projection dimension, base revision, temperatures, selection metadata, checksum |
| `seed-42/resources.jsonl` | Timestamped GPU memory, power, utilization, temperature and process RSS |
| `seed-42/metrics.jsonl` | Loss, reward, development accuracy, learning rate, and GPU memory |
| `seed-42/test-predictions.jsonl` | Stable example IDs, targets, raw logits |
| `seed-42/evaluation.json` | Raw/calibrated per-task, decision-type and candidate-count metrics |
| `acceptance/report.json` | Actual checkpoint reload, decision interface, and Bub SDK acceptance |
| `quality-audit/` | Candidate rotations, image ablations, and batch-size sensitivity |
| `benchmarks/` | End-to-end timing samples and resource measurements |
| `laya-chart/` | Frozen application/language suites, calibration diagnostics and parallel decision timing |
| `jevbench/` | JevBench public-task predictions, raw evidence, and same-ID reference comparisons |
| `metrics/` | CSV tables, comparison JSON, and a self-contained results report |
| `figures/` | Separately rendered comparison figures, chart data, and source checksums |
| `recipe.json` | Fixed settings, the objective's effective options, and source hashes |

```python
from dohnuts.predictor import Predictor

model = Predictor.from_checkpoint("runs/v1/checkpoint")
```

Render comparison figures from the saved results using the
[plotting instructions](local-benchmarks.md#model-card-figures).

The compact checkpoint requires its pinned base model, which defaults to
`.cache/models/Qwen3.5-0.8B`. Raw data, cached base weights, and generated results
are local artifacts rather than package contents.
