"""Resumable mixed-dataset training with development-only checkpoint selection.

Single-GPU by default.  Launched under ``torchrun --nproc_per_node=N`` the same
code trains with data parallelism: every rank owns one GPU, reads a disjoint
slice of each step's mixture, and DDP averages gradients across ranks.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dohnuts import __version__, distributed
from dohnuts.experiment import Sampler, emit, environment, memory
from dohnuts.metrics import by_dataset, by_primitive_and_candidates, fit_temperatures
from dohnuts.model import DecisionModel
from dohnuts.objectives import build_objective
from dohnuts.recipe import LR_DECAY_STEPS, training_recipe
from dohnuts.rlcd import RLCDConfig
from dohnuts.training_data import (
    DecisionCollator,
    EvaluationBatches,
    TrainingBatches,
    load_records,
    prefetch_batches,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, model, optimizer, step, config, score, training_state=None):
    payload = {
        "format_version": 2,
        "step": step,
        "config": config,
        "dev_macro_accuracy": score,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "training_state": training_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(model, path, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 2:
        raise ValueError(
            "Checkpoint format_version 1 or unknown cannot be resumed: the decision head changed "
            "from one Linear to two projections. Start a new run from --initialize-from instead."
        )
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(state["trainable"]) != expected:
        raise ValueError("Checkpoint trainable keys differ")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            params[name].copy_(value)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    return state


def reduce_consumed(consumed):
    """Global sample counts: ranks see disjoint batches, so their counts add."""
    keys = sorted(consumed)
    counts = torch.tensor(
        [consumed[key] for key in keys], dtype=torch.float32, device=distributed.device()
    )
    distributed.all_reduce_sum(counts)
    return {key: int(value) for key, value in zip(keys, counts.cpu().tolist())}


@torch.inference_mode()
def evaluate(model, groups, collator, output, batch_size=16):
    model.eval()
    records = []
    loader = torch.utils.data.DataLoader(
        EvaluationBatches(groups, collator, batch_size),
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )
    pending = deque()
    counts = Counter()

    with Path(output).open("w") as stream:

        def collect():
            ready, logits, rows = pending.popleft()
            # D2H is asynchronous: CPU must not inspect pinned results before this event.
            ready.synchronize()
            for i, row in enumerate(rows):
                result = {k: row[k] for k in ["id", "dataset", "group", "target"]}
                result.update(
                    type=row["question"]["type"],
                    logits=logits[i, : len(row["target"])].tolist(),
                )
                if not torch.isfinite(logits[i, : len(row["target"])]).all():
                    raise RuntimeError(f"Non-finite prediction: {row['id']}")
                stream.write(json.dumps(result) + "\n")
                records.append(result)
                counts[row["dataset"]] += 1
            key = rows[-1]["dataset"]
            if counts[key] == len(groups[key]):
                stream.flush()
                print(
                    json.dumps({"kind": "evaluation_progress", "dataset": key, "n": counts[key]}),
                    flush=True,
                )

        for (inputs, positions, decision_positions, _, _, _), rows in prefetch_batches(loader):
            logits = model(inputs, positions, decision_positions)
            host = torch.empty_like(logits, device="cpu", pin_memory=True)
            host.copy_(logits, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            pending.append((ready, host, rows))
            # Submit the following forward before serializing the preceding results.
            if len(pending) == 2:
                collect()
        while pending:
            collect()
    return records


def wrap_data_parallel(model, *, stage):
    """Replicate the model across ranks and synchronize gradients.

    ``find_unused_parameters`` is only needed when the vision tower is open:
    a microbatch without an image never touches the merger or vision blocks, so
    those parameters legitimately receive no gradient.  Language and head
    parameters are always exercised, so warmup/text keep the faster static path.
    DDP's default coalesced buffer sync is left on: it also guarantees the two
    replicas start identical, and the buffers are tiny.
    """
    if not distributed.is_distributed():
        return model
    return DDP(
        model,
        device_ids=[distributed.local_rank()],
        output_device=distributed.local_rank(),
        find_unused_parameters=stage in {"joint", "vision_top"},
    )


def train(config, run, *, resume=False, adapter=None, initialize_from=None):
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    raw = DecisionModel(config["model"], adapter=adapter, projection_dim=config["projection_dim"])
    raw.enable_stage(
        config["stage"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"]
    )
    parent = None
    if initialize_from is not None:
        metadata = raw.load_adapter(initialize_from)
        parent = {"checkpoint": str(initialize_from), "weights_sha256": metadata["weights_sha256"]}
    parameters = [p for p in raw.parameters() if p.requires_grad]
    groups = {"head": [], "lora": [], "merger": [], "vision": []}
    for name, parameter in raw.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            key = "head"
        elif "lora_" in name:
            key = "lora"
        elif ".visual.merger." in name:
            key = "merger"
        elif ".visual.blocks." in name:
            key = "vision"
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        groups[key].append(parameter)
    rates = {
        "head": config["head_lr"],
        "lora": config["backbone_lr"],
        "merger": config["merger_lr"],
        "vision": config["vision_lr"],
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": members, "lr": rates[key], "initial_lr": rates[key]}
            for key, members in groups.items()
            if members
        ],
        weight_decay=0.01,
    )
    data = Path(config["data"])
    groups = load_records(data / "train.jsonl", config["train_cap"])
    dev = load_records(data / "dev.jsonl", config["dev_cap"])
    source_hashes = {
        split: file_hash(data / f"{split}.jsonl")
        for split in ["train", "dev", "calibration", "test"]
    }
    config_path = run / "config.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    lr_decay_steps = (
        previous["lr_decay_steps"] if previous else min(config["steps"], LR_DECAY_STEPS)
    )
    step = 0
    best = -1.0
    training_state = None
    if resume:
        state = load_checkpoint(raw, run / "last.pt", optimizer)
        step = state["step"]
        best = state["dev_macro_accuracy"]
        training_state = state.get("training_state")
    # The reference snapshot is built from the unwrapped module: a DDP wrapper
    # must never be deep-copied.
    objective = build_objective(config, model=raw, training_state=training_state, resuming=resume)
    frozen = {
        **config,
        "source_hashes": source_hashes,
        "objective": objective.describe(),
        "sampling": "uniform dataset, replacement, seed+microbatch index; choice permutation only",
        "dev_selection": "fixed SHA-256 subset per dataset; unweighted dataset macro top-1 accuracy",
        "lr_decay_steps": lr_decay_steps,
    }
    if parent is not None:
        frozen["initialized_from"] = parent
    if raw.adapter.name != "qwen3.5":
        frozen["adapter"] = raw.adapter.name
        frozen["base_model"] = raw.adapter.base_model
    if distributed.is_distributed():
        # The world size fixes how each step's mixture is partitioned, so a
        # resume must reproduce it; single-GPU records stay byte-identical.
        frozen["world_size"] = distributed.world_size()
    if previous is not None:
        expected = {
            **previous,
            "lr_decay_steps": lr_decay_steps,
            "steps": config["steps"],
        }
        if not resume or expected != frozen or config["steps"] < previous["steps"]:
            raise ValueError("Resume may only extend the step budget of the same recipe and data")
    # Every rank has now read the config it started from. Without this barrier a
    # slow rank would read the file rank 0 is about to create and treat a fresh
    # run as an unresumed continuation of itself.
    distributed.barrier()
    if distributed.is_main_process():
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(frozen, indent=2) + "\n")
        temporary.replace(config_path)
    collator = DecisionCollator(config["model"], adapter=adapter)
    # Every rank holds all dataset keys so the collective count reduction is
    # well defined even before a rank has sampled from each dataset.
    consumed = Counter({key: 0 for key in groups})
    elapsed_before_resume = 0.0
    if resume:
        # Only saved updates belong to the resumed trajectory. A stopped process
        # may have logged later steps before its next checkpoint was committed.
        if distributed.is_main_process():
            metrics_path = run / "metrics.jsonl"
            committed = [
                line
                for line in metrics_path.read_text().splitlines()
                if json.loads(line).get("step", 0) <= step
                and json.loads(line).get("kind") != "train_complete"
            ]
            temporary = metrics_path.with_suffix(".tmp")
            temporary.write_text("\n".join(committed) + "\n")
            temporary.replace(metrics_path)
            emit(
                metrics_path,
                {
                    "kind": "resume",
                    "step": step,
                    "target_step": config["steps"],
                    "lr_decay_steps": lr_decay_steps,
                    "optimizer_lr": [group["lr"] for group in optimizer.param_groups],
                    "checkpoint_sha256": file_hash(run / "last.pt"),
                },
            )
        # Restore plotting counters from the last logged step at/before the
        # checkpoint. Read the file after the trim so all ranks agree.
        distributed.barrier()
        for line in (run / "metrics.jsonl").read_text().splitlines():
            logged = json.loads(line)
            if logged.get("kind") == "train" and logged["step"] <= step:
                consumed.update(logged["consumed"])
                elapsed_before_resume = logged["elapsed_s"]
    else:
        if (run / "metrics.jsonl").exists():
            raise ValueError("Existing run requires --resume")
    # The model is not wrapped until the reference snapshot exists, and the
    # constructor itself is a collective that every rank must reach.
    model = wrap_data_parallel(raw, stage=config["stage"])
    if not resume:
        if distributed.is_main_process():
            emit(
                run / "metrics.jsonl",
                environment(raw, Path(config["model"]), method=config["method"]),
            )
            (run / "samples.json").write_text(
                json.dumps(
                    {
                        "train": {k: [r["id"] for r in v] for k, v in groups.items()},
                        "dev": {k: [r["id"] for r in v] for k, v in dev.items()},
                    },
                    indent=2,
                )
            )
            baseline = evaluate(
                raw, dev, collator, run / "dev-step-000000.jsonl", config["eval_batch_size"]
            )
            baseline_metrics = by_dataset(baseline)
            emit(run / "metrics.jsonl", {"kind": "dev", "step": 0, "metrics": baseline_metrics})
            best = baseline_metrics["macro_accuracy"]
            # Continuing from a good checkpoint may never improve development
            # quality. In that case the untouched parent stays selected.
            save_checkpoint(
                run / "best.pt", raw, optimizer, 0, frozen, best, objective.state_dict()
            )
        distributed.barrier()
    dataset = TrainingBatches(
        groups,
        collator,
        seed=config["seed"],
        batch_size=config["batch_size"],
        steps=config["steps"],
        accumulation=config["accumulation"],
        start_step=step,
        rank=distributed.rank(),
        world_size=distributed.world_size(),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=config["workers"],
        prefetch_factor=2 if config["workers"] else None,
        pin_memory=True,
    )
    iterator = prefetch_batches(loader)
    start_time = time.perf_counter()
    with Sampler(
        Path("/sys/class/drm/card1/device"),
        interval=1.0,
        output=run / "resources.jsonl",
        enabled=distributed.is_main_process(),
    ) as telemetry:
        while step < config["steps"]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            decay_steps = frozen["lr_decay_steps"]
            warmup = max(1, int(decay_steps * 0.03))
            progress = (min(step, decay_steps) - warmup) / max(1, decay_steps - warmup)
            factor = (
                min(1.0, (step + 1) / warmup)
                if step < warmup
                else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            stats = Counter()
            step_start = time.perf_counter()
            updates = objective.num_iterations
            prepared = []
            for _micro in range(config["accumulation"]):
                batch, key, ids = next(iterator)
                inputs, positions, decision_positions, mask, target, ordinal = batch
                rollout = None
                reference_logits = None
                if objective.needs_rollout and updates > 1:
                    # One rollout frozen per group; every policy update reuses the
                    # same actions, advantages, and old log-probabilities. The
                    # rollout forward bypasses DDP: it carries no gradient and
                    # must not perturb the reducer's bookkeeping.
                    with torch.no_grad():
                        rollout = objective.prepare(
                            raw(inputs, positions, decision_positions), target, mask
                        )
                    reference_logits = objective.reference_logits(
                        inputs, positions, decision_positions
                    )
                prepared.append(
                    (
                        inputs,
                        positions,
                        decision_positions,
                        mask,
                        target,
                        ordinal,
                        rollout,
                        reference_logits,
                    )
                )
                consumed[key] += len(ids)
            weight = config["accumulation"] * updates
            for _update in range(updates):
                optimizer.zero_grad(set_to_none=True)
                finite = torch.ones((), dtype=torch.float32, device=distributed.device())
                for index, item in enumerate(prepared):
                    inputs, positions, decision_positions, mask, target, ordinal = item[:6]
                    rollout, reference_logits = item[6], item[7]
                    # Accumulate locally and synchronize once, on the last
                    # microbatch of this update; synchronizing mid-accumulation
                    # would mix a partial sum into the reduced gradient.
                    sync = index == len(prepared) - 1
                    context = (
                        model.no_sync()
                        if distributed.is_distributed() and not sync
                        else nullcontext()
                    )
                    with context:
                        logits = model(inputs, positions, decision_positions)
                        if objective.needs_rollout and rollout is None:
                            # A single iteration shares its forward with the rollout.
                            with torch.no_grad():
                                rollout = objective.prepare(logits, target, mask)
                            reference_logits = objective.reference_logits(
                                inputs, positions, decision_positions
                            )
                            prepared[index] = (*item[:6], rollout, reference_logits)
                        loss, metrics = objective.loss(
                            logits, target, mask, ordinal, rollout, reference_logits
                        )
                        finite *= torch.isfinite(loss.detach()).float()
                        (loss / config["accumulation"]).backward()
                    stats.update({k: v.detach() / weight for k, v in metrics.items()})
                    stats["loss"] += loss.detach() / weight
                    stats["accuracy"] += (
                        logits.detach().masked_fill(~mask, -torch.inf).argmax(-1)
                        == target.argmax(-1)
                    ).float().mean() / weight
                # Reduce before deciding: one rank must never raise while the
                # others wait inside the next collective.
                distributed.all_reduce_mean(finite)
                if not bool(finite):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}; optimizer was not advanced"
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                # Each iteration is a real policy update from the frozen rollout,
                # which is what lets the clipped ratio bind.
                optimizer.step()
            step += 1
            if step == 1 or step % config["log_every"] == 0:
                torch.cuda.synchronize()
                # Counter's stub assumes int values; these accumulators contain tensors.
                totals = cast(list[torch.Tensor], [stats[key] for key in stats])
                values = torch.stack(totals)
                distributed.all_reduce_mean(values)
                global_consumed = reduce_consumed(consumed)
                if distributed.is_main_process():
                    emit(
                        run / "metrics.jsonl",
                        {
                            "kind": "train",
                            "step": step,
                            "elapsed_s": elapsed_before_resume + time.perf_counter() - start_time,
                            "step_s": time.perf_counter() - step_start,
                            **dict(zip(stats, values.cpu().tolist())),
                            "grad_norm": float(grad_norm),
                            "lr": optimizer.param_groups[0]["lr"],
                            "consumed": global_consumed,
                            "memory": memory(),
                        },
                    )
            if step % config["eval_every"] == 0 or step == config["steps"]:
                if distributed.is_main_process():
                    predictions = evaluate(
                        raw,
                        dev,
                        collator,
                        run / f"dev-step-{step:06d}.jsonl",
                        config["eval_batch_size"],
                    )
                    metrics = by_dataset(predictions)
                    score = metrics["macro_accuracy"]
                    emit(run / "metrics.jsonl", {"kind": "dev", "step": step, "metrics": metrics})
                    if score > best:
                        best = score
                        save_checkpoint(
                            run / "best.pt",
                            raw,
                            optimizer,
                            step,
                            frozen,
                            best,
                            objective.state_dict(),
                        )
                    save_checkpoint(
                        run / "last.pt", raw, optimizer, step, frozen, best, objective.state_dict()
                    )
                # Hold every rank until the evaluation and its checkpoint land;
                # the next collective would otherwise wait on rank 0 anyway.
                distributed.barrier()
            elif step % config["save_every"] == 0 and distributed.is_main_process():
                save_checkpoint(
                    run / "last.pt", raw, optimizer, step, frozen, best, objective.state_dict()
                )
    global_consumed = reduce_consumed(consumed)
    if distributed.is_main_process():
        emit(
            run / "metrics.jsonl",
            {
                "kind": "train_complete",
                "step": step,
                "best_dev_macro_accuracy": best,
                "telemetry": telemetry.summary(),
                "consumed": global_consumed,
            },
        )


def final_evaluation(config, run, *, adapter=None):
    if distributed.is_distributed():
        raise ValueError("Final evaluation runs on one process; launch it without torchrun")
    frozen = json.loads((run / "config.json").read_text())
    for key in [
        "model",
        "data",
        "method",
        "stage",
        "projection_dim",
        "lora_rank",
        "lora_alpha",
        "image_pixels",
        "max_length",
        "backend",
    ]:
        if config.get(key) != frozen.get(key):
            raise ValueError(f"Evaluation recipe differs from the selected run: {key}")
    data = Path(config["data"])
    for split, expected in frozen["source_hashes"].items():
        if file_hash(data / f"{split}.jsonl") != expected:
            raise ValueError(f"Evaluation data changed since training: {split}")
    torch.set_num_threads(config["cpu_threads"])
    model = DecisionModel(config["model"], adapter=adapter, projection_dim=frozen["projection_dim"])
    if model.adapter.name != frozen.get("adapter", "qwen3.5"):
        raise ValueError("Evaluation adapter differs from the trained model")
    model.enable_stage(
        frozen["stage"],
        lora_rank=frozen["lora_rank"],
        lora_alpha=frozen["lora_alpha"],
        checkpointing=False,
    )
    state = load_checkpoint(model, run / "best.pt")
    # Calibration and final evaluation use exactly the deployed merged model.
    model.merge()
    collator = DecisionCollator(config["model"], adapter=adapter)
    calibration = evaluate(
        model,
        load_records(data / "calibration.jsonl"),
        collator,
        run / "calibration-predictions.jsonl",
        config["eval_batch_size"],
    )
    temperatures = fit_temperatures(calibration)
    (run / "temperatures.json").write_text(json.dumps(temperatures, indent=2) + "\n")
    predictions = evaluate(
        model,
        load_records(data / "test.jsonl"),
        collator,
        run / "test-predictions.jsonl",
        config["eval_batch_size"],
    )
    training_monitor = evaluate(
        model,
        load_records(data / "train.jsonl", config["dev_cap"]),
        collator,
        run / "train-monitor-predictions.jsonl",
        config["eval_batch_size"],
    )
    report = {
        "selected_step": state["step"],
        "checkpoint_sha256": file_hash(run / "best.pt"),
        "source_hashes": frozen["source_hashes"],
        "seed": frozen["seed"],
        "inference": "merged LoRA, BF16, fused operations, shared-prefix parallel candidate scoring",
        "temperatures": temperatures,
        "uncalibrated": by_dataset(predictions),
        "calibrated": by_dataset(predictions, temperatures),
        "primitive_candidate_slices": by_primitive_and_candidates(predictions, temperatures),
        "train_monitor": by_dataset(training_monitor, temperatures),
        "train_monitor_scope": "fixed subset of the capped training pool; diagnostic only, no selection",
    }
    (run / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    export_checkpoint(state, run, run.parent / "checkpoint")
    print(
        json.dumps(
            {
                "kind": "evaluation_complete",
                "run": str(run),
                "macro_accuracy": report["calibrated"]["macro_accuracy"],
            }
        ),
        flush=True,
    )


def export_checkpoint(state, run, output):
    """Export the development-selected, calibrated weights from this run."""
    from safetensors.torch import save_file

    config = state["config"]
    base = Path(config["model"])
    calibration_path = run / "temperatures.json"
    if not calibration_path.exists():
        raise FileNotFoundError("Calibrate the selected run before export")
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "adapter.safetensors"
    temporary = weights.with_suffix(".tmp")
    save_file(
        {name: value.contiguous() for name, value in state["trainable"].items()}, str(temporary)
    )
    temporary.replace(weights)
    metadata = {
        "format_version": 2,
        "project": "dohnuts",
        "distribution": "dohnuts",
        "version": __version__,
        "model_id": f"Dohnuts-{__version__}-0.8B",
        "base_model": config.get("base_model", "Qwen/Qwen3.5-0.8B"),
        "base_path": str(base),
        "base_revision": (base / "revision.txt").read_text().strip(),
        "adapter": config.get("adapter", "qwen3.5"),
        "method": config["method"],
        "stage": config["stage"],
        "projection_dim": config["projection_dim"],
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "image_pixels": config["image_pixels"],
        "max_length": config["max_length"],
        "backend": config.get("backend", {}),
        "inference": "merged LoRA, BF16, fused operations, shared-prefix parallel candidate scoring",
        "source_hashes": config["source_hashes"],
        "selected_run": str(run),
        "selected_step": state["step"],
        "selection": "maximum development macro accuracy within the run; no test selection",
        "seed": config["seed"],
        "dev_macro_accuracy": state["dev_macro_accuracy"],
        "temperatures": json.loads(calibration_path.read_text()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    if "initialized_from" in config:
        metadata["initialized_from"] = config["initialized_from"]
    (output / "dohnuts.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from", type=Path, help="Exported checkpoint used to initialize training"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = training_recipe(
        model=config["model"],
        data=config["data"],
        seed=config["seed"],
        rlcd=RLCDConfig(**config.get("rlcd", {})),
        steps=config["steps"],
        method=config.get("method", "rlcd"),
        stage=config.get("stage", "text"),
        projection_dim=config.get("projection_dim", 256),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        sft=config.get("sft"),
        grpo=config.get("grpo"),
    )
    if config != expected:
        raise ValueError("Training uses the fixed recipe, objective controls, and step budget")
    distributed.setup()
    try:
        if args.action == "train":
            train(config, args.run, resume=args.resume, initialize_from=args.initialize_from)
        else:
            final_evaluation(config, args.run)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
