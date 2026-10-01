"""Multi-GPU smoke test for data-parallel Dohnuts training.

Launch it with the elastic launcher, one process per GPU::

    torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py
    torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py --full   # real 0.8B, 2 steps

``--full`` takes ``--method`` (sft, rlcd, grpo) and ``--stage`` (warmup ..
vision_top), so any combination can be run for real; ``--data`` points at a
prepared mixture to take genuine screenshots from, and ``--workdir`` keeps the
run so a later launch can resume it.

Checks, in order:

* gradient synchronization: accumulating microbatches with the loop's
  ``no_sync`` policy yields exactly the analytic mean over the global batch.
* data partitioning: each rank reproduces the single-process batch at its
  global microbatch index, and the ranks' permutation seeds never collide.
* ``--full``: the real ``train()`` loop runs across ranks and rank 0 leaves a
  loadable ``format_version 2`` checkpoint whose trainable keys match the stage
  and whose config records the world size.
* ``--compare RUN_A RUN_B``: two runs that drew the same global microbatches --
  one process with ``accumulation`` A, N processes with A/N -- agree on the
  first update to ``--tolerance``.  Later records are reported as drift only,
  because Adam amplifies rounding differences once updates accumulate.  Needs no
  GPU and no torchrun.
"""

import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from dohnuts import distributed
from dohnuts.recipe import BASE_MODEL, EPOCHS, training_recipe
from dohnuts.train import train
from dohnuts.training_data import TrainingBatches, microbatch_index


def report(check, **fields):
    if distributed.is_main_process():
        print(json.dumps({"check": check, **fields}, default=str), flush=True)


def check_gradient_sync(steps=3, accumulation=3):
    """Every rank's reduced gradient must equal the global batch mean."""
    world = distributed.world_size()
    device = distributed.device()
    total = world * accumulation * steps
    samples = torch.arange(1.0, 1.0 + total, device=device).view(-1, 1)
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    parallel = DDP(model, device_ids=[distributed.local_rank()])
    optimizer = torch.optim.SGD(parallel.parameters(), lr=0.0)
    worst = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            index = step * accumulation * world + distributed.rank() * accumulation + micro
            # Same policy as the training loop: reduce once, on the last microbatch.
            context = parallel.no_sync() if micro < accumulation - 1 else nullcontext()
            with context:
                loss = parallel(samples[index]).sum()
                (loss / accumulation).backward()
        expected = samples[step * accumulation * world : (step + 1) * accumulation * world].mean()
        worst = max(worst, abs(model.weight.grad.item() - expected.item()))
    assert worst < 1e-5, f"reduced gradient differs from the global mean by {worst}"
    report("gradient_sync", world=world, accumulation=accumulation, max_error=worst)


class RecordingCollator:
    def __call__(self, rows, *, permutation_seed=None):
        return [row["id"] for row in rows], permutation_seed


def check_data_partition(steps=3, accumulation=2):
    """Ranks read disjoint global microbatches and each matches the serial run."""
    world = distributed.world_size()
    groups = {"gui": [{"id": f"gui-{i}"} for i in range(8)]}
    local = TrainingBatches(
        groups,
        RecordingCollator(),
        seed=7,
        batch_size=2,
        steps=steps,
        accumulation=accumulation,
        start_step=0,
        rank=distributed.rank(),
        world_size=world,
    )
    serial = TrainingBatches(
        groups,
        RecordingCollator(),
        seed=7,
        batch_size=2,
        steps=steps,
        accumulation=accumulation,
        start_step=0,
        rank=0,
        world_size=1,
    )
    seeds = []
    for step in range(steps):
        for micro in range(accumulation):
            global_index = microbatch_index(
                step * accumulation + micro,
                start_step=0,
                accumulation=accumulation,
                rank=distributed.rank(),
                world_size=world,
            )
            batch = local[step * accumulation + micro]
            assert batch == serial[global_index]
            seeds.append(batch[0][1])
    gathered = distributed.all_gather_object((distributed.rank(), seeds))
    flat = [seed for _, rank_seeds in gathered for seed in rank_seeds]
    assert len(set(flat)) == len(flat), "ranks drew the same microbatch permutation"
    report("data_partition", world=world, microbatches=len(flat))


HEAD_KEYS = {"head.decision.weight", "head.candidate.weight"}
# Wall-clock and the telemetry snapshot say nothing about the objective.  The
# gradient norm stays in: at the first update it is the sharpest available
# evidence that the reduced gradient is the global mean rather than a partial sum.
SKIP = {"elapsed_s", "step_s", "memory"}
TOLERANCE = 1e-4


def write_image(path, index):
    """A deterministic stand-in screenshot.

    It lets a vision stage be smoke-tested on a machine without the prepared
    mixture; the vision tower sees real patches either way.
    """
    side = 128
    data = bytearray(3 * side * side)
    for pixel in range(side * side):
        offset = pixel * 3
        data[offset] = (pixel + index) % 256
        data[offset + 1] = (pixel // side) * 2 % 256
        data[offset + 2] = (pixel % side) * 2 % 256
    Image.frombytes("RGB", (side, side), bytes(data)).save(path)
    return path


def synthetic_rows(split, count, *, dataset="text", images=None):
    rows = []
    templates = [
        ("choice", ["alpha", "beta"], [1.0, 0.0]),
        ("noul", {"false": "no", "true": "yes"}, [0.0, 1.0]),
        ("score", ["low", "mid", "high"], [0.0, 1.0, 0.0]),
    ]
    for index in range(count):
        kind, criteria, target = templates[index % len(templates)]
        row = {
            "id": f"{split}:{dataset}:{index}",
            "dataset": dataset,
            "group": dataset,
            "state": {"screen": f"page {index % 3}"},
            "question": {"type": kind, "instructions": "pick", "criteria": criteria},
            "target": target,
        }
        if images is not None:
            row["image"] = str(write_image(images / f"{dataset}-{split}-{index}.png", index))
        rows.append(row)
    return rows


def resolve_image(value, source):
    """A prepared mixture stores repository-relative screenshot paths."""
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"The mixture references a missing screenshot: {value}")
    for candidate in (Path.cwd() / path, source / path, source / "images" / path.name):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"The mixture references a missing screenshot: {value}")


def build_mixture(root, *, source, stage, train_rows, dev_rows):
    """A two-dataset mixture: screenshots and plain text.

    The text rows are not filler.  An image-less microbatch is exactly the case
    where the merger and the vision blocks receive no gradient, which is what
    makes the DDP ``find_unused_parameters`` decision load-bearing rather than
    cosmetic.
    """
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    sizes = {"train": train_rows, "dev": dev_rows, "calibration": dev_rows, "test": dev_rows}
    for split, count in sizes.items():
        rows = []
        if source is not None:
            with (source / f"{split}.jsonl").open() as stream:
                for line in stream:
                    if len(rows) >= count:
                        break
                    row = json.loads(line)
                    if row.get("image"):
                        row["image"] = str(resolve_image(row["image"], source))
                    rows.append(row)
        elif stage in {"joint", "vision_top"}:
            images.mkdir(exist_ok=True)
            rows = synthetic_rows(split, count, dataset="image", images=images)
        rows += synthetic_rows(split, max(4, count // 2), dataset="text")
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def check_trainable(stage, trainable):
    """Assert the structure a stage promises, by group rather than by name list.

    Every stage names both what it must open and what it must leave frozen, so a
    leaked ``requires_grad`` cannot slip through as merely unexpected.
    """
    names = set(trainable)
    assert HEAD_KEYS <= names, f"{stage} must keep the decision head trainable"
    groups = {
        "lora": {name for name in names if ".lora_" in name},
        "merger": {name for name in names if name.startswith("backbone.visual.merger")},
        "blocks": {name for name in names if name.startswith("backbone.visual.blocks")},
    }
    opened = {key for key, value in groups.items() if value}
    expected = {
        "warmup": set(),
        "text": {"lora"},
        "joint": {"lora", "merger"},
        "vision_top": {"lora", "merger", "blocks"},
    }[stage]
    assert opened == expected, f"{stage} opened {sorted(opened)}, expected {sorted(expected)}"
    if stage == "warmup":
        assert names == HEAD_KEYS, f"warmup trains the head only, found {sorted(names)[:3]}"
    return {key: len(value) for key, value in groups.items()}


def check_full_training(args, root):
    """Run the real loop, then verify what rank 0 leaves behind."""
    data = root / "data"
    if not (data / "train.jsonl").exists():
        build_mixture(
            data,
            source=args.data,
            stage=args.stage,
            train_rows=args.train_rows,
            dev_rows=args.dev_rows,
        )
    recipe = training_recipe(
        model=args.model,
        data=data,
        epochs=args.epochs if args.epochs is not None else EPOCHS,
        # An epoch budget leaves `steps` to derive; a fixed one ignores epochs.
        steps=None if args.epochs is not None else args.steps,
        method=args.method,
        stage=args.stage,
    )
    if args.accumulation is not None:
        # Held against world_size so two runs draw the same global microbatches.
        recipe["accumulation"] = args.accumulation
    if args.method == "grpo":
        # Two policy updates from one frozen rollout: the loop must synchronize
        # once per update, not once per step.  A small group keeps it cheap.
        recipe["grpo"] = {"group_size": 4, "num_iterations": 2}
    run = root / "run"
    train(recipe, run, resume=args.resume)
    distributed.barrier()
    if not distributed.is_main_process():
        return
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert state["format_version"] == 2
    opened = check_trainable(args.stage, state["trainable"])
    world = distributed.world_size()
    if world > 1:
        assert state["config"]["world_size"] == world
    else:
        assert "world_size" not in state["config"], (
            "a single-GPU config stays free of distributed keys"
        )
    report(
        "full_training",
        method=args.method,
        stage=args.stage,
        resumed=args.resume,
        steps=state["step"],
        # The budget the run resolved to: epochs arrive here as an update count.
        budget=state["config"]["steps"],
        epochs=state["config"].get("epochs"),
        run=str(run),
        world_size=world,
        opened=opened,
    )


def flatten(record, prefix=""):
    values = {}
    for key, value in record.items():
        if key in SKIP:
            continue
        if isinstance(value, dict):
            values.update(flatten(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"{prefix}{key}"] = float(value)
    return values


def index_metrics(path):
    table = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") == "train":
            table[f"train:{record['step']}"] = flatten(record)
        elif record.get("kind") == "dev":
            for dataset, metrics in record.get("metrics", {}).items():
                if isinstance(metrics, dict):
                    table[f"dev:{record['step']}:{dataset}"] = flatten(metrics)
    return table


def compare_runs(left, right, tolerance=TOLERANCE):
    """Prove DDP and one process compute the same objective.

    With ``accumulation * world_size`` equal, both runs draw the identical block
    of global microbatches.  Only the records describing the *first* update are
    asserted: the baseline evaluation and the first step's metrics are a pure
    function of the shared starting weights and that shared block, so any
    difference is reduction order -- and a wrong rank layout would show up here
    as a factor of the world size, not as noise.

    Later records are reported but never failed.  Adam divides by a near-zero
    second moment early, so a difference at the level of rounding is amplified
    into a visible evaluation gap within a few updates.  That is chaos, not a
    partition error, and asserting on it would bury the signal that matters.
    """
    first, second = (
        index_metrics(Path(directory) / "metrics.jsonl") for directory in (left, right)
    )
    common = sorted(set(first) & set(second))
    if not common:
        print(json.dumps({"check": "equivalence", "error": "no shared metric records"}), flush=True)
        return 1
    pinned_worst, pinned_where = 0.0, None
    drift_worst, drift_where = 0.0, None
    exact = 0
    for key in common:
        pinned = key == "train:1" or key.startswith("dev:0:")
        for name, value in first[key].items():
            if name not in second[key]:
                continue
            other = second[key][name]
            if name.startswith("consumed.") or name == "lr" or name.endswith(".n"):
                assert value == other, f"{key} {name}: {value} != {other}"
                exact += 1
                continue
            error = abs(value - other) / max(abs(value), abs(other), 1e-6)
            if pinned:
                if error > pinned_worst:
                    pinned_worst, pinned_where = error, f"{key}.{name}"
            elif error > drift_worst:
                drift_worst, drift_where = error, f"{key}.{name}"
    print(
        json.dumps(
            {
                "check": "equivalence",
                "records": len(common),
                "exact": exact,
                "first_update_error": pinned_worst,
                "first_update_at": pinned_where,
                "tolerance": tolerance,
                "later_drift_error": drift_worst,
                "later_drift_at": drift_where,
                "verdict": pinned_worst <= tolerance,
            }
        ),
        flush=True,
    )
    return 0 if pinned_worst <= tolerance else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_MODEL, help="Local pinned backbone")
    parser.add_argument("--full", action="store_true", help="Also run the real train() loop")
    parser.add_argument("--method", default="sft", choices=("sft", "rlcd", "grpo"))
    parser.add_argument(
        "--stage", default="warmup", choices=("warmup", "text", "joint", "vision_top")
    )
    parser.add_argument("--steps", type=int, default=2, help="Optimizer updates for --full")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Budget --full in passes over the training split instead of a fixed update count",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        help="Override microbatches per update (for the equivalence run)",
    )
    parser.add_argument(
        "--data", type=Path, help="Prepared mixture to draw real screenshots from", default=None
    )
    parser.add_argument("--train-rows", type=int, default=48)
    parser.add_argument("--dev-rows", type=int, default=16)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Keep the run here so a later launch can --resume it",
        default=None,
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume --workdir with a larger --steps"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two run directories and exit; needs no GPU and no torchrun",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help="Allowed relative error on the first update (text-only runs match bit for bit)",
    )
    args = parser.parse_args()
    if args.compare is not None:
        raise SystemExit(compare_runs(*args.compare, tolerance=args.tolerance))
    # Reject a missing snapshot before any rank has loaded a backbone.
    if args.full and not (args.model / "revision.txt").is_file():
        raise SystemExit(f"A pinned snapshot with revision.txt is required: {args.model}")
    distributed.setup()
    try:
        world = distributed.world_size()
        if world == 1 and not args.full:
            raise SystemExit(
                "Launch with torchrun, e.g. torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py"
            )
        report("launcher", world_size=world, device=str(distributed.device()))
        if world > 1:
            check_gradient_sync()
            check_data_partition()
        if args.full:
            holder = None
            if args.workdir is None:
                holder = tempfile.TemporaryDirectory(prefix="dohnuts-smoke-")
                root = Path(holder.name)
            else:
                root = args.workdir
                root.mkdir(parents=True, exist_ok=True)
            check_full_training(args, root)
            if holder is not None:
                holder.cleanup()
        report("complete", world_size=world)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
