"""Multi-GPU smoke test for data-parallel Dohnuts training.

Launch it with the elastic launcher, one process per GPU::

    torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py
    torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py --full   # real 0.8B, 2 SFT steps

Checks, in order:

* ``--full`` only: the real ``train()`` loop runs two SFT warmup steps across
  ranks and rank 0 leaves a loadable ``format_version 2`` checkpoint that
  records the world size.
* gradient synchronization: accumulating microbatches with the loop's
  ``no_sync`` policy yields exactly the analytic mean over the global batch.
* data partitioning: each rank reproduces the single-process batch at its
  global microbatch index, and the ranks' permutation seeds never collide.
"""

import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dohnuts import distributed
from dohnuts.recipe import BASE_MODEL, training_recipe
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


def synthetic_split(root, split, count):
    rows = []
    templates = [
        ("choice", ["alpha", "beta"], [1.0, 0.0]),
        ("noul", {"false": "no", "true": "yes"}, [0.0, 1.0]),
        ("score", ["low", "mid", "high"], [0.0, 1.0, 0.0]),
    ]
    for i in range(count):
        kind, criteria, target = templates[i % len(templates)]
        rows.append(
            {
                "id": f"{split}:{i}",
                "dataset": "smoke",
                "group": "smoke",
                "state": {"screen": f"page {i % 3}"},
                "question": {"type": kind, "instructions": "pick", "criteria": criteria},
                "target": target,
            }
        )
    path = root / f"{split}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def check_full_training(model_path, root):
    """Run the real loop for two SFT warmup steps and verify the checkpoint."""
    data = root / "data"
    data.mkdir()
    for split, count in {"train": 16, "dev": 6, "calibration": 3, "test": 3}.items():
        synthetic_split(data, split, count)
    recipe = training_recipe(
        model=model_path,
        data=data,
        steps=2,
        method="sft",
        stage="warmup",
    )
    run = root / "run"
    train(recipe, run)
    distributed.barrier()
    if distributed.is_main_process():
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        assert state["format_version"] == 2
        assert set(state["trainable"]) == {"head.decision.weight", "head.candidate.weight"}
        assert state["config"]["world_size"] == distributed.world_size()
        report(
            "full_training",
            steps=state["step"],
            trainable=sorted(state["trainable"]),
            world_size=state["config"]["world_size"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_MODEL, help="Local pinned backbone")
    parser.add_argument("--full", action="store_true", help="Also run two real 0.8B SFT steps")
    args = parser.parse_args()
    distributed.setup()
    try:
        if not distributed.is_distributed():
            raise SystemExit(
                "Launch with torchrun, e.g. torchrun --nproc_per_node=2 scripts/smoke_multi_gpu.py"
            )
        world = distributed.world_size()
        report("launcher", world_size=world, device=str(distributed.device()))
        check_gradient_sync()
        check_data_partition()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if args.full:
                if not (args.model / "revision.txt").is_file():
                    raise SystemExit(
                        f"A pinned snapshot with revision.txt is required: {args.model}"
                    )
                check_full_training(args.model, root)
        report("complete", world_size=world)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
