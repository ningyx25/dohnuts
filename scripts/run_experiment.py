"""Prepare the training mixture, train, calibrate, select, and accept a checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from dohnuts.recipe import BASE_MODEL, DATA, EPOCHS, TRAINING_STEPS, training_recipe
from dohnuts.rlcd import RLCDConfig
from dohnuts.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="Prepared mixture directory")
    parser.add_argument("--model", type=Path, default=BASE_MODEL, help="Local pinned backbone")
    parser.add_argument("--output", type=Path, default=Path("runs/v1"))
    parser.add_argument(
        "--steps",
        type=int,
        help="Fixed number of optimizer updates; overrides --epochs and extends a run in place",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Passes over the training split (default 3); ignored when --steps or TRAINING_STEPS is set",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initial LoRA/head weights for the training run",
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="RLCD logit perturbation strength")
    parser.add_argument(
        "--ce-weight", type=float, default=1.0, help="Auxiliary cross-entropy weight"
    )
    parser.add_argument("--method", default="rlcd", help="Training objective: rlcd, sft, or grpo")
    parser.add_argument("--stage", default="text", help="Opened parameters: warmup..vision_top")
    parser.add_argument("--projection-dim", type=int, default=256, help="Decision head width")
    parser.add_argument("--lora-rank", type=int, default=8, help="Language LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Language LoRA alpha")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs for data-parallel training (torchrun --nproc_per_node); evaluation stays single",
    )
    parser.add_argument("--master-port", type=int, default=29500, help="torchrun rendezvous port")
    args = parser.parse_args()
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")
    policy = RLCDConfig(sigma=args.sigma, ce_weight=args.ce_weight)
    recipe_path = args.output / "recipe.json"
    previous = json.loads(recipe_path.read_text()) if recipe_path.exists() else None
    # A step budget pinned in dohnuts.recipe outranks everything else; an explicit
    # --steps pins one too; otherwise the budget is epochs.
    steps = TRAINING_STEPS if TRAINING_STEPS is not None else args.steps
    epochs = args.epochs
    if steps is None and epochs is None and previous is not None:
        # Re-running with no budget keeps the existing run's budget, as before.
        steps = previous["recipe"].get("steps")
        epochs = previous["recipe"].get("epochs", EPOCHS)
    recipe = training_recipe(
        model=args.model,
        data=args.data,
        rlcd=policy,
        epochs=EPOCHS if epochs is None else epochs,
        steps=steps,
        method=args.method,
        stage=args.stage,
        projection_dim=args.projection_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(args.gpus)),
        "TRITON_CACHE_DIR": str(Path(".cache/triton").resolve()),
        "HF_HOME": str(Path(".cache/hf").resolve()),
        "TOKENIZERS_PARALLELISM": "false",
    }
    completed = args.output / "completed.json"
    done = json.loads(completed.read_text()) if completed.exists() else {}

    def run(name, *command):
        command = list(map(str, command))
        if done.get(name) == command:
            return
        print(json.dumps({"stage": name, "log": str(logs / (name + ".log"))}), flush=True)
        with (logs / (name + ".log")).open("a") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        done[name] = command
        temporary = completed.with_suffix(".tmp")
        temporary.write_text(json.dumps(done, indent=2) + "\n")
        temporary.replace(completed)

    def python(name, *arguments):
        run(name, sys.executable, *arguments)

    if not (args.model / "revision.txt").exists():
        from huggingface_hub import snapshot_download

        revision = "2fc06364715b967f1860aea9cf38778875588b17"
        snapshot_download("Qwen/Qwen3.5-0.8B", revision=revision, local_dir=args.model)
        (args.model / "revision.txt").write_text(revision + "\n")
    if not all(
        (args.data / (split + ".jsonl")).exists()
        for split in ["train", "dev", "calibration", "test"]
    ):
        python(
            "prepare-data", "scripts/prepare_data.py", "--output", args.data, "--model", args.model
        )
    frozen = {
        "recipe": recipe,
        "rlcd": asdict(policy),
        "source_hashes": {
            split: file_hash(args.data / (split + ".jsonl"))
            for split in ["train", "dev", "calibration", "test"]
        },
    }
    if args.initialize_from:
        frozen["initialized_from"] = {
            "checkpoint": str(args.initialize_from),
            "weights_sha256": file_hash(args.initialize_from / "adapter.safetensors"),
        }
    if previous is not None and previous != frozen:
        old = previous["recipe"]
        old_steps, old_epochs = old.get("steps"), old.get("epochs", EPOCHS)
        epochs_here = EPOCHS if epochs is None else epochs
        expected = {**previous, "recipe": {**old, "steps": steps, "epochs": epochs_here}}
        if expected != frozen:
            raise ValueError("An existing run permits only an increased step budget")
        # A budget is comparable when it is stated the same way; mixing a fixed
        # update count with an epoch budget is a different experiment.
        if steps is None and old_steps is None:
            raised = epochs_here > old_epochs
        elif steps is not None and old_steps is not None:
            raised = steps > old_steps
        else:
            raised = False
        if not raised:
            raise ValueError("An existing run permits only an increased step budget")
        # The run's own config records the update count the budget resolved to, so
        # the archived baseline is named after data actually seen, not intent.
        frozen_run = args.output / f"seed-{recipe['seed']}" / "config.json"
        if frozen_run.exists():
            archived = f"updates-{json.loads(frozen_run.read_text())['steps']}"
        elif old_steps is not None:
            archived = f"updates-{old_steps}"
        else:
            archived = f"epochs-{old_epochs}"
        baseline = args.output / "baselines" / archived
        if "report" not in done and not baseline.exists():
            raise ValueError("Complete the current workflow before extending its step budget")
        if baseline.exists() and json.loads((baseline / "recipe.json").read_text()) != previous:
            raise ValueError("The saved baseline belongs to a different recipe")
        if not baseline.exists():
            partial = baseline.with_suffix(".partial")
            shutil.copytree(
                args.output, partial, ignore=shutil.ignore_patterns("baselines"), dirs_exist_ok=True
            )
            partial.rename(baseline)
        affected = [
            f"train-{recipe['seed']}",
            f"evaluate-{recipe['seed']}",
            "accept-checkpoint",
            "audit-quality",
            "bench-dohnuts",
            "jevbench",
            "laya-chart",
            "report",
        ]
        for name in affected:
            done.pop(name, None)
        # These outputs are tied to the old checkpoint; retain their archived evidence.
        for name in ["acceptance", "quality-audit", "jevbench", "laya-chart", "metrics"]:
            path = args.output / name
            if path.exists():
                shutil.rmtree(path)
        (args.output / "benchmarks/dohnuts.jsonl").unlink(missing_ok=True)
        temporary = completed.with_suffix(".tmp")
        temporary.write_text(json.dumps(done, indent=2) + "\n")
        temporary.replace(completed)
    temporary = recipe_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(frozen, indent=2) + "\n")
    temporary.replace(recipe_path)
    seed = recipe["seed"]
    directory = args.output / f"seed-{seed}"
    config = args.output / f"recipe-seed-{seed}.json"
    config.write_text(json.dumps(recipe, indent=2) + "\n")
    resume = ["--resume"] if (directory / "last.pt").exists() else []
    initialize = ["--initialize-from", args.initialize_from] if args.initialize_from else []
    launcher = (
        [
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.gpus),
            "--master_port",
            str(args.master_port),
        ]
        if args.gpus > 1
        else []
    )
    if f"train-{seed}" not in done:
        python(
            f"train-{seed}",
            *launcher,
            "-m",
            "dohnuts.train",
            "train",
            "--config",
            config,
            "--run",
            directory,
            *resume,
            *initialize,
        )
    python(
        f"evaluate-{seed}",
        "-m",
        "dohnuts.train",
        "evaluate",
        "--config",
        config,
        "--run",
        directory,
    )
    checkpoint = args.output / "checkpoint"
    python(
        "accept-checkpoint",
        "tests/acceptance.py",
        "--checkpoint",
        checkpoint,
        "--output",
        args.output / "acceptance",
    )
    python(
        "audit-quality",
        "scripts/audit_checkpoint.py",
        "--checkpoint",
        checkpoint,
        "--data",
        args.data,
        "--output",
        args.output / "quality-audit",
    )
    from huggingface_hub import snapshot_download

    references = [
        (
            "laya",
            "https://github.com/NandhaKishorM/laya.git",
            "d113dca2512fb3eaca313534bc54c7162d87c1d4",
            "convaiinnovations/laya-multilingual",
            "052592a15d198d9ad47da779604259b10b47b7aa",
            ".cache/models/laya-multilingual",
        ),
        (
            "laya-vision",
            "https://github.com/r33drichards/laya-vision.git",
            "86ccec115ef3d72d1851168fc9b7194e8dbed35a",
            "thaitea/laya-vision-smolvlm-256m",
            "a2653db2831b9b03b875a6558618feda833b7ff0",
            ".cache/models/laya-vision-trained",
        ),
    ]
    for name, url, revision, model_id, model_revision, model_path in references:
        upstream = Path(".cache/upstream") / name
        if not upstream.exists():
            run("clone-" + name, "git", "clone", url, upstream)
        run("pin-" + name, "git", "-C", upstream, "checkout", "--detach", revision)
        if not (Path(model_path) / "revision.txt").exists():
            snapshot_download(model_id, revision=model_revision, local_dir=model_path)
            (Path(model_path) / "revision.txt").write_text(model_revision + "\n")
        python(
            "quality-" + name,
            "scripts/evaluate_upstream.py",
            name,
            "--data",
            args.data,
            "--output",
            args.output / "references" / name,
        )
    benchmarks = args.output / "benchmarks"
    benchmarks.mkdir(exist_ok=True)
    for engine in ["dohnuts", "laya", "laya-vision"]:
        extra = ["--checkpoint", checkpoint] if engine == "dohnuts" else []
        python(
            f"bench-{engine}",
            "scripts/benchmark.py",
            engine,
            "--output",
            benchmarks / f"{engine}.jsonl",
            *extra,
        )
    jevbench = Path(".cache/upstream/jevbench")
    if not jevbench.exists():
        run(
            "clone-jevbench",
            "git",
            "clone",
            "--branch",
            "v1.2.2",
            "--depth",
            "1",
            "https://github.com/fstandhartinger/jevbench.git",
            jevbench,
        )
    python(
        "jevbench",
        "scripts/run_jevbench.py",
        "--checkpoint",
        checkpoint,
        "--output",
        args.output / "jevbench",
        "--upstream",
        jevbench,
    )
    research = Path(".cache/upstream/laya-research")
    if not research.exists():
        run(
            "clone-laya-research",
            "git",
            "clone",
            "https://github.com/NandhaKishorM/laya.git",
            research,
        )
    run(
        "pin-laya-research",
        "git",
        "-C",
        research,
        "checkout",
        "--detach",
        "28d43add7e47ce502489c9433310d55276c64e0f",
    )
    suites = Path("data/benchmarks/laya")
    if not (suites / "suites.json").exists():
        python("prepare-laya-chart", "scripts/prepare_laya_benchmark.py", "--output", suites)
    python(
        "laya-chart",
        "scripts/run_laya_benchmark.py",
        "--checkpoint",
        checkpoint,
        "--data",
        suites,
        "--output",
        args.output / "laya-chart",
    )
    python("report", "scripts/report_results.py", "--run", args.output)
    print(
        json.dumps({"checkpoint": str(checkpoint), "metrics": str(args.output / "metrics")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
