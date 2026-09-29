"""Single-GPU smoke test for the Valen core migration.

Against the real Qwen3.5-0.8B backbone this checks the four stage parameter
sets and their per-group gradients, the two-projection head numerics, the image
feature cache bypass while the vision tower is open, two SFT and two GRPO
update steps with a fixed reference, a checkpoint round trip, and a merged
prediction that is still a legal distribution.

Run: pdm run python scripts/smoke_valen_migration.py
"""

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from dohnuts.model import DecisionHead, DecisionModel
from dohnuts.objectives import GRPOObjective, SFTObjective
from dohnuts.predictor import Predictor
from dohnuts.recipe import BASE_MODEL, STAGES
from dohnuts.train import load_checkpoint, save_checkpoint
from dohnuts.training_data import DecisionCollator, to_gpu


def report(check, **fields):
    print(json.dumps({"check": check, **fields}), flush=True)


def group_of(name):
    if name.startswith("head."):
        return "head"
    if "lora_" in name:
        return "lora"
    if ".visual.merger." in name:
        return "merger"
    if ".visual.blocks." in name:
        return "vision"
    return None


def gradient_norms(model):
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = group_of(name)
        if key is None:
            raise AssertionError(f"Unclassified trainable parameter: {name}")
        norms[key] = norms.get(key, 0.0) + float(parameter.grad.detach().norm()) ** 2
    return {key: value**0.5 for key, value in norms.items()}


def synthetic_rows(root):
    image = root / "screen.png"
    Image.new("RGB", (240, 320), "white").save(image)
    return [
        {
            "id": "smoke:image",
            "dataset": "smoke",
            "group": "smoke",
            "state": {"screen": "settings page"},
            "image": str(image),
            "question": {"type": "choice", "instructions": "pick", "criteria": ["alpha", "beta"]},
            "target": [1.0, 0.0],
        },
        {
            "id": "smoke:text",
            "dataset": "smoke",
            "group": "smoke",
            "state": {"screen": "settings page"},
            "question": {"type": "noul", "criteria": {"false": "no", "true": "yes"}},
            "target": [0.0, 1.0],
        },
    ]


def check_head_numerics():
    torch.manual_seed(0)
    head = DecisionHead(16, 4, device="cuda", dtype=torch.float32)
    hidden = torch.randn(2, 6, 16, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([[0, 2], [1, 5]], device="cuda")
    decisions = torch.tensor([3, 4], device="cuda")
    logits = head(hidden, positions, decisions)
    rows = torch.arange(2, device="cuda")
    expected = (
        head.candidate(hidden[rows[:, None], positions].float())
        * head.decision(hidden[rows, decisions].float())[:, None, :]
    ).sum(-1) / head.scale
    torch.testing.assert_close(logits, expected)
    logits.sum().backward()
    assert head.decision.weight.grad.abs().sum() > 0
    assert head.candidate.weight.grad.abs().sum() > 0
    assert set(head.state_dict()) == {"decision.weight", "candidate.weight"}
    report("head", matches_closed_form=True, fp32=logits.dtype == torch.float32)


def check_stage(model_path, batch, stage):
    model = DecisionModel(model_path)
    model.enable_stage(stage, lora_rank=8, lora_alpha=16, checkpointing=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    unknown = sorted(n for n in trainable if group_of(n) is None)
    assert not unknown, unknown
    blocks = sorted(
        {n.split(".visual.blocks.")[1].split(".")[0] for n in trainable if ".visual.blocks." in n},
        key=int,
    )
    assert any("lora_" in n for n in trainable) is (stage != "warmup")
    assert bool([n for n in trainable if ".visual.merger." in n]) is (
        stage in {"joint", "vision_top"}
    )
    assert len(blocks) == (4 if stage == "vision_top" else 0)
    assert model.adapter._vision_trainable is (stage in {"joint", "vision_top"})
    inputs, positions, decision_positions, mask, target, ordinal = batch
    model.train()
    if model.adapter._vision_trainable:
        assert not model.backbone._dohnuts_image_cache, (
            "the open vision tower must bypass the cache"
        )
    logits = model(inputs, positions, decision_positions)
    loss, _ = SFTObjective({}).loss(logits, target, mask, ordinal)
    loss.backward()
    norms = gradient_norms(model)
    if stage == "warmup":
        assert set(norms) == {"head"}
    else:
        assert set(norms) >= {"head", "lora"}
    if stage in {"joint", "vision_top"}:
        assert norms["merger"] > 0
        assert not model.backbone._dohnuts_image_cache
    if stage == "vision_top":
        assert norms["vision"] > 0
    report("stage", stage=stage, trainable=len(trainable), vision_blocks=blocks, grads=norms)
    return model


def check_steps(model, objective, batch, *, steps, lr):
    inputs, positions, decision_positions, mask, target, ordinal = batch
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    model.train()
    losses, metrics = [], {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, positions, decision_positions)
        loss, metrics = objective.loss(logits, target, mask, ordinal)
        assert torch.isfinite(loss), f"non-finite loss: {loss}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        assert all(torch.isfinite(value).all() for value in metrics.values())
    return losses, {key: float(value) for key, value in metrics.items()}


def check_sft(model_path, batch):
    model = check_stage(model_path, batch, "warmup")
    before = model.head.decision.weight.detach().clone()
    losses, metrics = check_steps(
        model, SFTObjective({"rps_weight": 1.0, "brier_weight": 1.0}), batch, steps=2, lr=5e-4
    )
    moved = not torch.equal(before, model.head.decision.weight.detach())
    assert moved, "two SFT steps left the decision head untouched"
    report("sft", losses=losses, metrics=metrics, head_moved=moved)
    del model


def check_grpo(model_path, batch, run_dir):
    model = check_stage(model_path, batch, "warmup")
    options = {"group_size": 4, "num_iterations": 2, "beta": 0.02}
    objective = GRPOObjective(options, model=model, resuming=False)
    reference = dict(objective.reference.named_parameters())
    before = {name: reference[name].detach().clone() for name in objective.reference_weights}
    inputs, positions, decision_positions, mask, target, ordinal = batch
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.01
    )
    model.train()
    with torch.no_grad():
        rollout = objective.prepare(model(inputs, positions, decision_positions), target, mask)
    reference_logits = objective.reference_logits(inputs, positions, decision_positions)
    losses, clip_fraction = [], 0.0
    for _ in range(objective.num_iterations):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, positions, decision_positions)
        loss, metrics = objective.loss(logits, target, mask, ordinal, rollout, reference_logits)
        assert torch.isfinite(loss), loss
        assert all(torch.isfinite(value).all() for value in metrics.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        clip_fraction = float(metrics["clip_fraction"])
    # The second iteration updates from the same rollout against a moved policy,
    # so it is a different objective value: the frozen rollout is real.
    assert losses[0] != losses[1], losses
    for name, value in before.items():
        assert torch.equal(value, reference[name]), name
    save_checkpoint(
        run_dir / "last.pt", model, optimizer, 2, {"stub": True}, 0.0, objective.state_dict()
    )
    fresh = DecisionModel(model_path)
    fresh.enable_stage("warmup", lora_rank=8, lora_alpha=16, checkpointing=False)
    state = load_checkpoint(fresh, run_dir / "last.pt")
    saved = dict(model.named_parameters())
    for name, parameter in fresh.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, saved[name].detach(), rtol=0, atol=0)
    resumed = GRPOObjective(
        options,
        model=fresh,
        training_state=state["training_state"],
        resuming=True,
    )
    for name, value in objective.reference_weights.items():
        torch.testing.assert_close(resumed.reference_weights[name], value, rtol=0, atol=0)
    report(
        "grpo",
        losses=losses,
        clip_fraction=clip_fraction,
        reference_frozen=True,
        checkpoint_resume_equivalent=True,
    )
    del resumed, fresh, objective, model
    gc.collect()
    torch.cuda.empty_cache()


def check_prediction(model_path):
    model = DecisionModel(model_path)
    model.enable_stage("text", lora_rank=8, lora_alpha=16, checkpointing=False)
    model.merge()
    assert not model.adapter._vision_trainable
    answer = Predictor(model).predict(
        {"image": Image.new("RGB", (240, 320), "white"), "screen": "settings page"},
        {"q": {"type": "choice", "instructions": "pick", "criteria": ["alpha", "beta"]}},
    )
    probabilities = answer["answers"]["q"]["probabilities"]
    assert set(probabilities) == {"alpha", "beta"}
    total = sum(probabilities.values())
    assert abs(total - 1.0) < 1e-4, total
    assert model.backbone._dohnuts_image_cache, "frozen vision should cache its features"
    report("prediction", probabilities=probabilities, usage=answer["usage"])
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_MODEL, help="Local pinned backbone")
    args = parser.parse_args()
    if not (args.model / "revision.txt").is_file():
        raise SystemExit(f"A pinned snapshot with revision.txt is required: {args.model}")
    if not torch.cuda.is_available():
        raise SystemExit("This smoke test needs one CUDA device")
    report("backbone", model=str(args.model), device=torch.cuda.get_device_name())

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "run"
        run_dir.mkdir()
        batch = to_gpu(DecisionCollator(args.model)(synthetic_rows(root)))
        check_head_numerics()
        for stage in STAGES:
            model = check_stage(args.model, batch, stage)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        check_sft(args.model, batch)
        gc.collect()
        torch.cuda.empty_cache()
        check_grpo(args.model, batch, run_dir)
        check_prediction(args.model)
    report("complete", stages=list(STAGES))


if __name__ == "__main__":
    main()
