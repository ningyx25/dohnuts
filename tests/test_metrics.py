import hashlib
import json
import math

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from dohnuts.metrics import by_dataset, fit_temperatures, summarize


def test_twenty_latency_samples_do_not_report_maximum_as_p95():
    from dohnuts.experiment import latency_stats

    result = latency_stats(list(range(1, 21)), 1)
    assert result["p95_ms"] == pytest.approx(19.05, rel=0, abs=5e-8)


def test_known_binary_distribution():
    rows = [
        {"type": "noul", "logits": [0.0, math.log(4)], "target": t}
        for t in [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    ]
    result = summarize(rows)
    assert result["accuracy"] == pytest.approx(0.75, rel=0, abs=5e-8)
    assert result["ece_15"] == pytest.approx(0.05, rel=0, abs=5e-8)
    expected_nll = -(0.75 * math.log(0.8) + 0.25 * math.log(0.2))
    assert result["nll"] == pytest.approx(expected_nll, rel=0, abs=5e-8)
    assert result["brier_sum"] == pytest.approx(0.38, rel=0, abs=5e-8)
    assert result["brier_per_candidate"] == pytest.approx(0.19, rel=0, abs=5e-8)
    empty_bins = [row for row in result["reliability"] if row["n"] == 0]
    assert len(empty_bins) == 14
    for row in empty_bins:
        assert row["confidence"] is None
        assert row["accuracy"] is None


def test_temperature_fits_soft_targets_without_changing_order():
    rows = [{"type": "noul", "logits": [0.0, 4.0], "target": [0.25, 0.75]} for _ in range(20)]
    temperatures = fit_temperatures(rows)
    assert temperatures["noul"] == pytest.approx(4 / math.log(3), rel=0, abs=0.02)
    assert summarize(rows, temperatures)["nll"] < summarize(rows)["nll"]


def test_by_dataset_suppresses_index_f1_for_mutable_vocabularies():
    # The element candidates of `screenshot_choice` differ per row, so index 0 of
    # one row is not index 0 of the next; a fixed vocabulary keeps its macro-F1.
    rows = [
        {
            "dataset": "screenshot_choice",
            "type": "choice",
            "logits": [2.0, 0.0],
            "target": [1.0, 0.0],
        },
        {
            "dataset": "screenshot_choice",
            "type": "choice",
            "logits": [0.0, 2.0],
            "target": [0.0, 1.0],
        },
        {"dataset": "gui_button", "type": "choice", "logits": [2.0, 0.0], "target": [1.0, 0.0]},
        {"dataset": "gui_button", "type": "choice", "logits": [0.0, 2.0], "target": [0.0, 1.0]},
        # The mobile-jev operation question offers a per-screen subset of a fixed
        # order, so its label indices shift between rows as well.
        {"dataset": "jev_operation", "type": "choice", "logits": [2.0, 0.0], "target": [1.0, 0.0]},
        {"dataset": "jev_operation", "type": "choice", "logits": [0.0, 2.0], "target": [0.0, 1.0]},
    ]
    result = by_dataset(rows)
    assert "macro_f1" not in result["screenshot_choice"]
    assert "macro_f1" not in result["jev_operation"]
    assert result["gui_button"]["macro_f1"] == pytest.approx(1.0, rel=0, abs=5e-8)


def test_ordinal_distance_and_soft_accuracy():
    result = summarize(
        [{"type": "score", "logits": [-100.0, -100.0, 0.0], "target": [1.0, 0.0, 0.0]}]
    )
    assert result["rps"] == pytest.approx(1.0, rel=0, abs=5e-8)
    assert result["score_mae"] == pytest.approx(2.0, rel=0, abs=5e-8)
    assert result["soft_accuracy"] == 0.0


class TrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 2)
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)


class StubAdapter:
    name = "qwen3.5"


class DecisionModelStub(nn.Module):
    """Enough of DecisionModel for load_adapter validation, without a backbone."""

    def __init__(self):
        super().__init__()
        self.adapter = StubAdapter()
        self.head = nn.Linear(3, 2)
        self.stage = "text"
        self.projection_dim = 256
        self.lora_rank = 8
        self.lora_alpha = 16
        self.base_path = None


@pytest.fixture
def cpu_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)


def test_checkpoint_round_trip_keeps_the_reference_state(tmp_path, cpu_rng):
    from dohnuts.train import load_checkpoint, save_checkpoint

    model = TrainableModel()
    optimizer = torch.optim.AdamW([model.head.weight], lr=0.1)
    training_state = {"reference_weights": {"head.weight": torch.zeros(2, 3)}}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 7, {"seed": 42}, 0.5, training_state)
    expected = model.head.weight.detach().clone()
    with torch.no_grad():
        model.head.weight.add_(1.0)
    state = load_checkpoint(model, path, optimizer)
    assert state["format_version"] == 2
    assert state["step"] == 7
    torch.testing.assert_close(model.head.weight, expected)
    restored = state["training_state"]["reference_weights"]
    assert set(restored) == {"head.weight"}
    torch.testing.assert_close(restored["head.weight"], torch.zeros(2, 3))


def test_resume_refuses_a_single_linear_head_checkpoint(tmp_path):
    from dohnuts.train import load_checkpoint

    model = TrainableModel()
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 1,
            "config": {},
            "dev_macro_accuracy": 0.0,
            "trainable": {n: p.detach() for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="format_version 1"):
        load_checkpoint(model, path)


def make_export(tmp_path, model, *, format_version=2, **overrides):
    from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH

    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "revision.txt").write_text("abc123\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    weights = checkpoint / "adapter.safetensors"
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in model.named_parameters()
            if value.requires_grad
        },
        str(weights),
    )
    metadata = {
        "format_version": format_version,
        "adapter": "qwen3.5",
        "base_revision": "abc123",
        "image_pixels": IMAGE_PIXELS,
        "max_length": MAX_LENGTH,
        "stage": model.stage,
        "projection_dim": model.projection_dim,
        "lora_rank": model.lora_rank,
        "lora_alpha": model.lora_alpha,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    metadata.update(overrides)
    (checkpoint / "dohnuts.json").write_text(json.dumps(metadata))
    model.base_path = base
    return checkpoint, metadata


def test_export_metadata_must_match_the_loaded_stage(tmp_path):
    from dohnuts.model import DecisionModel

    model = DecisionModelStub()
    checkpoint, metadata = make_export(tmp_path, model)
    assert DecisionModel.load_adapter(model, checkpoint) == metadata
    original = dict(metadata)
    for field, value in (
        ("stage", "warmup"),
        ("projection_dim", 512),
        ("lora_rank", 16),
        ("lora_alpha", 32),
        ("base_revision", "other"),
    ):
        metadata.update(original)
        metadata[field] = value
        (checkpoint / "dohnuts.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="pinned model"):
            DecisionModel.load_adapter(model, checkpoint)


def test_format_version_1_export_is_rejected_by_both_loaders(tmp_path):
    from dohnuts.model import DecisionModel
    from dohnuts.predictor import Predictor

    model = DecisionModelStub()
    checkpoint, _ = make_export(tmp_path, model, format_version=1)
    with pytest.raises(ValueError, match="format_version 1"):
        DecisionModel.load_adapter(model, checkpoint)
    with pytest.raises(ValueError, match="format_version 1"):
        Predictor.from_checkpoint(checkpoint)
