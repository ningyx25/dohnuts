"""Stage-owned parameters, enumerated LoRA targets, and the image-cache bypass."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dohnuts import adapters
from dohnuts.adapters import Qwen35Adapter


class StubLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.q_proj = nn.Linear(4, 4)
            attn.o_proj = nn.Linear(4, 4)
            attn.in_proj_qkv = nn.Linear(4, 4)
            attn.out_proj = nn.Linear(4, 4)
            layer.self_attn = attn
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(4, 4)
            layer.mlp = mlp
            self.layers.append(layer)


class StubVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.merger = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = StubLanguage()
        self.visual = StubVisual()
        self._dohnuts_image_cache = {}


class FakePeft(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.base = module
        self.peft_config = config

    def merge_and_unload(self, safe_merge=True):
        return self.base


@pytest.fixture
def lora_calls(monkeypatch):
    calls = []

    def fake_get_peft_model(module, config):
        calls.append(config)
        return FakePeft(module, config)

    monkeypatch.setattr(adapters, "get_peft_model", fake_get_peft_model)
    return calls


def apply(backbone, stage, **overrides):
    Qwen35Adapter().apply_stage(
        backbone, stage, lora_rank=8, lora_alpha=16, training=False, **overrides
    )


def test_warmup_opens_only_the_decision_head(lora_calls):
    backbone = StubBackbone()
    adapter = Qwen35Adapter()
    adapter.apply_stage(backbone, "warmup", lora_rank=8, lora_alpha=16, training=False)
    assert lora_calls == []
    assert not any(p.requires_grad for p in backbone.parameters())
    assert adapter._vision_trainable is False


def test_text_stage_enumerates_every_language_linear(lora_calls):
    backbone = StubBackbone()
    apply(backbone, "text")
    config = lora_calls[0]
    expected = sorted(
        name for name, module in StubLanguage().named_modules() if isinstance(module, nn.Linear)
    )
    assert sorted(config.target_modules) == expected
    assert "layers.0.self_attn.in_proj_qkv" in config.target_modules
    assert "layers.1.mlp.gate_proj" in config.target_modules
    assert all("visual" not in name for name in config.target_modules)
    assert (config.r, config.lora_alpha, config.lora_dropout, config.bias) == (8, 16, 0.0, "none")
    assert not backbone.visual.merger.weight.requires_grad
    assert not backbone.visual.blocks[0].weight.requires_grad


@pytest.mark.parametrize(
    ("stage", "merger", "blocks"),
    [
        ("warmup", False, []),
        ("text", False, []),
        ("joint", True, []),
        ("vision_top", True, [2, 3, 4, 5]),
    ],
)
def test_stage_owns_exactly_its_parameters(lora_calls, stage, merger, blocks):
    backbone = StubBackbone()
    backbone._dohnuts_image_cache["stale"] = torch.zeros(1)
    adapter = Qwen35Adapter()
    adapter.apply_stage(backbone, stage, lora_rank=8, lora_alpha=16, training=False)
    assert backbone.visual.merger.weight.requires_grad is merger
    open_blocks = [
        index for index, block in enumerate(backbone.visual.blocks) if block.weight.requires_grad
    ]
    assert open_blocks == blocks
    assert adapter._vision_trainable is (stage in {"joint", "vision_top"})
    if adapter._vision_trainable:
        assert backbone._dohnuts_image_cache == {}
    else:
        assert "stale" in backbone._dohnuts_image_cache


def test_train_mode_does_not_refreeze_open_vision(lora_calls):
    backbone = StubBackbone()
    adapter = Qwen35Adapter()
    adapter.apply_stage(backbone, "joint", lora_rank=8, lora_alpha=16, training=False)
    # DecisionModel.train() calls exactly this and must not clear requires_grad.
    adapter.freeze_vision(backbone)
    assert backbone.visual.merger.weight.requires_grad
    assert not backbone.visual.training


def test_merge_restores_the_plain_language_model(lora_calls):
    for stage in ("warmup", "text"):
        backbone = StubBackbone()
        adapter = Qwen35Adapter()
        adapter.apply_stage(backbone, stage, lora_rank=8, lora_alpha=16, training=False)
        adapter.merge(backbone)
        assert isinstance(backbone.language_model, StubLanguage)


def test_applying_a_stage_twice_is_rejected(lora_calls):
    adapter = Qwen35Adapter()
    backbone = StubBackbone()
    apply(backbone, "text")
    with pytest.raises(ValueError, match="already adapted"):
        adapter.apply_stage(backbone, "joint", lora_rank=8, lora_alpha=16, training=False)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unsupported training stage"):
        Qwen35Adapter().apply_stage(
            StubBackbone(), "full", lora_rank=8, lora_alpha=16, training=False
        )


class ForwardBackbone(nn.Module):
    """Enough of the backbone surface for Qwen35Adapter.forward."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.language_model = nn.Identity()
        self.visual = StubVisual()
        self._dohnuts_image_cache = {}
        self.image_encodings = 0

    def get_input_embeddings(self):
        return self.embed

    def get_placeholder_mask(self, input_ids, inputs_embeds=None, image_features=None):
        mask = input_ids == 2
        return mask.unsqueeze(-1).expand_as(inputs_embeds), None

    def compute_3d_position_ids(self, **kwargs):
        return None

    def get_image_features(self, pixel_values, grid, return_dict=True):
        self.image_encodings += 1
        # The real pooled output is one tensor per image, in image order.
        return SimpleNamespace(
            pooler_output=tuple(torch.zeros(1, 4) for _ in range(pixel_values.shape[0]))
        )


def image_inputs():
    input_ids = torch.tensor([[0, 2, 3, 4, 5], [0, 2, 3, 6, 0]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(2, 16),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "image_patches": [4, 4],
        "image_keys": ("first", "second"),
    }


@pytest.fixture
def cached_features(monkeypatch):
    calls = []

    def fake_image_features(self, backbone, inputs):
        calls.append(inputs["image_keys"])
        return torch.zeros(len(inputs["image_keys"]), 4)

    monkeypatch.setattr(Qwen35Adapter, "image_features", fake_image_features)
    monkeypatch.setattr(
        adapters, "language_forward", lambda language, embeddings, *rest: embeddings
    )
    return calls


def test_frozen_vision_reuses_the_feature_cache(cached_features):
    backbone = ForwardBackbone()
    adapter = Qwen35Adapter()
    adapter._vision_trainable = False
    hidden, _ = adapter.forward(backbone, image_inputs())
    assert len(cached_features) == 1
    assert backbone.image_encodings == 0
    assert hidden.shape == (2, 5, 4)


def test_trainable_vision_bypasses_the_feature_cache(cached_features):
    backbone = ForwardBackbone()
    adapter = Qwen35Adapter()
    adapter._vision_trainable = True
    hidden, _ = adapter.forward(backbone, image_inputs())
    assert cached_features == []
    assert backbone.image_encodings == 1
    assert hidden.shape == (2, 5, 4)
