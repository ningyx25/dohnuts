"""Parallel candidate scoring over shared text/image computation."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from dohnuts.adapters import Qwen35Adapter
from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH


def model_revision(directory):
    """Identify a downloaded Hub snapshot or a locally pinned training base."""
    directory = Path(directory)
    revision = directory / "revision.txt"
    if revision.is_file():
        return revision.read_text().strip()
    if directory.parent.name == "snapshots" and len(directory.name) == 40:
        return directory.name
    raise ValueError("The base model must be a pinned Hub snapshot or contain revision.txt")


class DecisionHead(nn.Module):
    """Two-projection candidate scoring against a separate decision query.

    The readout is ``dot(W_candidate h_candidate, W_decision h_decision) / sqrt(P)``,
    so the parameter count is independent of the candidate count and every
    projection stays in FP32 even under a BF16 backbone.
    """

    def __init__(self, hidden_size, projection_dim=256, *, device=None, dtype=None):
        super().__init__()
        self.decision = nn.Linear(
            hidden_size, projection_dim, bias=False, device=device, dtype=dtype
        )
        self.candidate = nn.Linear(
            hidden_size, projection_dim, bias=False, device=device, dtype=dtype
        )
        self.scale = math.sqrt(projection_dim)

    def forward(self, hidden: Tensor, positions: Tensor, decision_positions: Tensor) -> Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        decision = self.decision(hidden[rows, decision_positions].float())
        candidates = self.candidate(hidden[rows[:, None], positions].float())
        return (candidates * decision[:, None, :]).sum(-1) / self.scale


class DecisionModel(nn.Module):
    def __init__(self, checkpoint: str | Path, *, adapter=None, projection_dim=256):
        super().__init__()
        self.adapter = adapter or Qwen35Adapter()
        self.base_path = Path(checkpoint)
        self.processor = self.adapter.processor(checkpoint)
        self.backbone = self.adapter.load(checkpoint)
        self.projection_dim = projection_dim
        self.head = DecisionHead(
            self.adapter.hidden_size(self.backbone),
            projection_dim,
            device="cuda",
            dtype=torch.float32,
        )
        # Set by enable_stage; load_adapter only accepts a matching checkpoint.
        self.stage = "warmup"
        self.lora_rank = None
        self.lora_alpha = None
        self.marker_id = self.processor.tokenizer.convert_tokens_to_ids(self.adapter.marker)
        if self.processor.tokenizer.encode(self.adapter.marker, add_special_tokens=False) != [
            self.marker_id
        ]:
            raise ValueError("The candidate marker must be a single reserved token")

    def forward(
        self, inputs: dict[str, Tensor], positions: Tensor, decision_positions: Tensor
    ) -> Tensor:
        hidden, offset = self.adapter.forward(self.backbone, inputs)
        return self.score_hidden(
            hidden,
            (positions - offset).clamp_min(0),
            (decision_positions - offset).clamp_min(0),
        )

    def score_hidden(self, hidden: Tensor, positions: Tensor, decision_positions: Tensor) -> Tensor:
        return self.head(hidden, positions, decision_positions)

    def enable_stage(self, stage, *, lora_rank=8, lora_alpha=16, checkpointing=True):
        """Open the parameters a training stage owns, then adapt the language model."""
        self.adapter.apply_stage(
            self.backbone,
            stage,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training=checkpointing,
        )
        self.stage = stage
        # Recorded even for warmup, which ignores them, so a checkpoint can be
        # checked against the recipe that produced it.
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def load_adapter(self, directory):
        """Load the same verified unmerged weights for learning or deployment."""
        from safetensors.torch import load_file

        directory = Path(directory)
        metadata = json.loads((directory / "dohnuts.json").read_text())
        weights = directory / "adapter.safetensors"
        version = metadata.get("format_version")
        if version != 2:
            raise ValueError(
                "Checkpoint format_version 1 or unknown is not loadable: the decision head changed "
                "from one Linear to two projections, so its parameter keys differ. Retrain, or "
                "export again with --initialize-from on a format_version 2 recipe."
            )
        differing = [
            name
            for name, found, expected in (
                ("adapter", metadata.get("adapter", "qwen3.5"), self.adapter.name),
                ("base_revision", metadata.get("base_revision"), model_revision(self.base_path)),
                ("image_pixels", metadata.get("image_pixels"), IMAGE_PIXELS),
                ("max_length", metadata.get("max_length"), MAX_LENGTH),
                ("stage", metadata.get("stage"), self.stage),
                ("projection_dim", metadata.get("projection_dim"), self.projection_dim),
                ("lora_rank", metadata.get("lora_rank"), self.lora_rank),
                ("lora_alpha", metadata.get("lora_alpha"), self.lora_alpha),
                (
                    "weights_sha256",
                    hashlib.sha256(weights.read_bytes()).hexdigest(),
                    metadata.get("weights_sha256"),
                ),
            )
            if found != expected
        ]
        if differing:
            raise ValueError(
                "Checkpoint differs from the pinned model, training recipe, or checksum: "
                + ", ".join(differing)
            )
        state = load_file(str(weights))
        parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if state.keys() != parameters.keys() or any(
            state[name].shape != parameter.shape for name, parameter in parameters.items()
        ):
            raise ValueError("Checkpoint trainable parameters differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name])
        return metadata

    def merge(self):
        self.adapter.merge(self.backbone)
        # Inference may cache image features again: the visual tower now holds
        # the final weights and no longer moves.
        self.adapter._vision_trainable = False
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Only switches the vision tower to eval; it must never clear the
        # gradients a joint or vision_top stage opened.
        self.adapter.freeze_vision(self.backbone)
        return self


def marker_positions(input_ids: Tensor, marker_id: int, candidates: int) -> Tensor:
    positions = []
    for row in input_ids:
        found = (row == marker_id).nonzero(as_tuple=True)[0]
        if found.numel() != candidates:
            raise ValueError(f"Expected {candidates} candidate markers, found {found.numel()}")
        positions.append(found)
    return torch.stack(positions)
