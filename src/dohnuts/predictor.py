"""Candidate selection, truth estimates, and ordered scores over shared inputs."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from dohnuts import distributed
from dohnuts.adapters import Qwen35Adapter
from dohnuts.execution import plan_prefix
from dohnuts.gui_data import MAX_CANDIDATES, MIN_CANDIDATES
from dohnuts.model import DecisionModel, marker_positions
from dohnuts.recipe import BASE_MODEL, IMAGE_PIXELS

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options_for(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        if not isinstance(criteria, Mapping):
            raise ValueError("noul criteria must map false/true to descriptions")
        labels = ["false", "true"]
        options = [
            "false: " + render(criteria.get("false", "no, the statement does not hold")),
            "true: " + render(criteria.get("true", "yes, the statement holds")),
        ]
    elif kind == "choice":
        if isinstance(criteria, list):
            labels = [str(value) for value in criteria]
            options = [render(value) for value in criteria]
        elif isinstance(criteria, Mapping):
            labels = list(criteria)
            options = [
                str(key) if value is None or value == "" else f"{key}: {render(value)}"
                for key, value in criteria.items()
            ]
        else:
            raise ValueError("choice requires a candidate list or mapping")
        if len(set(labels)) != len(labels):
            raise ValueError("Candidate labels must be unique")
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score requires an ordered list of level descriptions")
        labels = [str(i) for i in range(len(criteria))]
        options = [f"level {i}: {render(value)}" for i, value in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported decision type: {kind}")
    if not MIN_CANDIDATES <= len(options) <= MAX_CANDIDATES:
        raise ValueError(
            f"Each question requires {MIN_CANDIDATES}–{MAX_CANDIDATES} candidates"
        )
    return labels, options


def render_question(state_text, question, *, has_image=False, adapter=None):
    """One prompt template shared by training, evaluation, and serving.

    Instructions may be a string or, as the mobile-jev rows have it, an object;
    `render` keeps a string as it is and serialises anything else the way the
    state is serialised, so a `{'goal', 'rules'}` pair reaches the prompt as the
    JSON the agent would have sent.

    Also returns the character offset at which the question stem ends. That
    boundary locates the decision readout: the last token starting before it
    cannot describe any candidate option.
    """
    adapter = adapter or Qwen35Adapter()
    marker = adapter.marker
    labels, options = options_for(question)
    stem = (
        f"State: {state_text}\n{question['type']} question: "
        f"{render(question.get('instructions', ''))}\nOptions:\n"
    )
    if marker in stem or any(marker in option for option in options):
        raise ValueError("Input contains the reserved candidate marker")
    prefix = adapter.image_prefix if has_image else ""
    content = prefix + stem + "".join(f"- {option}{marker}" for option in options)
    return content, labels, len(prefix) + len(stem)


def _stem_end_token(offsets, stem_end):
    """The last token to prefer is one fully inside the stem; never a candidate."""
    inside = [i for i, (start, end) in enumerate(offsets) if end > start and end <= stem_end]
    if inside:
        return inside[-1]
    containing = [i for i, (start, end) in enumerate(offsets) if end > start and start < stem_end]
    if not containing:
        raise ValueError("The question stem produced no decision token")
    return containing[-1]


def _placeholder_shift(raw_ids, row_ids, image_token_id):
    """The processor replaces one image placeholder with one token per patch."""
    if image_token_id is None:
        return 0
    raw_index = next((i for i, token in enumerate(raw_ids) if token == image_token_id), None)
    if raw_index is None:
        return 0
    expanded = (row_ids == image_token_id).nonzero(as_tuple=True)[0]
    if not expanded.numel():
        raise ValueError("The rendered prompt lost its image placeholder")
    return int(expanded[0]) - raw_index + int(expanded.numel()) - 1


def locate_decision_positions(processor, texts, stem_ends, *, input_ids):
    """Map every stem end to the token that carries the decision query.

    Character offsets come from the fast tokenizer. The processor then expands
    the single image placeholder into one token per visual patch, shifting every
    later index, so the expansion is subtracted back out. Right padding keeps a
    row's index valid inside a batch.
    """
    tokenizer = processor.tokenizer
    if tokenizer.padding_side != "right":
        raise ValueError("The decision readout requires a right-padded tokenizer")
    image_token_id = getattr(processor, "image_token_id", None)
    positions = []
    for row, (text, stem_end) in enumerate(zip(texts, stem_ends, strict=True)):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        raw_index = _stem_end_token(encoded["offset_mapping"], stem_end)
        index = raw_index + _placeholder_shift(encoded["input_ids"], input_ids[row], image_token_id)
        if int(input_ids[row, index]) != encoded["input_ids"][raw_index]:
            raise ValueError("The decision position drifted from the rendered prompt")
        positions.append(index)
    return torch.tensor(positions, dtype=torch.long)


class Predictor:
    metadata: dict[str, Any]

    def __init__(self, model: DecisionModel):
        self.model = model
        self.image_pixels = IMAGE_PIXELS
        self.max_length = model.adapter.max_input_tokens
        self.temperatures = [1.0, 1.0, 1.0]

    @classmethod
    def from_checkpoint(cls, directory, *, revision=None, base_model=None, adapter=None):
        """Load a local export or Hub repository and merge its calibrated decision model."""
        source = directory
        directory = Path(source).expanduser()
        if not directory.is_dir():
            if isinstance(source, Path) or str(source).startswith(("/", ".", "~")):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
            directory = Path(
                snapshot_download(
                    str(source),
                    revision=revision,
                    allow_patterns=["dohnuts.json", "adapter.safetensors", "LICENSE"],
                )
            )
        config = json.loads((directory / "dohnuts.json").read_text())
        if config.get("format_version") != 2:
            raise ValueError(
                "Checkpoint format_version 1 or unknown is not loadable: the decision head changed "
                "from one Linear to two projections. Evaluate published versions from the commit "
                "that produced them, or retrain with the current recipe."
            )
        if base_model is None:
            base_model = Path(config.get("base_path", BASE_MODEL)).expanduser()
            if not base_model.is_dir():
                base_model = snapshot_download(
                    config["base_model"], revision=config["base_revision"]
                )
        base_model = Path(base_model).expanduser()
        model = DecisionModel(
            base_model, adapter=adapter, projection_dim=config.get("projection_dim", 256)
        )
        model.enable_stage(
            config["stage"],
            lora_rank=config.get("lora_rank") or 8,
            lora_alpha=config.get("lora_alpha") or 16,
            checkpointing=False,
        )
        config = model.load_adapter(directory)
        model.merge()
        predictor = cls(model)
        predictor.metadata = config
        predictor.temperatures = [config["temperatures"][kind] for kind in QTYPES]
        return predictor

    def prepare(self, state, questions):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        image = None
        if isinstance(state, Mapping):
            image = state.get("image")
            if "images" in state:
                raise ValueError("Pass one decoded PIL image via state['image']")
            state_text = render({key: value for key, value in state.items() if key != "image"})
        else:
            state_text = render(state)
        if image is not None and not isinstance(image, Image.Image):
            raise ValueError("Decode the image as a PIL image before passing it in state['image']")
        texts, stem_ends, metadata = [], [], []
        for qid, question in questions.items():
            content, labels, stem_end = render_question(
                state_text, question, has_image=image is not None, adapter=self.model.adapter
            )
            texts.append(content)
            stem_ends.append(stem_end)
            metadata.append((qid, question["type"], labels))
        if image is not None:
            inputs = self.model.adapter.shared_image_inputs(self.model.processor, texts, image)
        else:
            inputs = self.model.adapter.batch_inputs(self.model.processor, texts, [])
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Input exceeds the token budget; no question or candidate was truncated"
            )
        decision_positions = locate_decision_positions(
            self.model.processor, texts, stem_ends, input_ids=inputs["input_ids"]
        )
        maximum = max(len(labels) for _, _, labels in metadata)
        positions = torch.zeros(len(texts), maximum, dtype=torch.long)
        mask = torch.zeros_like(positions, dtype=torch.bool)
        for row, (_, _, labels) in enumerate(metadata):
            positions[row, : len(labels)] = marker_positions(
                inputs["input_ids"][row : row + 1],
                self.model.marker_id,
                len(labels),
            )[0]
            mask[row, : len(labels)] = True
        plan_prefix(inputs, positions)
        return inputs, positions, decision_positions, mask, metadata

    @torch.inference_mode()
    def predict(self, state, questions):
        self.model.eval()
        inputs, positions, decision_positions, mask, metadata = self.prepare(state, questions)
        token_count = int(inputs["attention_mask"].sum())
        has_image = "pixel_values" in inputs
        target = distributed.device()
        inputs = {
            key: value.to(target) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        positions = positions.to(target)
        decision_positions = decision_positions.to(target)
        logits = (
            self.model(inputs, positions, decision_positions).cpu().masked_fill(~mask, -torch.inf)
        )
        answers = {}
        for row, (qid, kind, labels) in enumerate(metadata):
            temperature = self.temperatures[QTYPES[kind]]
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Calibration temperatures must be positive and finite")
            probabilities = (logits[row, : len(labels)] / temperature).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model produced a non-finite distribution")
            values = probabilities.tolist()
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in values)
            answer = {
                "type": kind,
                "confidence": max(0.0, min(1.0, 1 - entropy / math.log(len(labels)))),
            }
            if kind == "noul":
                answer["noul"] = values[1]
                answer["confidence"] = max(values)
            else:
                answer["probabilities"] = dict(zip(labels, values, strict=True))
                if kind == "choice":
                    answer["choice"] = labels[int(probabilities.argmax())]
                else:
                    answer["score"] = sum(i * p for i, p in enumerate(values))
            answers[qid] = answer
        return {
            "model": "dohnuts",
            "answers": answers,
            "usage": {"input_tokens": token_count, "images": int(has_image)},
        }
