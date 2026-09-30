"""Token length of a decision row, measured the way the collator will see it.

The conversion deliberately does not gate on the token budget: the budget is
redefined from the whole corpus once it has been measured, not guessed from a
subset. This module is what measures it -- the prompt a row renders to, plus the
image placeholders the processor expands, which is exactly what
`dohnuts.training_data.DecisionCollator` compares against `MAX_LENGTH`.

It reads rows rather than steps, so the count is of what was actually written.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS


def measure(processor, row: dict) -> int:
    """Rendered tokens plus expanded image placeholders for one row.

    The image term is the number of visual patches the processor substitutes for
    the single image placeholder, minus that placeholder itself: the prompt
    renders one `<image>` token and the collator sees one token per patch.
    """
    prompt, _, _ = render_question(render(row["state"]), row["question"], has_image=True)
    length = len(processor.tokenizer(prompt, truncation=False)["input_ids"])
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    with Image.open(row["image"]) as image:
        width, height = smart_resize(
            image.height,
            image.width,
            factor=factor,
            min_pixels=IMAGE_PIXELS,
            max_pixels=IMAGE_PIXELS,
        )[::-1]
    return length + (height // factor) * (width // factor) - 1


def percentile(values: list[int], fraction: float) -> int | None:
    """The nearest-rank percentile of `values`, or None when there are none."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))]


def summarize(entries: Iterable[dict], *, max_length: int) -> dict:
    """Per dataset and split, the distribution of measured row lengths.

    `over_max_length` is reported against the budget in force when the run was
    made; redefining that budget is the point of measuring, so it is a reference
    line rather than a judgement.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for entry in entries:
        groups.setdefault((entry["dataset"], entry["split"]), []).append(entry["tokens"])
    summary: dict[str, dict] = {}
    for (dataset, split), lengths in sorted(groups.items()):
        summary.setdefault(split, {})[dataset] = _stats(lengths, max_length)
    everything = [value for lengths in groups.values() for value in lengths]
    summary["all"] = _stats(everything, max_length)
    return summary


def _stats(lengths: list[int], max_length: int) -> dict:
    return {
        "rows": len(lengths),
        "tokens_min": min(lengths) if lengths else None,
        "tokens_p50": percentile(lengths, 0.50),
        "tokens_p90": percentile(lengths, 0.90),
        "tokens_p99": percentile(lengths, 0.99),
        "tokens_max": max(lengths) if lengths else None,
        "over_max_length": sum(1 for value in lengths if value > max_length),
    }


def write_report(
    output: Path,
    entries: list[dict],
    *,
    summary: dict,
    model: str | None,
    max_length: int,
) -> dict:
    """Write `token_lengths.jsonl` and `token_stats.json` under `output`."""
    output = Path(output)
    with (output / "token_lengths.jsonl").open("w") as stream:
        for entry in sorted(entries, key=lambda item: item["id"]):
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    report = {
        "model": model,
        "image_pixels": IMAGE_PIXELS,
        "max_length": max_length,
        "note": (
            "measured after conversion; nothing was excluded for its length. "
            "token_lengths.jsonl holds one line per row, so any future budget can "
            "be re-derived without measuring again"
        ),
        "stats": summary,
    }
    (output / "token_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
