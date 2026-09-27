"""Convert GUI step records into Dohnuts decision splits.

Deterministic: sorted input files, fixed vocabularies, the split seed, and the
conversion rules decide every output byte. Run from the repository root.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    COMPLETE_CRITERIA,
    INSTRUCTIONS,
    SPLIT_LIMITS,
    SPLIT_SEED,
    SWIPE_DIRECTIONS,
    isolate,
    parse_step,
    rows_for_step,
    validate_rows,
)
from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH

SPLITS = ["train", "dev", "calibration", "test"]


def digest_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repository_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise SystemExit("Run from the dohnuts repository root: no git repository found") from error
    return Path(result.stdout.strip()).resolve()


def input_files(source: Path) -> list[Path]:
    return [source] if source.is_file() else sorted(source.glob("*.json"))


def records(paths: list[Path]):
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"Unreadable step record file: {path} ({type(error).__name__}: {error})"
            ) from error
        if not isinstance(data, list):
            raise SystemExit(f"Unreadable step record file: {path} (expected a JSON array)")
        yield from data


def store_image(step, output: Path) -> Path:
    target = output / "images" / (step.image_sha256 + ".png")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(step.image, target)
    return target


def token_length(processor, row: dict) -> int:
    """Rendered tokens plus expanded image placeholders, as prepare_data.filter_data counts."""
    prompt, _ = render_question(render(row["state"]), row["question"], has_image=True)
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


def convert(source: Path, output: Path, *, processor=None) -> dict:
    root = repository_root()
    if Path.cwd().resolve() != root:
        raise SystemExit(f"Run from the repository root: {root}")
    source = Path(source)
    image_root = source if source.is_dir() else source.parent
    paths = input_files(source)
    if not paths:
        raise SystemExit(f"No step record *.json found under {source}")
    output.mkdir(parents=True, exist_ok=True)
    audit: Counter = Counter()
    excluded = []
    rows = []
    images_written: set[str] = set()
    for record in records(paths):
        try:
            step, reason = parse_step(record, image_root=image_root)
            if step is None:
                audit[f"parse:{reason}"] += 1
                excluded.append(
                    {
                        "id": record.get("id") if isinstance(record, dict) else None,
                        "reason": reason,
                        "detail": "",
                        "stage": "parse",
                    }
                )
                continue
            if processor is not None:
                lengths = [
                    token_length(processor, row) for row in rows_for_step(step, str(step.image))
                ]
                if any(length > MAX_LENGTH for length in lengths):
                    audit["parse:token_budget"] += 1
                    excluded.append(
                        {
                            "id": step.id,
                            "reason": "token_budget",
                            "detail": str(max(lengths)),
                            "stage": "parse",
                        }
                    )
                    continue
            stored = store_image(step, output)
            images_written.add(stored.name)
            rows.extend(rows_for_step(step, os.path.relpath(stored, root)))
        except Exception as error:
            # Filesystem and decode failures outside parse_step must not abort a run.
            audit["parse:unexpected"] += 1
            excluded.append(
                {
                    "id": record.get("id") if isinstance(record, dict) else None,
                    "reason": "unexpected",
                    "detail": f"{type(error).__name__}: {error}",
                    "stage": "parse",
                }
            )
            continue
    dropped: list = []
    kept = isolate(rows, audit, dropped)
    try:
        validate_rows(kept, root=root)
    except ValueError as error:
        raise SystemExit(f"Self-check failed: {error}") from error
    handles = {split: (output / f"{split}.jsonl").open("w") for split in SPLITS}
    counts: Counter = Counter()
    classes: dict[str, Counter] = {}
    try:
        for row in kept:
            handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[(row["dataset"], row["split"])] += 1
            if row["dataset"] == "gui_action":
                label = list(ACTIONS)[row["target"].index(1.0)]
                classes.setdefault(row["split"], Counter())[label] += 1
    finally:
        for handle in handles.values():
            handle.close()
    with (output / "excluded.jsonl").open("w") as stream:
        for entry in [*excluded, *dropped]:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    empty = [split for split in SPLITS if not any(key[1] == split for key in counts)]
    if empty:
        print(
            json.dumps({"warning": "empty splits: " + ", ".join(empty)}),
            file=sys.stderr,
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "split_limits": SPLIT_LIMITS,
        "source": {
            "input": str(source),
            "files": [{"path": str(path), "sha256": digest_file(path)} for path in paths],
        },
        "path_convention": f"repository-root relative ({root})",
        "counts": [
            {"dataset": dataset, "split": split, "n": count}
            for (dataset, split), count in sorted(counts.items())
        ],
        "action_classes": {
            split: dict(sorted(values.items())) for split, values in sorted(classes.items())
        },
        "exclusions": dict(sorted(audit.items())),
        "images": sorted(images_written),
        "dataset_weighting": "uniform per dataset name; button and swipe rows are upweighted",
        "token_check": "skipped" if processor is None else "enabled",
        "vocabularies": {
            "actions": ACTIONS,
            "buttons": BUTTONS,
            "swipe_directions": SWIPE_DIRECTIONS,
            "instructions": INSTRUCTIONS,
            "complete_criteria": COMPLETE_CRITERIA,
        },
        "sha256": {split: digest_file(output / f"{split}.jsonl") for split in SPLITS},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Step record JSON file, or a directory of *.json step-record arrays (not an output directory)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Split directory to create")
    parser.add_argument(
        "--model", type=Path, default=None, help="Local model used for the token budget check"
    )
    parser.add_argument("--no-token-check", dest="token_check", action="store_false")
    args = parser.parse_args(argv)
    processor = None
    if args.model is not None and args.token_check:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    convert(args.input, args.output, processor=processor)


if __name__ == "__main__":
    main()
