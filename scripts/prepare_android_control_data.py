"""Convert Android Control episodes into Dohnuts decision splits.

Deterministic: numerically sorted episode directories, content-addressed
screenshots, the split seed, and the conversion rules decide every output byte.
Run from the repository root.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.android_control_data import (
    AC_ACTIONS,
    ELEMENT_INSTRUCTION,
    parse_metadata,
    parse_step,
    rows_for_ac_step,
)
from dohnuts.android_control_mark import mark_screenshot
from dohnuts.gui_data import (
    BUTTONS,
    COMPLETE_CRITERIA,
    INSTRUCTIONS,
    SPLIT_LIMITS,
    SPLIT_SEED,
    SWIPE_DIRECTIONS,
    isolate,
    validate_rows,
)
from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH

SPLITS = ["train", "dev", "calibration", "test"]

# One progress line per this many episodes keeps a multi-hour run observable
# without flooding stderr on a small input.
PROGRESS_EVERY = 1000

# The rules behind a `screenshot_choice` row are stated in the manifest, so a
# consumer can tell what its label indices mean without reading this script.
ELEMENT_RULE = (
    "A click or long_press step lists the visible, non-degenerate, clickable nodes "
    "of its step_NNN_a11y.json in window order and node order; the kept nodes are "
    "numbered contiguously r0..r{N-1} (nothing is deduplicated), the ground truth "
    "is the smallest node containing the recorded tap point with ties going to the "
    "earlier candidate, and only steps with 2..128 candidates are converted."
)
MARKED_IMAGES = (
    "The screenshot of a screenshot_choice row is a set-of-mark rendering: every "
    "candidate is boxed and numbered in the same order as the criteria, nothing "
    "else is drawn, and the PNG drops the image metadata of its source, so its "
    "bytes are a pure function of the pixels, the candidate list, and the Pillow "
    "version recorded under environment."
)


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


def episode_dirs(source: Path) -> list[Path]:
    """The episode directories directly under `source`, in numeric id order.

    An episode is a directory whose name is an all-digit id, which is what the
    corpus uses and what its metadata file name is built from. Anything else --
    a README, a stray file, a directory with a textual name -- is not part of the
    corpus and is ignored. The sort key includes the name so that ids which only
    differ by leading zeros still order the same way on every run.
    """
    try:
        children = list(source.iterdir())
    except OSError:
        return []
    episodes = [
        child
        for child in children
        if child.name.isascii() and child.name.isdigit() and child.is_dir()
    ]
    return sorted(episodes, key=lambda path: (int(path.name), path.name))


def read_metadata(episode_dir: Path, name: str) -> tuple[object | None, bytes | None, str]:
    """Read `<episode_dir>/metadata_<name>.json`: value, raw bytes, failure note.

    A missing, unreadable, or non-JSON file comes back as `(None, bytes-or-None,
    detail)` so that one bad episode is excluded instead of aborting the batch.
    The bytes come back whenever the file could be read at all, because the
    manifest hashes them whether or not they parse.
    """
    path = episode_dir / f"metadata_{name}.json"
    try:
        raw = path.read_bytes()
    except OSError as error:
        return None, None, f"{type(error).__name__}: {error}"
    try:
        return json.loads(raw), raw, ""
    except ValueError as error:
        # JSONDecodeError and UnicodeDecodeError are both ValueError, and a torn
        # or non-UTF-8 file must not abort the batch with a traceback.
        return None, raw, f"{type(error).__name__}: {error}"


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


def store_raw(step, output: Path) -> Path:
    """Copy the step screenshot to its content-addressed name; return that path.

    The copy is byte-identical to the source: the row rules alias the file by the
    digest of the bytes `parse_step` hashed and `validate_rows` decodes the
    stored copy, so re-encoding here would break both.
    """
    target = output / "images" / (step.image_sha256 + ".png")
    if not target.exists() or digest_file(target) != step.image_sha256:
        # Re-copy a target whose content does not match its name: a killed
        # previous run can leave a truncated file that later runs would trust.
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(step.image, target)
    return target


def store_marked(image: Path, elements: list[dict], output: Path) -> tuple[Path, str]:
    """Render the set-of-mark screenshot of one step; return its path and digest.

    The name comes from a sha256 of the exact bytes written, computed before the
    write, so the file is content-addressed by construction and a rerun over an
    existing file with a matching digest has nothing to do. The PNG is encoded
    from an in-memory buffer rather than saved to a path, which is what keeps the
    bytes -- and therefore every name derived from them -- a property of the
    image and the candidate list alone.
    """
    with Image.open(image) as handle:
        marked = mark_screenshot(handle, elements)
    buffer = io.BytesIO()
    marked.save(buffer, format="PNG")
    payload = buffer.getvalue()
    sha256 = hashlib.sha256(payload).hexdigest()
    target = output / "images" / (sha256 + ".png")
    if not target.exists() or digest_file(target) != sha256:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return target, sha256


def convert(source: Path, output: Path, *, processor=None) -> dict:
    root = repository_root()
    if Path.cwd().resolve() != root:
        raise SystemExit(f"Run from the repository root: {root}")
    source = Path(source)
    output = Path(output)
    episodes = episode_dirs(source)
    if not episodes:
        raise SystemExit(f"No episode directories found under {source}")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"Cannot create the output directory {output}: {error}") from error
    audit: Counter = Counter()
    excluded: list[dict] = []
    rows: list[dict] = []
    images_written: set[str] = set()
    empty_targets: dict[str, bool] = {}
    element_rows = 0
    metadata_files_hashed = 0
    metadata_sha256 = hashlib.sha256()
    for number, episode_dir in enumerate(episodes, 1):
        name = episode_dir.name
        record, raw, detail = read_metadata(episode_dir, name)
        if raw is not None:
            metadata_files_hashed += 1
            metadata_sha256.update(raw)
        if record is not None:
            # A valid JSON document of the wrong shape is as unusable as a torn
            # file, and comes back with the same reason; there is no exception
            # text to report, so the entry carries no detail.
            record, _ = parse_metadata(record)
            detail = ""
        if record is not None and int(name) != record["episode_id"]:
            # The metadata file is named after its directory while the rows are
            # keyed by the record's id: a corpus whose two ids disagree would be
            # keyed unpredictably, so the episode is excluded instead.
            detail = f"episode_id mismatch: dir {name} vs record {record['episode_id']}"
            record = None
        if record is None:
            # One unreadable episode out of 15,283 is not a reason to stop the
            # batch, and the directory name is the only id the episode has left.
            audit["parse:unparsable_metadata"] += 1
            excluded.append(
                {"id": name, "reason": "unparsable_metadata", "detail": detail, "stage": "parse"}
            )
        else:
            for index in range(len(record["steps"])):
                step_id = f"android_control_{record['episode_id']}_step{index}"
                try:
                    step, reason = parse_step(record, index, episode_dir=episode_dir)
                    if step is None:
                        audit[f"parse:{reason}"] += 1
                        excluded.append(
                            {"id": step_id, "reason": reason, "detail": "", "stage": "parse"}
                        )
                        continue
                    if processor is not None:
                        # The budget is checked before anything is written, and
                        # the probe rows are the rows this step would mint: only
                        # the element question carries the candidate texts, so
                        # the answer is the same either way. Its dummy marked
                        # copy duplicates the raw alias, which never leaves this
                        # computation.
                        probe = (str(step.image), step.image_sha256)
                        lengths = [
                            token_length(processor, row)
                            for row in rows_for_ac_step(
                                step,
                                str(step.image),
                                marked=probe if step.target_element is not None else None,
                            )
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
                    raw_target = store_raw(step, output)
                    marked = None
                    marked_target = None
                    if step.target_element is not None:
                        marked_target, marked_sha256 = store_marked(
                            step.image, step.elements, output
                        )
                        marked = (os.path.relpath(marked_target, root), marked_sha256)
                    rows.extend(
                        rows_for_ac_step(step, os.path.relpath(raw_target, root), marked=marked)
                    )
                    # An image enters the manifest only once the rows naming it
                    # exist, so a failure above cannot leave an unreferenced file
                    # behind in it.
                    images_written.add(raw_target.name)
                    if step.target_element is not None:  # hence marked_target is set too
                        images_written.add(marked_target.name)
                        element_rows += 1
                        target = step.elements[step.target_element]
                        # The payload is `{}` exactly when the ground-truth
                        # candidate carries neither a text nor a description,
                        # which makes the question unanswerable from the prompt.
                        empty_targets[f"{step.id}:element"] = not (
                            target["text"] or target["content_description"]
                        )
                except Exception as error:
                    # A filesystem or decode failure outside the rule parsers must
                    # not abort the batch, and the half-made rows are dropped:
                    # a step is never partially converted.
                    audit["parse:unexpected"] += 1
                    excluded.append(
                        {
                            "id": step_id,
                            "reason": "unexpected",
                            "detail": f"{type(error).__name__}: {error}",
                            "stage": "parse",
                        }
                    )
                    continue
        if number % PROGRESS_EVERY == 0:
            print(
                json.dumps(
                    {
                        "progress": {
                            "episodes": number,
                            "rows": len(rows),
                            "exclusions": len(excluded),
                        }
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    dropped: list = []
    kept = isolate(rows, audit, dropped)
    try:
        validate_rows(kept, root=root)
    except ValueError as error:
        raise SystemExit(f"Self-check failed: {error}") from error
    counts: Counter = Counter()
    classes: dict[str, Counter] = {}
    candidates: dict[str, list[int]] = {}
    empty_payloads: Counter = Counter()
    try:
        handles = {split: (output / f"{split}.jsonl").open("w") for split in SPLITS}
        try:
            for row in kept:
                handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[(row["dataset"], row["split"])] += 1
                if row["dataset"] == "gui_action":
                    label = list(AC_ACTIONS)[row["target"].index(1.0)]
                    classes.setdefault(row["split"], Counter())[label] += 1
                elif row["dataset"] == "screenshot_choice":
                    candidates.setdefault(row["split"], []).append(len(row["question"]["criteria"]))
                    if empty_targets.get(row["id"]):
                        empty_payloads[row["split"]] += 1
        finally:
            for handle in handles.values():
                handle.close()
    except OSError as error:
        raise SystemExit(f"Cannot write the split files under {output}: {error}") from error
    try:
        with (output / "excluded.jsonl").open("w") as stream:
            for entry in [*excluded, *dropped]:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as error:
        raise SystemExit(f"Cannot write the exclusions under {output}: {error}") from error
    empty = [split for split in SPLITS if not any(key[1] == split for key in counts)]
    if empty:
        print(
            json.dumps({"warning": "empty splits: " + ", ".join(empty)}),
            file=sys.stderr,
            flush=True,
        )
    element_stats = {}
    for split in SPLITS:
        widths = candidates.get(split)
        if not widths:
            continue
        empties = empty_payloads[split]
        element_stats[split] = {
            "basis": "post_isolation",
            "rows": len(widths),
            "candidates_min": min(widths),
            "candidates_mean": sum(widths) / len(widths),
            "candidates_max": max(widths),
            "empty_target_payloads": empties,
            "empty_target_payload_rate": empties / len(widths),
        }
    misses = [
        audit[f"parse:{reason}"]
        for reason in ("no_target_element", "too_few_candidates", "too_many_candidates")
    ]
    # The hit rate is over the steps that asked an element question at all: the
    # audit counts the three ways `resolve_element_choice` can refuse one, and
    # `element_rows` counts the steps it resolved (one choice row each).
    asked = element_rows + sum(misses)
    element_resolution = {
        "basis": "pre_isolation",
        "element_rows": element_rows,
        "no_target_element": misses[0],
        "too_few_candidates": misses[1],
        "too_many_candidates": misses[2],
        "hit_rate": element_rows / asked if asked else None,
    }
    manifest = {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "split_limits": SPLIT_LIMITS,
        "source": {
            "input": str(source),
            "episodes": len(episodes),
            "metadata_sha256": metadata_sha256.hexdigest(),
            "metadata_files_hashed": metadata_files_hashed,
        },
        "path_convention": f"repository-root relative ({root})",
        "counts": [
            {"dataset": dataset, "split": split, "n": count}
            for (dataset, split), count in sorted(counts.items())
        ],
        "action_classes": {
            split: dict(sorted(values.items())) for split, values in sorted(classes.items())
        },
        "element_stats": element_stats,
        "element_resolution": element_resolution,
        "exclusions": dict(sorted(audit.items())),
        "images": sorted(images_written),
        "dataset_weighting": (
            "TrainingBatches draws a dataset name uniformly at random before drawing a row "
            "from it, so the five dataset names carry equal weight regardless of how many "
            "rows each has; because gui_button and gui_swipe rows only exist on the steps "
            "that press a button or scroll, those rows are relatively upweighted"
        ),
        "token_check": "skipped" if processor is None else "enabled",
        "vocabularies": {
            "ac_actions": AC_ACTIONS,
            "buttons": BUTTONS,
            "swipe_directions": SWIPE_DIRECTIONS,
            "instructions": {**INSTRUCTIONS, "element": ELEMENT_INSTRUCTION},
            "complete_criteria": COMPLETE_CRITERIA,
            "element_rule": ELEMENT_RULE,
            "marked_images": MARKED_IMAGES,
        },
        "environment": {"python": platform.python_version(), "pillow": PIL.__version__},
        "sha256": {split: digest_file(output / f"{split}.jsonl") for split in SPLITS},
    }
    try:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except OSError as error:
        raise SystemExit(f"Cannot write the manifest under {output}: {error}") from error
    print(json.dumps(manifest), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory of `{episode_id}` episode directories (not an output directory)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Split directory to create")
    parser.add_argument(
        "--model", type=Path, default=None, help="Local model used for the token budget check"
    )
    parser.add_argument("--no-token-check", dest="token_check", action="store_false")
    args = parser.parse_args(argv)
    processor = None
    if args.model is not None and args.token_check:
        try:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot load the token-check model {args.model}: {error}") from error
    convert(args.input, args.output, processor=processor)


if __name__ == "__main__":
    main()
