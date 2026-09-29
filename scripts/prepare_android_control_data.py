"""Convert Android Control episodes into Dohnuts decision splits.

Deterministic: numerically sorted episode directories, content-addressed
screenshots, the split seed, and the conversion rules decide every output byte.
Episodes may be converted in parallel (`--workers`); the merge is ordered, so
the published files are the ones a serial run would have written. Run from the
repository root.
"""

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from functools import partial
from itertools import count
from pathlib import Path

import PIL
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.android_control_data import (
    AC_ACTIONS,
    ELEMENT_INSTRUCTION,
    metadata_detail,
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

# The token-check processor: a module global rather than a job argument because
# it holds a tokenizer and an image processor that do not pickle, while a forked
# worker inherits the object as it stands when the pool is created.
_TOKEN_PROCESSOR = None

# Temporary names of `staged_write`, unique within a process and across them.
_TEMP_NAMES = count()

# The rules behind a `screenshot_choice` row are stated in the manifest, so a
# consumer can tell what its label indices mean without reading this script.
ELEMENT_RULE = (
    "A click or long_press step lists the visible, non-degenerate, clickable nodes "
    "of its step_NNN_a11y.json in window order and node order; the kept nodes are "
    "numbered contiguously r0..r{N-1} (nothing is deduplicated), the ground truth "
    "is every node whose box contains the recorded tap point -- weighted by inverse "
    "box area and normalized over those hits, so a single hit is a one-hot and the "
    "smaller of several nested boxes carries the larger share -- and only steps "
    "with 2..128 candidates are converted."
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
    except (ValueError, RecursionError) as error:
        # JSONDecodeError and UnicodeDecodeError are both ValueError, and a torn
        # or non-UTF-8 file must not abort the batch with a traceback. A document
        # nested past the interpreter's recursion limit is as unusable, and a
        # RecursionError escaping here would kill a pool worker rather than
        # exclude the episode.
        return None, raw, f"{type(error).__name__}: {error}"


def token_length(processor, row: dict) -> int:
    """Rendered tokens plus expanded image placeholders, as prepare_data.filter_data counts."""
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


def image_path(output: Path, sha256: str) -> Path:
    """The content-addressed path of one stored screenshot."""
    return Path(output) / "images" / (sha256 + ".png")


@contextlib.contextmanager
def staged_write(target: Path):
    """Yield a temporary sibling of `target`, renamed over it on success.

    Several workers can store the same content-addressed file, and a reader can
    open a name the moment it appears, so writing the final name directly would
    let either of them see half a PNG. Filling a temporary file and renaming it
    is atomic within one directory; a failure removes the temporary file rather
    than leaving it behind for the next run to trip over.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{os.getpid()}.{next(_TEMP_NAMES)}.tmp")
    try:
        yield staged
        os.replace(staged, target)
    except BaseException:
        with contextlib.suppress(OSError):
            staged.unlink()
        raise


def store_raw(step, target: Path) -> None:
    """Copy the step screenshot to its content-addressed `target`.

    The copy is byte-identical to the source: the row rules alias the file by the
    digest of the bytes `parse_step` hashed and the model reads back exactly the
    bytes that were decoded at parse time, so re-encoding here would break both.
    """
    if not target.exists() or digest_file(target) != step.image_sha256:
        # Re-copy a target whose content does not match its name: a killed
        # previous run can leave a truncated file that later runs would trust.
        with staged_write(target) as staged:
            shutil.copyfile(step.image, staged)


def render_marked(image: Path, elements: list[dict]) -> tuple[bytes, str]:
    """Render the set-of-mark screenshot of one step; return its PNG bytes and digest.

    The digest comes from a sha256 of the exact bytes written, computed before
    the write, so the file is content-addressed by construction and a rerun over
    an existing file with a matching digest has nothing to do. The PNG is encoded
    from an in-memory buffer rather than saved to a path, which is what keeps the
    bytes -- and therefore every name derived from them -- a property of the
    image and the candidate list alone. Rendering before storing also means a
    step that never reaches its rows writes nothing at all.
    """
    with Image.open(image) as handle:
        marked = mark_screenshot(handle, elements)
    buffer = io.BytesIO()
    marked.save(buffer, format="PNG")
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def store_marked(target: Path, payload: bytes) -> None:
    """Write one rendered set-of-mark PNG to its content-addressed `target`."""
    sha256 = target.name.removesuffix(".png")
    if not target.exists() or digest_file(target) != sha256:
        with staged_write(target) as staged:
            staged.write_bytes(payload)


def process_episode(episode_dir: Path, *, output: Path, root: Path, token_check: bool) -> dict:
    """Convert one episode into the plain-data slice of the run it contributes.

    Everything returned is picklable, because the result crosses a process
    boundary when the run is parallel: `rows`, `excluded`, `images`, `audit`,
    `element_rows`, `empty_targets`, and `metadata_bytes` (the raw metadata file
    the parent folds into its digest in episode order -- a digest cannot be
    merged after the fact, so the bytes have to travel).

    Never raises. Each step of the episode is already guarded, the metadata read
    reports its own failures, and an episode that fails outside those guards
    comes back as a single `unexpected` exclusion with none of its half-made
    rows -- but with the metadata bytes it did read, so the parent's digest
    count stays what the serial loop would have counted.
    """
    name = Path(episode_dir).name
    try:
        # Deliberately outside the guarded body: `read_metadata` never raises,
        # and its bytes have to be in hand even if the conversion then fails.
        document, raw, detail = read_metadata(episode_dir, name)
    except Exception:  # a backstop for the never-raises contract, not a path
        document, raw, detail = None, None, ""
    try:
        result = convert_episode(
            episode_dir, name, document, detail, output=output, root=root, token_check=token_check
        )
    except Exception as error:
        result = {
            "rows": [],
            "excluded": [
                {
                    "id": name,
                    "reason": "unexpected",
                    "detail": f"{type(error).__name__}: {error}",
                    "stage": "parse",
                }
            ],
            "images": [],
            "audit": {"parse:unexpected": 1},
            "element_rows": 0,
            "empty_targets": {},
        }
    result["metadata_bytes"] = raw
    return result


def convert_episode(
    episode_dir: Path,
    name: str,
    document: object,
    detail: str,
    *,
    output: Path,
    root: Path,
    token_check: bool,
) -> dict:
    """Parse, probe, store and derive the rows of one episode.

    The body of the conversion loop, keyed by the directory `name` the episode
    is addressed by; `document` is the metadata as read (or None when it could
    not be read, with `detail` naming why) and `process_episode` turns whatever
    escapes here into an exclusion.
    """
    audit: Counter = Counter()
    excluded: list[dict] = []
    rows: list[dict] = []
    images_written: list[str] = []
    empty_targets: dict[str, bool] = {}
    element_rows = 0
    record = document
    if isinstance(record, dict):
        # A valid JSON document of the wrong shape is as unusable as a torn
        # file, and comes back with the same reason; the detail names the first
        # step or field that does not satisfy the schema.
        violation = metadata_detail(record)
        if violation:
            record, detail = None, violation
    elif record is not None:
        # A JSON document that is not an object cannot carry the keys the steps
        # are read from; `metadata_detail` refuses it in the same words as any
        # other structural failure. This branch is also the type check that
        # lets the rest of the function treat `record` as a mapping.
        record, detail = None, metadata_detail(record)
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
                if token_check:
                    # The budget is checked before anything is written, and
                    # the probe rows are the rows this step would mint: only
                    # the element question carries the candidate texts, so
                    # the answer is the same either way. The probe's raw path
                    # stands in for the marked copy because marking never
                    # changes the dimensions `token_length` reads (never the
                    # pixels), and its duplicated alias never leaves this
                    # computation.
                    probe = (str(step.image), step.image_sha256)
                    lengths = [
                        token_length(_TOKEN_PROCESSOR, row)
                        for row in rows_for_ac_step(
                            step,
                            str(step.image),
                            marked=probe if step.element_weights is not None else None,
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
                raw_target = image_path(output, step.image_sha256)
                marked = None
                marked_png = None
                if step.element_weights is not None:
                    payload, marked_sha256 = render_marked(step.image, step.elements)
                    marked_png = (image_path(output, marked_sha256), payload)
                    marked = (os.path.relpath(marked_png[0], root), marked_sha256)
                # Rows are minted before anything is written, and an image is
                # written and listed only for a step whose rows exist: no
                # failure can leave a file on disk that the manifest omits.
                produced = rows_for_ac_step(step, os.path.relpath(raw_target, root), marked=marked)
                store_raw(step, raw_target)
                images_written.append(raw_target.name)
                if marked_png is not None:
                    store_marked(*marked_png)
                    images_written.append(marked_png[0].name)
                rows.extend(produced)
                if step.element_weights is not None:
                    element_rows += 1
                    # The payload is `{}` exactly when the candidate carries
                    # neither a text nor a description, which makes the question
                    # unanswerable from the prompt: the row counts as an empty
                    # target only when no candidate the point touched carries
                    # one, since any of them is a correct answer.
                    empty_targets[f"{step.id}:element"] = not any(
                        element["text"] or element["content_description"]
                        for position, element in enumerate(step.elements)
                        if step.element_weights[position] > 0
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
    return {
        "rows": rows,
        "excluded": excluded,
        "images": images_written,
        "audit": dict(audit),
        "element_rows": element_rows,
        "empty_targets": empty_targets,
    }


def episode_results(episodes: list[Path], *, output: Path, root: Path, workers: int):
    """Yield one `process_episode` result per episode, in episode order.

    `workers == 1` runs the very same function in this process, so a serial run
    and a parallel one differ in nothing but who does the work. Above that, a
    forked pool hands episodes to `workers` processes and `imap` yields their
    results in submission order -- which is what keeps the merged rows,
    exclusions, image list and metadata digest identical to the serial run, no
    matter how the workers interleave. The jobs carry no processor: the token
    check reads the module global the fork inherited. That is fork-only by
    design -- under `spawn` or `forkserver` a worker would re-import this module
    with `_TOKEN_PROCESSOR` at None and silently skip the token check, so the
    start method is pinned rather than left to the platform default.
    """
    job = partial(
        process_episode, output=output, root=root, token_check=_TOKEN_PROCESSOR is not None
    )
    if workers <= 1:
        for episode_dir in episodes:
            yield job(episode_dir)
        return
    context = multiprocessing.get_context("fork")
    with context.Pool(workers) as pool:
        yield from pool.imap(job, episodes, chunksize=1)


def sweep_staged(output: Path) -> None:
    """Delete the temporary files a killed run left in the image directory.

    `staged_write` removes its own temporary file whenever it sees a failure,
    but a SIGKILL or a lost machine leaves one behind, and nothing else ever
    deletes it: the operator's next run into the same `--output` would then find
    a file under `images/` that its manifest does not list. Only the
    `.<name>.<pid>.<n>.tmp` names this module writes are touched; stored PNGs,
    referenced or orphaned, are left alone.
    """
    images = Path(output) / "images"
    if not images.is_dir():
        return
    for stale in images.glob(".*.tmp"):
        with contextlib.suppress(OSError):
            stale.unlink()


def convert(source: Path, output: Path, *, processor=None, workers: int = 1) -> dict:
    """Convert every episode under `source` into the splits under `output`.

    `processor` enables the token budget check; `workers` above 1 converts that
    many episodes at a time. The workers only ever run rules on their own
    episode and store content-addressed files, so the run is as deterministic
    as the serial one: every episode's results are merged in episode order,
    which fixes the row order, the counters, the image list, and the metadata
    digest. Returns the manifest it published.

    A rerun into an existing `--output` reuses the screenshots whose bytes still
    match their names, sweeps the temporaries a killed run may have left, and
    writes everything else in place; stale PNGs from an older input are left
    where they are, so `manifest["images"]` describes this run rather than the
    directory.
    """
    global _TOKEN_PROCESSOR
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
    sweep_staged(output)
    _TOKEN_PROCESSOR = processor
    audit: Counter = Counter()
    excluded: list[dict] = []
    rows: list[dict] = []
    images_written: set[str] = set()
    empty_targets: dict[str, bool] = {}
    element_rows = 0
    metadata_files_hashed = 0
    metadata_sha256 = hashlib.sha256()
    merged = 0
    try:
        for number, result in enumerate(
            episode_results(episodes, output=output, root=root, workers=workers), 1
        ):
            if result["metadata_bytes"] is not None:
                metadata_files_hashed += 1
                metadata_sha256.update(result["metadata_bytes"])
            audit.update(result["audit"])
            excluded.extend(result["excluded"])
            rows.extend(result["rows"])
            images_written.update(result["images"])
            empty_targets.update(result["empty_targets"])
            element_rows += result["element_rows"]
            merged = number
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
    except KeyboardInterrupt as interrupt:
        # Ctrl-C is a decision, not a crash: leave the pool to clean up after
        # itself and report how far the run got, the way the rest of the CLI
        # reports a stop instead of printing a traceback.
        raise SystemExit(f"Interrupted after {merged} of {len(episodes)} episodes") from interrupt
    dropped: list = []
    kept = isolate(rows, audit, dropped)
    try:
        validate_rows(kept, root=root)
    except ValueError as error:
        raise SystemExit(f"Self-check failed: {error}") from error
    counts: Counter = Counter()
    classes: dict[str, Counter] = {}
    candidates: dict[str, list[int]] = {}
    hits: dict[str, list[int]] = {}
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
                    # The positive weights are the hits the row actually teaches;
                    # more than one makes it a soft target rather than a one-hot.
                    hits.setdefault(row["split"], []).append(
                        sum(1 for weight in row["target"] if weight > 0)
                    )
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
        hit_counts = hits[split]
        soft = sum(1 for count in hit_counts if count > 1)
        element_stats[split] = {
            "basis": "post_isolation",
            "rows": len(widths),
            "candidates_min": min(widths),
            "candidates_mean": sum(widths) / len(widths),
            "candidates_max": max(widths),
            # How much of the family is a distribution rather than a one-hot:
            # a soft target names every hit, so the model is only graded on
            # picking one of them, not on picking the smallest.
            "soft_targets": soft,
            "multi_hit_rate": soft / len(widths),
            "max_hits": max(hit_counts),
            "empty_target_payloads": empties,
            "empty_target_payload_rate": empties / len(widths),
        }
    misses = [
        audit[f"parse:{reason}"]
        for reason in ("no_target_element", "too_few_candidates", "too_many_candidates")
    ]
    # The hit rate is over the steps that asked an element question at all: the
    # audit counts the three ways `resolve_element_target` can refuse one, and
    # `element_rows` counts the steps that resolved and within budget (one choice
    # row each). A step that resolved but was excluded by the budget is in
    # neither count, so the rates here and in the manifest stay additive.
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Episodes to convert at a time; the output is identical to --workers 1",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, not {args.workers}")
    processor = None
    if args.model is not None and args.token_check:
        try:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot load the token-check model {args.model}: {error}") from error
    convert(args.input, args.output, processor=processor, workers=args.workers)


if __name__ == "__main__":
    main()
