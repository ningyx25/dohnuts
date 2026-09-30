"""Convert Android Control episodes into mobile-jev shaped decision splits.

Every row's prompt is the one `ClientMobileJev` builds for the same screen and
goal (see docs/superpowers/specs/2026-09-30-mobile-jev-prompt-construction-design.md):
the `state` and the `questions` come from `dohnuts.mobile_jev_prompt`, and the
row's target is the action the corpus actually recorded.

Deterministic: numerically sorted episode directories, a corpus-level app
inventory collected in a first pass over the same files, content-addressed
screenshots, the split seed, and the conversion rules decide every output byte.
Episodes may be converted in parallel (`--workers`); the merge is ordered, so the
published files are the ones a serial run would have written. Run from the
repository root. Token lengths are measured separately by
`scripts/report_token_lengths.py`; nothing is excluded for its size here.
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

from dohnuts import mobile_jev_prompt as jev
from dohnuts.android_control_data import (
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    app_vocabulary,
    metadata_detail,
    parse_episode,
    question_criteria,
    rows_for_ac_step,
    tap_candidates,
    target_family,
)
from dohnuts.android_control_mark import mark_screenshot
from dohnuts.gui_data import SPLIT_LIMITS, SPLIT_SEED, isolate, validate_rows

SPLITS = ["train", "dev", "calibration", "test"]

# One progress line per this many episodes keeps a long run observable without
# flooding stderr on a small input.
PROGRESS_EVERY = 1000

# The family-level reasons each question can lose its row to, so the manifest can
# attribute a drop to the family that asked for it.
FAMILY_REASONS = {
    "operation": (),
    "tap_target": ("no_target_element", "too_few_candidates", "too_many_candidates"),
    "text_value": ("text_not_a_goal_span", "too_few_candidates", "too_many_candidates"),
    "app_target": ("app_not_offered", "too_few_candidates", "too_many_candidates"),
}

# Temporary names of `staged_write`, unique within a process and across them.
_TEMP_NAMES = count()

# The rules behind a row are stated in the manifest, so a consumer can tell what
# its criteria and targets mean without reading this script.
PROMPT_RULE = (
    "Every row's state and question are byte-for-byte what android_world's "
    "ClientMobileJev builds for the same accessibility tree and goal: the nine "
    "state keys (goal, app, isEditable, textSource, textEntryAvailableAfterFocus, "
    "visibleText, elements, availableApps, recentActions) plus an optional "
    "focusedField, and one question whose id, instructions and criteria text come "
    "from that policy. The operation question's criteria keep its fixed order "
    "(OPEN_APP, TAP, TYPE_TEXT, SCROLL_DOWN, SCROLL_UP, SCROLL_LEFT, SCROLL_RIGHT, "
    "BACK, HOME, ENTER, WAIT, DONE, BLOCKED) and only include what the screen "
    "offers; element indices are the policy's shared 1-based numbering (TAP "
    "targets first, then scroll-only regions), and instructions are the "
    "{'goal', 'rules'} object the policy sends."
)
TARGET_RULE = (
    "A target row exists only for the operation the step recorded: a click lists "
    "the TAP candidates as '[i] label' and the ground truth is every candidate "
    "whose box contains the recorded point, weighted by inverse box area and "
    "normalized over the hits (a single hit is a one-hot); a typed step lists the "
    "goal's 1..8 word spans plus NONE and the ground truth is the typed span when "
    "the goal contains it; an open_app step lists the offered app inventory and "
    "the ground truth is the app the corpus opened. Only steps with 2.."
    f"{MAX_CANDIDATES} candidates produce the row."
)
MARKED_IMAGES = (
    "The screenshot of a tap_target row is a set-of-mark rendering: every TAP "
    "candidate is boxed and labelled with the criteria key it is offered under, "
    "nothing else is drawn, and the PNG drops the image metadata of its source, so "
    "its bytes are a pure function of the pixels, the candidate list, and the "
    "Pillow version recorded under environment."
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


def metadata_documents(episodes: list[Path]):
    """Yield each episode's metadata value, in episode order, skipping failures.

    The first pass of the run reads the same small files the conversion reads, so
    the inventory it produces is a property of the input rather than of the
    worker schedule. Kept as a generator so 15k documents are never all in
    memory at once.
    """
    for episode_dir in episodes:
        document, _, _ = read_metadata(episode_dir, Path(episode_dir).name)
        yield document


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
    digest of the bytes `parse_episode` hashed and the model reads back exactly
    the bytes that were decoded at parse time, so re-encoding here would break
    both.
    """
    if not target.exists() or digest_file(target) != step.image_sha256:
        # Re-copy a target whose content does not match its name: a killed
        # previous run can leave a truncated file that later runs would trust.
        with staged_write(target) as staged:
            shutil.copyfile(step.image, staged)


def marked_candidates(step) -> list[tuple[str, object]]:
    """The `(criteria key, element)` pairs a tap_target row's screenshot draws.

    Only a step whose ground truth is a tap target gets a mark: every step whose
    screen offers TAP has a `tap_target` question, and rendering one for all of
    them would write a file no row mentions.
    """
    if step.target is None or step.target.family != "tap_target":
        return []
    criteria = question_criteria(step.request, "tap_target")
    if criteria is None:  # not reachable: a resolved target needs its question
        return []
    return list(zip(criteria, tap_candidates(step.request, step.observation)))


def render_marked(image: Path, candidates: list[tuple[str, object]]) -> tuple[bytes, str]:
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
        marked = mark_screenshot(handle, candidates)
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


def process_episode(episode_dir: Path, *, output: Path, root: Path, apps: tuple[str, ...]) -> dict:
    """Convert one episode into the plain-data slice of the run it contributes.

    Everything returned is picklable, because the result crosses a process
    boundary when the run is parallel: `rows`, `excluded`, `images`, `audit`,
    `attempts`, and `metadata_bytes` (the raw metadata file the parent folds into
    its digest in episode order -- a digest cannot be merged after the fact, so
    the bytes have to travel).

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
            episode_dir, name, document, detail, output=output, root=root, apps=apps
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
            "attempts": {},
            "family_drops": {},
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
    apps: tuple[str, ...],
) -> dict:
    """Parse, store and mint the rows of one episode.

    The body of the conversion loop, keyed by the directory `name` the episode
    is addressed by; `document` is the metadata as read (or None when it could
    not be read, with `detail` naming why) and `process_episode` turns whatever
    escapes here into an exclusion.
    """
    audit: Counter = Counter()
    excluded: list[dict] = []
    rows: list[dict] = []
    images_written: list[str] = []
    attempts: Counter = Counter()
    family_drops: Counter = Counter()
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
        steps, step_exclusions = parse_episode(record, episode_dir=episode_dir, apps=apps)
        for entry in step_exclusions:
            stage = entry["stage"]
            audit[f"{stage}:{entry['reason']}"] += 1
            if stage == "family":
                # The id ends in the family that failed, which is what lets the
                # manifest attribute a shared reason to the right question.
                family_drops[f"{entry['id'].rsplit(':', 1)[-1]}:{entry['reason']}"] += 1
            excluded.append(entry)
        for step in steps:
            try:
                attempts["operation"] += 1
                family = target_family(step.operation)
                if family is not None:
                    attempts[family] += 1
                raw_target = image_path(output, step.image_sha256)
                marked = None
                marked_png = None
                candidates = marked_candidates(step)
                if candidates:
                    payload, marked_sha256 = render_marked(step.image, candidates)
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
            except Exception as error:
                # A filesystem or decode failure outside the rule parsers must
                # not abort the batch, and the half-made rows are dropped:
                # a step is never partially converted.
                audit["parse:unexpected"] += 1
                excluded.append(
                    {
                        "id": step.id,
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
        "attempts": dict(attempts),
        "family_drops": dict(family_drops),
    }


def episode_results(
    episodes: list[Path], *, output: Path, root: Path, apps: tuple[str, ...], workers: int
):
    """Yield one `process_episode` result per episode, in episode order.

    `workers == 1` runs the very same function in this process, so a serial run
    and a parallel one differ in nothing but who does the work. Above that, a
    forked pool hands episodes to `workers` processes and `imap` yields their
    results in submission order -- which is what keeps the merged rows,
    exclusions, image list and metadata digest identical to the serial run, no
    matter how the workers interleave. The app inventory travels as a job
    argument rather than a module global, so the output cannot depend on when the
    pool was forked.
    """
    job = partial(process_episode, output=output, root=root, apps=apps)
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


def family_stats(counts: Counter, widths: dict, positives: dict) -> dict:
    """Per split, per dataset: row count and candidate-width distribution."""
    stats: dict[str, dict] = {}
    for (dataset, split), n in sorted(counts.items()):
        candidates = widths.get((dataset, split), [])
        hits = positives.get((dataset, split), [])
        stats.setdefault(split, {})[dataset] = {
            "rows": n,
            "candidates_min": min(candidates) if candidates else None,
            "candidates_mean": (sum(candidates) / len(candidates)) if candidates else None,
            "candidates_max": max(candidates) if candidates else None,
            # A soft target names more than one correct candidate: the element
            # family is the only one that can, because a click can land inside
            # several nested boxes.
            "soft_targets": sum(1 for count in hits if count > 1),
            "max_positive_weights": max(hits) if hits else None,
        }
    return stats


def convert(source: Path, output: Path, *, workers: int = 1) -> dict:
    """Convert every episode under `source` into the splits under `output`.

    `workers` above 1 converts that many episodes at a time. The workers only
    ever run rules on their own episode and store content-addressed files, so the
    run is as deterministic as the serial one: every episode's results are merged
    in episode order, which fixes the row order, the counters, the image list,
    and the metadata digest. Returns the manifest it published.

    A rerun into an existing `--output` reuses the screenshots whose bytes still
    match their names, sweeps the temporaries a killed run may have left, and
    writes everything else in place; stale PNGs from an older input are left
    where they are, so `manifest["images"]` describes this run rather than the
    directory.
    """
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
    # Pass one: the offered app inventory. The first pass over the corpus is
    # what lets a row ask which app to open, since Android Control never records
    # the device's installed apps -- only the apps it actually opened.
    inventory = tuple(app_vocabulary(metadata_documents(episodes)))
    print(
        json.dumps({"inventory": {"apps": len(inventory), "cap": jev.MAX_APPS}}),
        file=sys.stderr,
        flush=True,
    )
    audit: Counter = Counter()
    excluded: list[dict] = []
    rows: list[dict] = []
    images_written: set[str] = set()
    attempts: Counter = Counter()
    family_drops: Counter = Counter()
    metadata_files_hashed = 0
    metadata_sha256 = hashlib.sha256()
    merged = 0
    try:
        results = episode_results(
            episodes, output=output, root=root, apps=inventory, workers=workers
        )
        for number, result in enumerate(results, 1):
            if result["metadata_bytes"] is not None:
                metadata_files_hashed += 1
                metadata_sha256.update(result["metadata_bytes"])
            audit.update(result["audit"])
            excluded.extend(result["excluded"])
            rows.extend(result["rows"])
            images_written.update(result["images"])
            attempts.update(result["attempts"])
            family_drops.update(result["family_drops"])
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
    # Coverage is measured on the rows this run minted, before isolation drops
    # the ones that collide across splits, so it stays additive with the family
    # drop counters. `counts` below is post-isolation, like the split files.
    minted: Counter = Counter(row["dataset"] for row in rows)
    dropped: list = []
    kept = isolate(rows, audit, dropped)
    try:
        validate_rows(kept, root=root)
    except ValueError as error:
        raise SystemExit(f"Self-check failed: {error}") from error
    counts: Counter = Counter()
    operations: dict[str, Counter] = {}
    widths: dict[tuple[str, str], list[int]] = {}
    positives: dict[tuple[str, str], list[int]] = {}
    try:
        handles = {split: (output / f"{split}.jsonl").open("w") for split in SPLITS}
        try:
            for row in kept:
                handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[(row["dataset"], row["split"])] += 1
                if row["dataset"] == "jev_operation":
                    criteria = list(row["question"]["criteria"])
                    operations.setdefault(row["split"], Counter())[
                        criteria[row["target"].index(1.0)]
                    ] += 1
                widths.setdefault((row["dataset"], row["split"]), []).append(
                    len(row["question"]["criteria"])
                )
                positives.setdefault((row["dataset"], row["split"]), []).append(
                    sum(1 for weight in row["target"] if weight > 0)
                )
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
    coverage = {}
    for family, dataset in (
        ("operation", "jev_operation"),
        ("tap_target", "jev_tap_target"),
        ("text_value", "jev_text_value"),
        ("app_target", "jev_app_target"),
    ):
        asked = attempts.get(family, 0)
        rows_produced = minted.get(dataset, 0)
        coverage[family] = {
            "basis": "rows are pre-isolation; steps_asking counts parsed steps",
            "dataset": dataset,
            "steps_asking": asked,
            "rows": rows_produced,
            "rate": (rows_produced / asked) if asked else None,
            "dropped": {
                reason: family_drops[f"{family}:{reason}"]
                for reason in FAMILY_REASONS[family]
                if family_drops.get(f"{family}:{reason}")
            },
        }
    manifest = {
        "schema_version": 2,
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
        "operation_classes": {
            split: dict(sorted(values.items())) for split, values in sorted(operations.items())
        },
        "family_stats": family_stats(counts, widths, positives),
        "family_coverage": coverage,
        "family_drops": {
            reason: count for reason, count in sorted(audit.items()) if reason.startswith("family:")
        },
        "exclusions": dict(sorted(audit.items())),
        "images": sorted(images_written),
        "app_inventory": {
            "apps": len(inventory),
            "offered_cap": jev.MAX_APPS,
            "source": "corpus open_app display names, sorted by (casefold, name)",
        },
        "dataset_weighting": (
            "TrainingBatches draws a dataset name uniformly at random before drawing a row "
            "from it, so the four dataset names carry equal weight regardless of how many "
            "rows each has; because tap_target, text_value and app_target rows only exist on "
            "the steps that recorded such an action, those rows are relatively upweighted"
        ),
        "token_stats": (
            "not measured here; run scripts/report_token_lengths.py over this directory to "
            "get token_lengths.jsonl and token_stats.json. No row is excluded for its length"
        ),
        "vocabularies": {
            "rules": jev.RULES,
            "operation_descriptions": jev.OPERATION_DESCRIPTIONS,
            "target_question_template": jev.TARGET_QUESTION_TEMPLATE,
            "text_value_none": jev.TEXT_VALUE_NONE,
            "text_value_instructions": jev.TEXT_VALUE_EXTRA_INSTRUCTIONS,
            "state_keys": [
                "goal",
                "app",
                "isEditable",
                "textSource",
                "textEntryAvailableAfterFocus",
                "visibleText",
                "elements",
                "availableApps",
                "recentActions",
                "focusedField",
            ],
            "question_ids": [
                "operation",
                "app_target",
                "tap_target",
                "scroll_target",
                "text_value",
            ],
            "limits": {
                "max_choice_options": jev.MAX_CHOICE_OPTIONS,
                "max_text_candidates": jev.MAX_TEXT_CANDIDATES,
                "max_text_ngram": jev.MAX_TEXT_NGRAM,
                "max_apps": jev.MAX_APPS,
                "max_payload_bytes": jev.MAX_PAYLOAD_BYTES,
                "min_candidates_per_row": MIN_CANDIDATES,
                "max_candidates_per_row": MAX_CANDIDATES,
            },
            "prompt_rule": PROMPT_RULE,
            "target_rule": TARGET_RULE,
            "marked_images": MARKED_IMAGES,
            "deviations": [
                "rows carry a screenshot (the agent is text only): the raw frame, or the "
                "marked frame for tap_target",
                "scroll_target rows are never produced: Android Control records a scroll "
                "direction but no coordinates, so the scrolled region is unknown",
                "a row's question has 2..128 options while the agent allows 255",
                "a history entry's scroll region is the lowest-indexed scroll candidate of "
                "that screen, because the corpus does not record which region moved",
                "the app inventory is the corpus's own open_app names, not the device's "
                "installed apps, and is capped at "
                + str(jev.MAX_APPS)
                + " like the agent",
                "the last step of an episode is assumed to be a successful termination "
                "(the parsed corpus dropped goal_status)",
            ],
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


def _family_reasons(family: str) -> tuple[str, ...]:
    """The audit keys that explain why one family produced no row."""
    return FAMILY_REASONS[family]


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
        "--workers",
        type=int,
        default=1,
        help="Episodes to convert at a time; the output is identical to --workers 1",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, not {args.workers}")
    convert(args.input, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
