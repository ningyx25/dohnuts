"""Measure the token length of every row a conversion produced.

The Android Control conversion no longer excludes rows for their size: the token
budget is redefined from the whole corpus once it has been measured. This script
is that measurement -- it renders every row of every split through the same
prompt template the collator uses, counts tokens and image patches, and writes
`token_lengths.jsonl` (one line per row, so a future budget can be re-derived
without measuring again) plus `token_stats.json` (per dataset and split, the
percentiles and how many rows sit above the budget in force).

Run from the repository root, after the conversion:

    python scripts/report_token_lengths.py \
        --input data/processed/ac-jev-v1 --model .cache/models/Qwen3.5-0.8B --workers 32

Rows are measured in file order by a forked pool; the pooled workers inherit the
processor (which does not pickle) and read their own line ranges, so the parent
never ships a row across a process boundary and the result does not depend on how
the workers interleave.
"""

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

from transformers import AutoProcessor

from dohnuts.recipe import MAX_LENGTH
from dohnuts.token_stats import measure, summarize, write_report

SPLITS = ["train", "dev", "calibration", "test"]

# One progress line per this many chunks keeps a long run observable.
PROGRESS_EVERY = 200

# Rows per job. Small enough to keep the pool busy, large enough that the
# per-job file read is not the cost.
CHUNK = 200

# The processor the forked workers inherit; see the module docstring.
_PROCESSOR = None


def line_count(path: Path) -> int:
    """The number of rows in one split file, or 0 when it does not exist."""
    try:
        with Path(path).open("rb") as stream:
            return sum(1 for _ in stream)
    except OSError:
        return 0


def measure_job(job: tuple[Path, int, int]) -> list[dict]:
    """One pool job: `measure_chunk` over the range it names.

    A module-level function, because a forked pool pickles the callable by name.
    """
    return measure_chunk(*job)


def measure_chunk(path: Path, start: int, end: int) -> list[dict]:
    """Measure rows `[start, end)` of one split file, in file order."""
    entries = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream):
            if number < start:
                continue
            if number >= end:
                break
            row = json.loads(line)
            entries.append(
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "tokens": measure(_PROCESSOR, row),
                }
            )
    return entries


def jobs(input_dir: Path, chunk: int = CHUNK):
    """Every `(path, start, end)` range to measure, in split order."""
    for split in SPLITS:
        path = Path(input_dir) / f"{split}.jsonl"
        total = line_count(path)
        for start in range(0, total, chunk):
            yield path, start, min(start + chunk, total)


def run(input_dir: Path, *, processor, workers: int) -> dict:
    """Measure every row under `input_dir` and write the report next to it."""
    global _PROCESSOR
    _PROCESSOR = processor
    planned = list(jobs(input_dir))
    if not planned:
        raise SystemExit(f"No split files with rows under {input_dir}")
    entries: list[dict] = []
    done = 0
    if workers <= 1:
        for job in planned:
            entries.extend(measure_job(job))
            done += 1
            _progress(done, len(planned), entries)
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(workers) as pool:
            for chunk in pool.imap(measure_job, planned, chunksize=1):
                entries.extend(chunk)
                done += 1
                _progress(done, len(planned), entries)
    model = getattr(processor, "name_or_path", None)
    summary = summarize(entries, max_length=MAX_LENGTH)
    report = write_report(
        input_dir,
        entries,
        summary=summary,
        model=model,
        max_length=MAX_LENGTH,
    )
    print(json.dumps(report), flush=True)
    return report


def _progress(done: int, total: int, entries: list[dict]) -> None:
    if done % PROGRESS_EVERY == 0 or done == total:
        print(
            json.dumps({"progress": {"chunks": done, "of": total, "rows": len(entries)}}),
            file=sys.stderr,
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A conversion output directory holding the split jsonl files",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local processor (or Hub id) whose tokenizer and patch size are used",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Measurement processes; the report is identical to --workers 1",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, not {args.workers}")
    if args.model is None:
        raise SystemExit("--model is required: the measurement needs the row's tokenizer")
    try:
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot load the model {args.model}: {error}") from error
    run(args.input, processor=processor, workers=args.workers)


if __name__ == "__main__":
    main()
