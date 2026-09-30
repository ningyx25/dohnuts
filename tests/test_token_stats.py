"""Token measurement: the length formula, the statistics, and the CLI report.

The processor is a stub: what is under test is the arithmetic and the report,
not a tokenizer. The stub counts one token per whitespace-separated piece and
reports a patch grid, which is enough to pin both terms of the formula.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from dohnuts import token_stats
from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS


class StubTokenizer:
    def __call__(self, text, truncation=False):
        del truncation
        return {"input_ids": list(range(len(text.split())))}


class StubImageProcessor:
    patch_size = 16
    merge_size = 2


class StubProcessor:
    def __init__(self, name="stub"):
        self.tokenizer = StubTokenizer()
        self.image_processor = StubImageProcessor()
        self.name_or_path = name


def write_png(path: Path, size=(64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def test_measure_counts_prompt_tokens_plus_image_patches(tmp_path):
    image = write_png(tmp_path / "frame.png")
    row = {
        "state": {"goal": "g", "app": "com.example"},
        "question": {
            "type": "choice",
            "instructions": "Which one?",
            "criteria": {"1": "a", "2": "b"},
        },
        "image": str(image),
    }
    processor = StubProcessor()
    prompt, _, _ = render_question(render(row["state"]), row["question"], has_image=True)
    # `image_pixels` pins both the minimum and the maximum, so every screenshot
    # is resized to the same square and the image term is one constant: the
    # patch grid minus the placeholder the prompt already counted.
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    side = int(IMAGE_PIXELS**0.5) // factor
    assert token_stats.measure(processor, row) == len(prompt.split()) + side * side - 1
    # The same image at another size still lands on that same grid.
    other = write_png(tmp_path / "small.png", size=(32, 32))
    assert token_stats.measure(processor, {**row, "image": str(other)}) == len(
        prompt.split()
    ) + side * side - 1


def test_percentile_uses_nearest_rank():
    assert token_stats.percentile([], 0.5) is None
    assert token_stats.percentile([5], 0.99) == 5
    assert token_stats.percentile([1, 2, 3, 4, 5], 0.0) == 1
    assert token_stats.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert token_stats.percentile([1, 2, 3, 4, 5], 1.0) == 5


def test_summarize_groups_by_dataset_and_split():
    entries = [
        {"id": "a", "dataset": "jev_operation", "split": "train", "tokens": 10},
        {"id": "b", "dataset": "jev_operation", "split": "train", "tokens": 30},
        {"id": "c", "dataset": "jev_tap_target", "split": "dev", "tokens": 20},
        {"id": "d", "dataset": "jev_tap_target", "split": "dev", "tokens": 5000},
    ]
    summary = token_stats.summarize(entries, max_length=100)
    assert summary["train"]["jev_operation"] == {
        "rows": 2,
        "tokens_min": 10,
        "tokens_p50": 10,
        "tokens_p90": 30,
        "tokens_p99": 30,
        "tokens_max": 30,
        "over_max_length": 0,
    }
    assert summary["dev"]["jev_tap_target"]["over_max_length"] == 1
    assert summary["all"]["rows"] == 4
    assert summary["all"]["tokens_max"] == 5000


def test_write_report_is_sorted_by_id_and_names_the_model(tmp_path):
    entries = [
        {"id": "b", "dataset": "d", "split": "train", "tokens": 2},
        {"id": "a", "dataset": "d", "split": "train", "tokens": 1},
    ]
    report = token_stats.write_report(
        tmp_path, entries, summary=token_stats.summarize(entries, max_length=10), model="stub",
        max_length=10,
    )
    assert report["model"] == "stub"
    assert report["stats"]["all"]["tokens_max"] == 2
    lines = (tmp_path / "token_lengths.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]
    assert json.loads((tmp_path / "token_stats.json").read_text())["max_length"] == 10


def load_script():
    name = "report_token_lengths"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts/report_token_lengths.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


def write_splits(directory: Path, rows_per_split=3):
    """Three split files, each with rows whose images exist."""
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "calibration", "test"):
        with (directory / f"{split}.jsonl").open("w") as stream:
            for index in range(rows_per_split):
                image = write_png(directory / "images" / f"{split}_{index}.png")
                stream.write(
                    json.dumps(
                        {
                            "id": f"{split}-{index}",
                            "dataset": "jev_operation" if index % 2 else "jev_tap_target",
                            "split": split,
                            "state": {"goal": "g"},
                            "question": {
                                "type": "choice",
                                "instructions": "Which?",
                                "criteria": {"1": "a", "2": "b"},
                            },
                            "image": str(image),
                        }
                    )
                    + "\n"
                )
    return directory


def test_jobs_cover_every_line_once(tmp_path):
    write_splits(tmp_path)
    planned = list(script.jobs(tmp_path, chunk=2))
    assert script.line_count(tmp_path / "train.jsonl") == 3
    assert script.line_count(tmp_path / "missing.jsonl") == 0
    covered = [(path.name, start, end) for path, start, end in planned]
    assert covered == [
        ("train.jsonl", 0, 2),
        ("train.jsonl", 2, 3),
        ("dev.jsonl", 0, 2),
        ("dev.jsonl", 2, 3),
        ("calibration.jsonl", 0, 2),
        ("calibration.jsonl", 2, 3),
        ("test.jsonl", 0, 2),
        ("test.jsonl", 2, 3),
    ]


def test_run_writes_a_report_for_every_row(tmp_path):
    write_splits(tmp_path)
    report = script.run(tmp_path, processor=StubProcessor(), workers=1)
    assert report["stats"]["all"]["rows"] == 12
    entries = [
        json.loads(line) for line in (tmp_path / "token_lengths.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 12
    assert {entry["split"] for entry in entries} == {"train", "dev", "calibration", "test"}
    assert all(entry["tokens"] > 0 for entry in entries)
    assert (tmp_path / "token_stats.json").is_file()


def test_run_is_identical_across_workers(tmp_path):
    write_splits(tmp_path)
    serial = script.run(tmp_path, processor=StubProcessor(), workers=1)
    first = (tmp_path / "token_lengths.jsonl").read_text()
    parallel = script.run(tmp_path, processor=StubProcessor(), workers=2)
    assert first == (tmp_path / "token_lengths.jsonl").read_text()
    assert serial["stats"] == parallel["stats"]


def test_run_needs_rows(tmp_path):
    with pytest.raises(SystemExit):
        script.run(tmp_path, processor=StubProcessor(), workers=1)
