"""GUI step conversion: parsing, row derivation, isolation, and CLI output."""

import hashlib
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    SWIPE_DIRECTIONS,
    isolate,
    parse_step,
    rows_for_step,
    split_for,
    validate_rows,
)

QUERY = (
    "In the Files app, locate the 'task.html' file within the Downloads folder and switch "
    "the display layout from the current grid view to a list view.."
)
PROGRESS = (
    "(You have done the following operation on the current device): "
    "Step 1: I swiped up on the home screen to open the app drawer and locate the Files app.; ."
)


def write_image(root: Path, name: str, color: str) -> Path:
    path = root / name
    Image.new("RGB", (8, 8), color).save(path)
    return path


def user_content(query: str = QUERY, progress: str = PROGRESS) -> str:
    return f"The user query: {query}\nTask progress {progress}\n<image>"


def make_record(uid, arguments, *, image="shot.png", images=None, name="mobile_use"):
    tool_call = json.dumps({"name": name, "arguments": arguments})
    return {
        "id": uid,
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": user_content()},
            {
                "role": "assistant",
                "content": (
                    "Thought: Clear the dialog first.\nAction: Press Back.\n"
                    f"<tool_call>\n{tool_call}\n</tool_call>"
                ),
            },
        ],
        "images": [image] if images is None else images,
        "bbox": None,
    }


@pytest.fixture
def image_root(tmp_path):
    write_image(tmp_path, "shot.png", "red")
    write_image(tmp_path, "other.png", "blue")
    return tmp_path


def test_parse_step_reads_state_and_tool_call(image_root):
    step, reason = parse_step(
        make_record("645_BrowserMaze_step3", {"action": "system_button", "button": "Back"}),
        image_root=image_root,
    )
    assert reason is None
    assert step.id == "645_BrowserMaze_step3"
    assert step.group == "task:645_BrowserMaze"
    assert step.state == {"user_query": QUERY, "task_progress": PROGRESS}
    assert step.arguments == {"action": "system_button", "button": "Back"}
    assert step.reference["thought"] == "Clear the dialog first."
    assert step.reference["tool_call"]["name"] == "mobile_use"
    assert step.image == image_root / "shot.png"
    assert step.image_sha256 == hashlib.sha256((image_root / "shot.png").read_bytes()).hexdigest()


def test_parse_step_rejects_unparsable_state(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"][1]["content"] = "no template here"
    assert parse_step(record, image_root=image_root) == (None, "unparsable_state")


def test_parse_step_rejects_missing_tool_call(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"][2]["content"] = "Thought: x\nAction: y"
    assert parse_step(record, image_root=image_root) == (None, "missing_tool_call")


def test_parse_step_rejects_unknown_tool(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1}, name="computer_use")
    assert parse_step(record, image_root=image_root) == (None, "unknown_tool")


def test_parse_step_rejects_missing_id(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["id"] = ""
    assert parse_step(record, image_root=image_root) == (None, "missing_id")


def test_parse_step_rejects_unknown_action(image_root):
    record = make_record("a_step1", {"action": "double_click"})
    assert parse_step(record, image_root=image_root) == (None, "unknown_action")


def test_parse_step_rejects_invalid_button(image_root):
    record = make_record("a_step1", {"action": "system_button", "button": "Volume"})
    assert parse_step(record, image_root=image_root) == (None, "invalid_button")


def test_parse_step_rejects_diagonal_swipe_tie(image_root):
    record = make_record(
        "a_step1", {"action": "swipe", "coordinate": [0, 0], "coordinate2": [10, 10]}
    )
    assert parse_step(record, image_root=image_root) == (None, "invalid_swipe")


def test_parse_step_rejects_missing_swipe_coordinates(image_root):
    record = make_record("a_step1", {"action": "swipe", "coordinate": [0, 0]})
    assert parse_step(record, image_root=image_root) == (None, "invalid_swipe")


def test_parse_step_rejects_multi_image(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1}, images=["shot.png", "other.png"])
    assert parse_step(record, image_root=image_root) == (None, "multi_image")


def test_parse_step_rejects_missing_image(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1}, image="gone.png")
    assert parse_step(record, image_root=image_root) == (None, "missing_image")


def test_parse_step_rejects_multi_turn_records(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"].append({"role": "assistant", "content": "Thought: again\nAction: y"})
    assert parse_step(record, image_root=image_root) == (None, "multi_turn")


def test_parse_step_keeps_messages_of_other_roles(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"].insert(0, {"role": "system", "content": "tools"})
    step, reason = parse_step(record, image_root=image_root)
    assert reason is None
    assert step.state["user_query"] == QUERY


def test_parse_step_rejects_malformed_containers(image_root):
    assert parse_step("not a record", image_root=image_root) == (None, "unparsable_state")
    assert parse_step({"messages": "hello", "id": "a_step1"}, image_root=image_root) == (
        None,
        "unparsable_state",
    )
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"] = [record["messages"][1], record["messages"][2], "junk"]
    assert parse_step(record, image_root=image_root) == (None, "unparsable_state")


def test_parse_step_rejects_non_string_action_and_button(image_root):
    assert parse_step(make_record("a_step1", {"action": ["click"]}), image_root=image_root) == (
        None,
        "unknown_action",
    )
    assert parse_step(
        make_record("a_step1", {"action": "system_button", "button": ["Back"]}),
        image_root=image_root,
    ) == (None, "invalid_button")


def test_parse_step_rejects_malformed_images_container(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["images"] = {"0": "shot.png"}
    assert parse_step(record, image_root=image_root) == (None, "multi_image")
    record["images"] = "shot.png"
    assert parse_step(record, image_root=image_root) == (None, "multi_image")
    record["images"] = [""]
    assert parse_step(record, image_root=image_root) == (None, "multi_image")


def test_parse_step_rejects_image_paths_outside_the_input_root(image_root):
    assert parse_step(
        make_record("a_step1", {"action": "wait", "time": 1}, image="../shot.png"),
        image_root=image_root,
    ) == (None, "missing_image")
    assert parse_step(
        make_record("a_step1", {"action": "wait", "time": 1}, image="/etc/hostname"),
        image_root=image_root,
    ) == (None, "missing_image")


def test_parse_step_rejects_undecodable_image(image_root):
    (image_root / "shot.png").write_bytes(b"not an image")
    record = make_record("a_step1", {"action": "wait", "time": 1})
    assert parse_step(record, image_root=image_root) == (None, "missing_image")


def test_parse_step_prefers_earlier_checks(image_root):
    record = make_record("a_step1", {"action": "double_click"}, name="computer_use")
    record["images"] = []
    assert parse_step(record, image_root=image_root) == (None, "unknown_tool")


def test_parse_step_rejects_overlong_image_names(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1}, image="x" * 300 + ".png")
    assert parse_step(record, image_root=image_root) == (None, "missing_image")


def test_parse_step_rejects_invalid_tool_call_bodies(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"][2]["content"] = "Thought: x\nAction: y\n<tool_call>\nnot json\n</tool_call>"
    assert parse_step(record, image_root=image_root) == (None, "missing_tool_call")
    record["messages"][2]["content"] = (
        'Thought: x\nAction: y\n<tool_call>\n{"name": "mobile_use"}\n</tool_call>'
    )
    assert parse_step(record, image_root=image_root) == (None, "missing_tool_call")


def test_parse_step_maps_missing_role_to_multi_turn(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"][1]["content"] = None
    assert parse_step(record, image_root=image_root) == (None, "multi_turn")
    record = make_record("a_step1", {"action": "wait", "time": 1})
    record["messages"][2]["content"] = 42
    assert parse_step(record, image_root=image_root) == (None, "multi_turn")


def test_parse_step_pins_empty_turn_reason(image_root):
    record = {"id": "a_step1", "messages": [{"role": "user", "content": None}]}
    assert parse_step(record, image_root=image_root) == (None, "unparsable_state")


def test_parse_step_rejects_decompression_bombs(image_root):
    import struct
    import zlib

    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 50000, 50000, 8, 2, 0, 0, 0)
    (image_root / "shot.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 10))
        + chunk(b"IEND", b"")
    )
    record = make_record("a_step1", {"action": "wait", "time": 1})
    assert parse_step(record, image_root=image_root) == (None, "missing_image")


def test_parse_step_rejects_deeply_nested_tool_calls(image_root):
    record = make_record("a_step1", {"action": "wait", "time": 1})
    nested = "[" * 200_000 + "]" * 200_000
    record["messages"][2]["content"] = f"Thought: x\nAction: y\n<tool_call>\n{nested}\n</tool_call>"
    assert parse_step(record, image_root=image_root) == (None, "missing_tool_call")


def test_parse_step_groups_steps_with_provenance_suffixes(image_root):
    record = make_record(
        "1001_MarkorEditNote_step13__from0208_qwen3vl_supple_new", {"action": "wait", "time": 1}
    )
    step, reason = parse_step(record, image_root=image_root)
    assert reason is None
    assert step.group == "task:1001_MarkorEditNote"


def test_parse_step_rejects_non_finite_swipe_coordinates(image_root):
    record = make_record(
        "a_step1", {"action": "swipe", "coordinate": [float("nan"), 0], "coordinate2": [1, 1]}
    )
    assert parse_step(record, image_root=image_root) == (None, "invalid_swipe")
    record = make_record(
        "a_step1", {"action": "swipe", "coordinate": [0, 0], "coordinate2": [float("inf"), 0]}
    )
    assert parse_step(record, image_root=image_root) == (None, "invalid_swipe")


def test_parse_step_never_raises_on_malformed_records(image_root):
    malformed = [
        None,
        [],
        "record",
        {},
        {"id": "a_step1"},
        {"id": "a_step1", "messages": None},
        {"id": "a_step1", "messages": [None, 1, "x"]},
        {"id": "a_step1", "messages": [{"role": "user", "content": None}]},
        make_record("a_step1", {"action": {"nested": "dict"}}),
        make_record(
            "a_step1", {"action": "swipe", "coordinate": [[1], [2]], "coordinate2": [[3], [4]]}
        ),
    ]
    for record in malformed:
        step, reason = parse_step(record, image_root=image_root)
        assert step is None
        assert isinstance(reason, str)


def step_for(image_root, uid, arguments):
    step, reason = parse_step(make_record(uid, arguments), image_root=image_root)
    assert reason is None
    return step


def test_system_button_step_rows(image_root):
    step = step_for(
        image_root, "645_BrowserMaze_step3", {"action": "system_button", "button": "Home"}
    )
    rows = rows_for_step(step, "data/processed/gui-v1/images/x.png")
    assert [row["id"] for row in rows] == [
        "645_BrowserMaze_step3:action",
        "645_BrowserMaze_step3:button",
        "645_BrowserMaze_step3:complete",
    ]
    assert [row["dataset"] for row in rows] == ["gui_action", "gui_button", "gui_complete"]
    assert {row["group"] for row in rows} == {"task:645_BrowserMaze"}
    assert {row["image"] for row in rows} == {"data/processed/gui-v1/images/x.png"}
    action, button, complete = rows
    assert list(action["question"]["criteria"]) == list(ACTIONS)
    assert action["question"]["type"] == "choice"
    assert action["target"][list(ACTIONS).index("system_button")] == 1.0
    assert sum(action["target"]) == 1.0
    assert button["target"] == [0.0, 1.0, 0.0, 0.0]
    assert list(button["question"]["criteria"]) == list(BUTTONS)
    assert complete["question"]["type"] == "noul"
    assert complete["target"] == [1.0, 0.0]
    assert action["split"] == split_for("task:645_BrowserMaze")
    assert action["aliases"] == ["image-bytes:" + step.image_sha256]
    assert action["reference"]["tool_call"]["arguments"]["button"] == "Home"


def test_terminate_step_rows(image_root):
    step = step_for(image_root, "demo_step2", {"action": "terminate", "status": "success"})
    rows = rows_for_step(step, "img.png")
    assert [row["id"] for row in rows] == ["demo_step2:action", "demo_step2:complete"]
    assert rows[0]["target"][list(ACTIONS).index("terminate")] == 1.0
    assert rows[1]["target"] == [0.0, 1.0]


def test_swipe_step_rows_use_dominant_axis(image_root):
    arguments = {"action": "swipe", "coordinate": [500, 800], "coordinate2": [200, 800]}
    rows = rows_for_step(step_for(image_root, "demo_step1", arguments), "img.png")
    assert [row["id"] for row in rows] == [
        "demo_step1:action",
        "demo_step1:complete",
        "demo_step1:swipe_dir",
    ]
    swipe = rows[-1]
    assert swipe["dataset"] == "gui_swipe"
    assert list(swipe["question"]["criteria"]) == list(SWIPE_DIRECTIONS)
    assert swipe["target"][list(SWIPE_DIRECTIONS).index("left")] == 1.0


def test_wait_step_rows_have_no_conditional_row(image_root):
    rows = rows_for_step(
        step_for(image_root, "demo_step1", {"action": "wait", "time": 2}), "img.png"
    )
    assert [row["id"] for row in rows] == ["demo_step1:action", "demo_step1:complete"]


def test_split_for_is_stable_and_covers_partitions():
    assert split_for("task:645_BrowserMaze") == "train"
    assert split_for("task:demo") == "test"
    assert split_for("task:beta") == "test"
    assert split_for("task:002_Gallery") == "dev"
    assert split_for("task:delta") == "calibration"


def row_stub(
    uid,
    *,
    group,
    split,
    alias="image-bytes:deadbeef",
    dataset="gui_action",
    question=None,
    state=None,
):
    return {
        "id": uid,
        "dataset": dataset,
        "group": group,
        "aliases": [alias],
        "split": split,
        "state": {"user_query": uid} if state is None else state,
        "image": "img.png",
        "question": question
        or {"type": "choice", "instructions": "q", "criteria": {"a": "a", "b": "b"}},
        "target": [1.0, 0.0],
    }


def collision_rows():
    """One shared-image pair across splits plus three independent groups."""
    return [
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:1"),
        row_stub("b:action", group="task:b", split="test", alias="image-bytes:1"),
        row_stub("c:action", group="task:c", split="test", alias="image-bytes:2"),
        row_stub("d:action", group="task:d", split="dev", alias="image-bytes:3"),
        row_stub("e:action", group="task:e", split="calibration", alias="image-bytes:4"),
    ]


def test_isolate_drops_content_duplicates_with_distinct_ids():
    audit, dropped = Counter(), []
    state = {"user_query": "same"}
    rows = [
        row_stub("a:complete", group="task:a", split="train", alias="image-bytes:1", state=state),
        row_stub("b:complete", group="task:a", split="train", alias="image-bytes:2", state=state),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["a:complete"]
    assert dropped == [
        {"id": "b:complete", "reason": "duplicate_input", "detail": "", "stage": "isolate"}
    ]


def test_isolate_accounts_for_every_input_row():
    audit, dropped = Counter(), []
    rows = collision_rows()
    kept = list(isolate(rows, audit, dropped))
    assert Counter(row["id"] for row in kept) + Counter(entry["id"] for entry in dropped) == (
        Counter(row["id"] for row in rows)
    )
    assert sum(audit.values()) == len(dropped)
    rerun_audit, rerun_dropped = Counter(), []
    assert [row["id"] for row in isolate(kept, rerun_audit, rerun_dropped)] == [
        row["id"] for row in kept
    ]
    assert not rerun_dropped
    assert not rerun_audit


def test_isolate_is_invariant_under_input_permutation():
    fingerprints = set()
    for order in itertools.permutations(range(5)):
        rows = collision_rows()
        kept = isolate([rows[index] for index in order], Counter(), [])
        fingerprints.add(
            json.dumps(sorted([[row["id"], row["group"], row["split"]] for row in kept]))
        )
    assert len(fingerprints) == 1


def test_isolate_drops_only_the_lower_priority_rows_of_a_shared_screenshot():
    audit, dropped = Counter(), []
    rows = [
        row_stub("a:action", group="task:a", split="train"),
        row_stub("b:action", group="task:b", split="test"),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["b:action"]
    assert kept[0]["split"] == "test"
    assert kept[0]["group"] == "task:b"  # groups are never rewritten
    assert sum(audit.values()) == 1
    assert "cross_split_group" in next(iter(audit))
    assert dropped == [
        {"id": "a:action", "reason": "cross_split_group", "detail": "", "stage": "isolate"}
    ]


def test_isolate_drops_duplicate_inputs():
    audit, dropped = Counter(), []
    question = {"type": "choice", "instructions": "q", "criteria": {"a": "a", "b": "b"}}
    rows = [
        row_stub(
            "a:action", group="task:a", split="train", alias="image-bytes:1", question=question
        ),
        row_stub(
            "a:action", group="task:a", split="train", alias="image-bytes:1", question=question
        ),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert len(kept) == 1
    assert "duplicate_input" in next(iter(audit))
    assert dropped[0]["reason"] == "duplicate_input"
    assert dropped[0]["stage"] == "isolate"


def test_swipe_direction_vertical_axis(image_root):
    arguments = {"action": "swipe", "coordinate": [500, 200], "coordinate2": [500, 800]}
    rows = rows_for_step(step_for(image_root, "demo_step1", arguments), "img.png")
    assert rows[-1]["target"][list(SWIPE_DIRECTIONS).index("down")] == 1.0


def test_isolate_is_order_independent_for_shared_screenshots():
    audit, dropped = Counter(), []
    rows = [
        row_stub("b:action", group="task:b", split="test"),
        row_stub("a:action", group="task:a", split="train"),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["b:action"]
    assert kept[0]["group"] == "task:b"
    assert dropped[0]["id"] == "a:action"


def test_isolate_drops_duplicate_row_ids():
    audit, dropped = Counter(), []
    other = {"type": "choice", "instructions": "other", "criteria": {"a": "a", "b": "b"}}
    rows = [
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:1"),
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:2", question=other),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["a:action"]
    assert dropped == [
        {
            "id": "a:action",
            "reason": "duplicate_input",
            "detail": "row id already seen",
            "stage": "isolate",
        }
    ]


def test_isolate_keeps_same_split_screenshots_untouched():
    audit, dropped = Counter(), []
    rows = [
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:9"),
        row_stub("b:action", group="task:b", split="train", alias="image-bytes:9"),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["a:action", "b:action"]
    assert {row["group"] for row in kept} == {"task:a", "task:b"}
    assert not dropped
    assert not audit


def test_validate_rows_rejects_screenshots_across_splits(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    # A distinct task keeps the group check quiet, so the alias check is the one
    # that must fire: two rows of one step always share both group and alias.
    rows[1]["group"] = "task:other"
    rows[1]["split"] = "test" if rows[0]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="Screenshots span multiple splits"):
        validate_rows(rows)


def test_isolate_drops_a_row_whose_extra_alias_is_leaked():
    audit, dropped = Counter(), []
    leaked = row_stub("a:action", group="task:a", split="train", alias="image-bytes:own")
    leaked["aliases"].append("image-bytes:shared")
    rows = [leaked, row_stub("b:action", group="task:b", split="test", alias="image-bytes:shared")]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["b:action"]
    assert dropped[0]["id"] == "a:action"


def test_validate_rows_rejects_unnormalized_target(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["target"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="not a distribution"):
        validate_rows(rows)


def test_validate_rows_rejects_target_width_mismatch(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["target"] = [1.0, 0.0]
    with pytest.raises(ValueError, match="width"):
        validate_rows(rows)


def test_validate_rows_rejects_duplicate_ids(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["id"] = rows[1]["id"]
    with pytest.raises(ValueError, match="Duplicate row id"):
        validate_rows(rows)


def test_validate_rows_rejects_unknown_split(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="Unknown split"):
        validate_rows(rows)


def test_validate_rows_reports_unreadable_images(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["image"] = "does-not-exist.png"
    with pytest.raises(ValueError, match="Image is not readable: demo_step1:action"):
        validate_rows(rows)


def corrupted_png(size=(8, 8)) -> bytes:
    """A PNG whose container is valid and whose pixels cannot be decoded.

    The chunk CRCs are correct and the IDAT is a valid zlib stream, so the
    container walk the self-check performs passes; every scanline starts with a
    filter type PNG does not define, so decoding the pixels fails.
    """
    import struct
    import zlib

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    width, height = size
    stride = 1 + width * 3
    raw = bytearray(stride * height)
    for row in range(height):
        raw[row * stride] = 99
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def test_validate_rows_verifies_each_distinct_image_once(monkeypatch, image_root):
    rows = [
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:1"),
        row_stub("b:action", group="task:b", split="train", alias="image-bytes:2"),
        row_stub("c:action", group="task:c", split="train", alias="image-bytes:3"),
    ]
    # Three rows over one file, spelled two ways, plus one over a second file:
    # the self-check re-opening a screenshot per row pays the same cost again
    # each time, so a path that already passed must not be checked again.
    rows[0]["image"] = "shot.png"
    rows[1]["image"] = "./shot.png"
    rows[2]["image"] = "other.png"
    opened = []
    verified = []
    real_open = Image.open
    real_verify = Image.Image.verify

    def counting_open(path, *arguments, **keywords):
        opened.append(Path(path).name)
        return real_open(path, *arguments, **keywords)

    def counting_verify(self):
        verified.append(Path(self.filename).name)
        return real_verify(self)

    monkeypatch.setattr("dohnuts.gui_data.Image.open", counting_open)
    # PNG overrides the base no-op `verify` with the chunk and CRC walk, and
    # every stored screenshot is a PNG, so this is the class that runs.
    monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.verify", counting_verify)
    validate_rows(rows, root=image_root)
    assert opened == ["shot.png", "other.png"]
    assert verified == ["shot.png", "other.png"]
    # A path that does not open still fails with the row that pointed at it.
    rows.append(row_stub("d:action", group="task:d", split="train", alias="image-bytes:4"))
    rows[-1]["image"] = "gone.png"
    with pytest.raises(ValueError, match="Image is not readable: d:action"):
        validate_rows(rows, root=image_root)
    assert opened[-1] == "gone.png"


def test_validate_rows_accepts_images_whose_pixels_do_not_decode(image_root):
    # Deliberate weakening, pinned here so nobody "fixes" it back: the
    # self-check walks the container -- chunks and CRCs for PNG -- and does not
    # decode, so a PNG that is structurally intact over an undecodable pixel
    # stream passes it. The real guard is the full decode both converters
    # perform on the source when they parse a step; the stored copy is then
    # either a byte-for-byte copy whose digest is re-checked whenever it is
    # reused, or bytes this process encoded from an image it just decoded.
    # Decoding here instead would cost about 2.4 h over the AC corpus, against
    # the roughly 1.2 min a container walk takes over the same files.
    screenshot = image_root / "shot.png"
    screenshot.write_bytes(corrupted_png())
    with Image.open(screenshot) as image:
        with pytest.raises(OSError, match="data stream"):
            image.convert("RGB")  # the pixels do not decode ...
    with Image.open(screenshot) as image:
        image.verify()  # ... while the container checks pass
    rows = [row_stub("a:action", group="task:a", split="train")]
    rows[0]["image"] = "shot.png"
    validate_rows(rows, root=image_root)


def test_validate_rows_reports_an_unresolvable_image_path(image_root):
    # A symlink loop resolves to a RuntimeError rather than an OSError; callers
    # abort on ValueError alone, so the row must fail like any other unreadable
    # image instead of escaping the self-check as a traceback.
    (image_root / "loop.png").symlink_to(image_root / "loop.png")
    rows = [row_stub("a:action", group="task:a", split="train")]
    rows[0]["image"] = "loop.png"
    with pytest.raises(ValueError, match="Image is not readable: a:action"):
        validate_rows(rows, root=image_root)


def test_validate_rows_rejects_group_across_splits(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[1]["split"] = "test" if rows[0]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="span multiple splits"):
        validate_rows(rows)


def test_validate_rows_accepts_converted_rows(image_root):
    step = step_for(image_root, "demo_step1", {"action": "system_button", "button": "Home"})
    validate_rows(rows_for_step(step, str(step.image)))


def test_isolate_prefers_the_row_id_reason_over_the_content_reason():
    audit, dropped = Counter(), []
    rows = [
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:1"),
        row_stub("a:action", group="task:a", split="train", alias="image-bytes:2"),
    ]
    kept = list(isolate(rows, audit, dropped))
    assert [row["id"] for row in kept] == ["a:action"]
    assert dropped[0]["detail"] == "row id already seen"


def test_validate_rows_rejects_candidate_counts_out_of_range(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[0]["question"]["criteria"] = {"only": "one candidate"}
    rows[0]["target"] = [1.0]
    with pytest.raises(ValueError, match="Candidate count out of range"):
        validate_rows(rows)
    rows = rows_for_step(step, str(step.image))
    criteria = {f"c{index}": f"candidate {index}" for index in range(129)}
    rows[0]["question"]["criteria"] = criteria
    rows[0]["target"] = [1.0] + [0.0] * 128
    with pytest.raises(ValueError, match="Candidate count out of range"):
        validate_rows(rows)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_gui_data", Path(__file__).parents[1] / "scripts/prepare_gui_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script()


def convert(tmp_path, steps):
    source = tmp_path / "steps.json"
    source.write_text(json.dumps(steps))
    output = tmp_path / "out"
    prepare.main(["--input", str(source), "--output", str(output)])
    return output


def test_cli_writes_splits_manifest_and_images(tmp_path, image_root):
    steps = [
        make_record("001_TaskA_step1", {"action": "system_button", "button": "Back"}),
        make_record(
            "002_TaskB_step2",
            {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]},
            image="other.png",
        ),
    ]
    output = convert(image_root, steps)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["split_limits"] == [["calibration", 10], ["dev", 20], ["test", 30]]
    assert manifest["token_check"] == "skipped"
    assert manifest["exclusions"] == {}
    assert set(manifest["sha256"]) == set(prepare.SPLITS)
    assert sum(entry["n"] for entry in manifest["counts"]) == 6
    assert len(manifest["images"]) == 2
    assert (output / "excluded.jsonl").read_text() == ""
    rows = [
        json.loads(line)
        for split in prepare.SPLITS
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    validate_rows(rows)
    for row in rows:
        with Image.open(Path(row["image"])) as image:
            image.convert("RGB")


def test_cli_is_deterministic(tmp_path, image_root):
    # The four split hashes are reproducible from the same --output: a row's
    # `image` field embeds the output directory, so a different --output
    # legitimately produces different bytes.
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    first = convert(image_root, steps)
    before = json.loads((first / "manifest.json").read_text())
    second = convert(image_root, steps)
    assert first == second
    after = json.loads((second / "manifest.json").read_text())
    assert after["sha256"] == before["sha256"]
    assert after["images"] == before["images"]


def test_cli_records_exclusions(image_root):
    steps = [
        make_record("001_TaskA_step1", {"action": "wait", "time": 2}),
        make_record("002_TaskB_step1", {"action": "long_press", "coordinate": [1, 2], "time": 1}),
    ]
    steps[1]["messages"][2]["content"] = "Thought: x\nAction: y"
    output = convert(image_root, steps)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["exclusions"] == {"parse:missing_tool_call": 1}
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert entry["id"] == "002_TaskB_step1"
    assert entry["stage"] == "parse"


def test_cli_survives_unexpected_failures(image_root, monkeypatch):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    output = image_root / "out"

    def explode(record, *, image_root):
        raise RuntimeError("boom")

    monkeypatch.setattr(prepare, "parse_step", explode)
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:unexpected": 1}
    assert sum(row["n"] for row in manifest["counts"]) == 0
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert entry["reason"] == "unexpected"
    assert entry["detail"] == "RuntimeError: boom"
    assert entry["stage"] == "parse"


def test_cli_reads_every_json_in_a_directory(tmp_path, image_root):
    (image_root / "b_second.json").write_text(
        json.dumps(
            [make_record("002_Gallery_step1", {"action": "wait", "time": 2}, image="other.png")]
        )
    )
    (image_root / "a_first.json").write_text(
        json.dumps([make_record("001_TaskA_step1", {"action": "wait", "time": 2})])
    )
    output = image_root / "out"
    manifest = prepare.convert(image_root, output)
    assert [Path(entry["path"]).name for entry in manifest["source"]["files"]] == [
        "a_first.json",
        "b_second.json",
    ]
    assert sum(entry["n"] for entry in manifest["counts"]) == 4


def test_cli_split_files_match_the_manifest(tmp_path, image_root):
    steps = [
        make_record("001_TaskA_step1", {"action": "wait", "time": 2}),
        make_record(
            "002_Gallery_step1", {"action": "system_button", "button": "Home"}, image="other.png"
        ),
    ]
    output = convert(image_root, steps)
    manifest = json.loads((output / "manifest.json").read_text())
    counts = Counter()
    for split in prepare.SPLITS:
        path = output / f"{split}.jsonl"
        assert manifest["sha256"][split] == prepare.digest_file(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["split"] == split
            counts[row["dataset"], split] += 1
    assert {f"{dataset}:{split}": n for (dataset, split), n in counts.items()} == {
        f"{entry['dataset']}:{entry['split']}": entry["n"] for entry in manifest["counts"]
    }


def test_cli_records_isolate_drops_and_shares_image_files(tmp_path, image_root):
    steps = [
        make_record("645_BrowserMaze_step1", {"action": "wait", "time": 2}),
        make_record("demo_step1", {"action": "wait", "time": 2}),
    ]
    output = convert(image_root, steps)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["images"]) == 1  # one screenshot file, two records
    assert manifest["exclusions"] == {
        "gui_action:train:cross_split_group": 1,
        "gui_complete:train:cross_split_group": 1,
    }
    entries = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    assert [entry["stage"] for entry in entries] == ["isolate", "isolate"]
    assert sum(entry["n"] for entry in manifest["counts"]) == 2


def test_cli_refuses_to_run_outside_the_repository_root(tmp_path, image_root, monkeypatch):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Run from the dohnuts repository root"):
        prepare.convert(source, image_root / "out")


def test_cli_requires_the_repository_root_as_the_working_directory(monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1] / "scripts")
    with pytest.raises(SystemExit, match="Run from the repository root"):
        prepare.convert(Path("scripts"), Path("/tmp/gui-cwd-check"))


def test_cli_rejects_inputs_without_step_records(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No step record"):
        prepare.convert(empty, tmp_path / "out")
    with pytest.raises(SystemExit, match="No step record"):
        prepare.convert(tmp_path / "missing", tmp_path / "out")


def test_cli_names_unreadable_input_files(tmp_path, image_root):
    (image_root / "broken.json").write_text("{not json")
    with pytest.raises(SystemExit, match="Unreadable step record file"):
        prepare.convert(image_root, image_root / "out")
    (image_root / "broken.json").write_bytes(b'{"id": "\xe9"}')
    with pytest.raises(SystemExit, match="Unreadable step record file"):
        prepare.convert(image_root, image_root / "out")


def test_cli_rejects_an_output_path_that_is_a_file(tmp_path, image_root):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    output = image_root / "out"
    output.write_text("not a directory")
    with pytest.raises(SystemExit, match="Cannot create the output directory"):
        prepare.convert(source, output)


def test_cli_stores_the_exact_screenshot_bytes(tmp_path, image_root):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    output = image_root / "out"
    prepare.convert(source, output)
    stored = output / "images" / (prepare.digest_file(image_root / "shot.png") + ".png")
    assert stored.read_bytes() == (image_root / "shot.png").read_bytes()


def test_cli_reports_an_unloadable_token_check_model(tmp_path, image_root):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    with pytest.raises(SystemExit, match="Cannot load the token-check model"):
        prepare.main(
            [
                "--input",
                str(source),
                "--output",
                str(image_root / "out"),
                "--model",
                "/nonexistent",
            ]
        )


def test_cli_output_is_consumable_by_training_data(image_root):
    model = Path("Qwen/Qwen3.5-0.8B")
    if not model.is_dir():
        pytest.skip("local Qwen3.5-0.8B snapshot is not available")
    from dohnuts.training_data import DecisionCollator, load_records

    steps = [
        make_record("645_BrowserMaze_step1", {"action": "system_button", "button": "Home"}),
        make_record(
            "002_Gallery_step1",
            {"action": "swipe", "coordinate": [500, 800], "coordinate2": [200, 800]},
            image="other.png",
        ),
    ]
    output = convert(image_root, steps)
    rows = [
        row
        for split in prepare.SPLITS
        for group in load_records(output / f"{split}.jsonl").values()
        for row in group
    ]
    assert rows
    collated, *_ = DecisionCollator(model)(rows)
    assert collated["input_ids"].shape[0] == len(rows)


def test_example_data_converts_when_present():
    source = Path(__file__).parents[1] / "example-data" / "raw_data.json"
    if not source.exists():
        pytest.skip("example-data is local-only")
    step, reason = parse_step(json.loads(source.read_text())[0], image_root=source.parent)
    assert reason is None
    rows = rows_for_step(step, "example-data/raw_images/screenshot_step3.png")
    assert [row["id"] for row in rows] == [
        "645_BrowserMaze_step3:action",
        "645_BrowserMaze_step3:button",
        "645_BrowserMaze_step3:complete",
    ]
    assert rows[0]["target"][list(ACTIONS).index("system_button")] == 1.0
    assert rows[1]["target"][list(BUTTONS).index("Back")] == 1.0
    assert rows[2]["target"] == [1.0, 0.0]


class StubImageProcessor:
    patch_size = 14
    merge_size = 2


class StubTokenizer:
    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text.split())))}


class StubProcessor:
    image_processor = StubImageProcessor()
    tokenizer = StubTokenizer()


def test_token_budget_excludes_whole_record(tmp_path, image_root):
    long_progress = "(You have done the following operation on the current device): " + " ".join(
        ["step"] * 5000
    )
    over_budget = make_record("001_TaskA_step1", {"action": "wait", "time": 2})
    over_budget["messages"][1]["content"] = user_content(progress=long_progress)
    fine = make_record("002_TaskB_step1", {"action": "wait", "time": 2}, image="other.png")
    source = image_root / "steps.json"
    source.write_text(json.dumps([over_budget, fine]))
    output = image_root / "out"
    manifest = prepare.convert(source, output, processor=StubProcessor())
    assert manifest["token_check"] == "enabled"
    assert manifest["exclusions"] == {"parse:token_budget": 1}
    assert sum(row["n"] for row in manifest["counts"]) == 2
    assert len(manifest["images"]) == 1
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert entry["id"] == "001_TaskA_step1"
    assert entry["reason"] == "token_budget"
    assert entry["stage"] == "parse"
    assert int(entry["detail"]) > 2048


def test_cli_warns_on_empty_splits(image_root, capsys):
    steps = [make_record("645_BrowserMaze_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    prepare.convert(source, image_root / "out")
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"warning": "empty splits: dev, calibration, test"}
    assert "warning" not in captured.out


def test_cli_skips_the_model_when_token_checks_are_disabled(tmp_path, image_root, monkeypatch):
    steps = [make_record("001_TaskA_step1", {"action": "wait", "time": 2})]
    source = image_root / "steps.json"
    source.write_text(json.dumps(steps))
    output = image_root / "out"

    def explode(*args, **kwargs):
        raise AssertionError("the token-check model must not be loaded")

    monkeypatch.setattr(prepare.AutoProcessor, "from_pretrained", explode)
    prepare.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--model",
            "/nonexistent",
            "--no-token-check",
        ]
    )
    assert json.loads((output / "manifest.json").read_text())["token_check"] == "skipped"


def test_token_length_matches_the_training_collator(image_root):
    model = Path("Qwen/Qwen3.5-0.8B")
    if not model.is_dir():
        pytest.skip("local Qwen3.5-0.8B snapshot is not available")
    from dohnuts.training_data import DecisionCollator

    record = make_record("001_TaskA_step1", {"action": "system_button", "button": "Home"})
    step, reason = parse_step(record, image_root=image_root)
    assert reason is None
    rows = rows_for_step(step, str(image_root / "shot.png"))
    processor = prepare.AutoProcessor.from_pretrained(model, local_files_only=True)
    collated, *_ = DecisionCollator(model)([rows[0]])
    assert prepare.token_length(processor, rows[0]) == int(collated["input_ids"].shape[1])


def test_token_length_counts_words_and_image_patches(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    row = rows_for_step(step, str(step.image))[0]
    prompt, _, _ = prepare.render_question(
        prepare.render(row["state"]), row["question"], has_image=True
    )
    # An 8x8 screenshot resizes to 532x532: 19x19 = 361 patches at factor 28, minus
    # the single placeholder token already present in the rendered prompt.
    assert prepare.token_length(StubProcessor(), row) == len(prompt.split()) + 360
