"""GUI step conversion: parsing, row derivation, isolation, and CLI output."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    SWIPE_DIRECTIONS,
    parse_step,
    rows_for_step,
    split_for,
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
        image_root, "645_BrowserMaze_step3", {"action": "system_button", "button": "Back"}
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
    assert action["question"]["criteria"] == ACTIONS
    assert action["question"]["type"] == "choice"
    assert action["target"][list(ACTIONS).index("system_button")] == 1.0
    assert sum(action["target"]) == 1.0
    assert button["target"] == [1.0, 0.0, 0.0, 0.0]
    assert list(button["question"]["criteria"]) == list(BUTTONS)
    assert complete["question"]["type"] == "noul"
    assert complete["target"] == [1.0, 0.0]
    assert action["split"] == split_for("task:645_BrowserMaze")
    assert action["aliases"] == ["image-bytes:" + step.image_sha256]
    assert action["reference"]["tool_call"]["arguments"]["button"] == "Back"


def test_terminate_step_rows(image_root):
    step = step_for(image_root, "demo_step2", {"action": "terminate", "status": "success"})
    rows = rows_for_step(step, "img.png")
    assert [row["id"] for row in rows] == ["demo_step2:action", "demo_step2:complete"]
    assert rows[0]["target"][list(ACTIONS).index("terminate")] == 1.0
    assert rows[1]["target"] == [0.0, 1.0]


def test_swipe_step_rows_use_dominant_axis(image_root):
    arguments = {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]}
    rows = rows_for_step(step_for(image_root, "demo_step1", arguments), "img.png")
    assert [row["id"] for row in rows][-1] == "demo_step1:swipe_dir"
    swipe = rows[-1]
    assert swipe["dataset"] == "gui_swipe"
    assert list(swipe["question"]["criteria"]) == list(SWIPE_DIRECTIONS)
    assert swipe["target"][list(SWIPE_DIRECTIONS).index("up")] == 1.0


def test_wait_step_rows_have_no_conditional_row(image_root):
    rows = rows_for_step(
        step_for(image_root, "demo_step1", {"action": "wait", "time": 2}), "img.png"
    )
    assert [row["id"] for row in rows] == ["demo_step1:action", "demo_step1:complete"]


def test_split_for_is_stable_and_covers_partitions():
    assert split_for("task:645_BrowserMaze") == "train"
    assert split_for("task:demo") == "test"
    assert split_for("task:beta") == "test"
    assert split_for("task:delta") == "calibration"
