# GUI Direct-Decision Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 GUI 逐步 SFT 轨迹(`raw_data.json`,每步一条)确定性地转成 Dohnuts 决策行(action 8 类 / button 4 类 / complete noul / swipe_dir 4 类),产出四个 split 与审计清单,可直接喂给 `train.ipynb` 的 C 流程。

**Architecture:** 纯函数核心放 `src/dohnuts/gui_data.py`(解析、派生、分桶、隔离(截图逐记录消歧)、自检),CLI 外壳放 `scripts/prepare_gui_data.py`(输入发现、图像复制、token 预算、manifest)。一步产 2–3 行,共享 state/image/group,一行一问题,完全复用现有 `DecisionCollator`/`Predictor`/`train.py`/`metrics.py`。规格:`docs/superpowers/specs/2026-09-27-gui-direct-decision-data-design.md`。

**Tech Stack:** Python 3.12、PIL、transformers(AutoProcessor/smart_resize)、pytest、ruff、pdm。

**约定(每个任务都适用):**

- 所有命令从仓库根目录运行(转换器会强制校验 `Path.cwd() == git rev-parse --show-toplevel`)。
- 本仓库工作区**已有他人暂存的 11 个文件**。每次提交必须显式给出路径:
  `git add <新文件> && git commit -m "<msg>" -- <同一批路径>`。**不要**用不带路径的
  `git commit`,那会连同暂存区里无关的改动一起提交。
- 代码风格:ruff 行宽 100;提交前跑 `pdm run format && pdm run lint`;`src/` 内代码还要过
  `pdm run typecheck`。
- 测试:`pdm run test`(即 pytest,从仓库根运行,不加载模型权重)。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `src/dohnuts/gui_data.py`(新建) | 词表与规则常量、`Step` 解析(`parse_step`)、行派生(`rows_for_step`)、分桶(`split_for`)、并查隔离(`isolate`)、自检(`validate_rows`) |
| `scripts/prepare_gui_data.py`(新建) | CLI:输入发现、图像按内容哈希复制、token 预算检查、写四个 jsonl + `excluded.jsonl` + `manifest.json` |
| `tests/test_gui_data.py`(新建) | 合成 fixture 驱动的单元测试与 CLI 集成测试(CI 可跑,不依赖被 gitignore 的 `example-data/`) |
| `docs/data-and-evaluation.md`(修改) | 追加 "GUI step conversion" 小节 |

---

## Task 1: 解析与校验(`parse_step`)

**Files:**
- Create: `src/dohnuts/gui_data.py`
- Test: `tests/test_gui_data.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_gui_data.py`:

```python
"""GUI step conversion: parsing, row derivation, isolation, and CLI output."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from dohnuts.gui_data import parse_step

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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pdm run pytest tests/test_gui_data.py -x -q`
Expected: FAIL —`ModuleNotFoundError: No module named 'dohnuts.gui_data'`

- [ ] **Step 3: 实现最小代码**

创建 `src/dohnuts/gui_data.py`:

```python
"""Deterministic conversion of GUI step records into Dohnuts decision rows.

One step record yields two or three rows that share state and image. Rows are
built only from the ground-truth tool call: no candidate list, element tree, or
model output is required. See
docs/superpowers/specs/2026-09-27-gui-direct-decision-data-design.md.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SPLIT_SEED = "doh-gui-split-2026"
SPLIT_ORDER = {"train": 0, "calibration": 1, "dev": 2, "test": 3}
SPLIT_LIMITS = [("calibration", 10), ("dev", 20), ("test", 30)]
DATASETS = {
    "action": "gui_action",
    "button": "gui_button",
    "complete": "gui_complete",
    "swipe_dir": "gui_swipe",
}
ACTIONS = {
    "click": "tap a single point on the screen",
    "long_press": "press and hold a point for some time",
    "swipe": "drag from one point to another",
    "type": "enter text into the active input field",
    "answer": "output the answer to the user",
    "system_button": "press a system button",
    "wait": "wait for the screen to change",
    "terminate": "finish the task and report the result",
}
BUTTONS = {
    "Back": "return to the previous screen",
    "Home": "go to the home screen",
    "Menu": "open the application menu or recents",
    "Enter": "press the enter key",
}
SWIPE_DIRECTIONS = {
    "up": "swipe towards the top edge",
    "down": "swipe towards the bottom edge",
    "left": "swipe towards the left edge",
    "right": "swipe towards the right edge",
}
INSTRUCTIONS = {
    "action": "What is the next action the agent should take?",
    "button": "Which system button should be pressed?",
    "complete": "Should the agent terminate the task now?",
    "swipe_dir": "In which direction should the screen be swiped?",
}
COMPLETE_CRITERIA = {
    "false": "no, more UI actions are needed",
    "true": "yes, the agent should terminate now",
}
STATE_PATTERN = re.compile(
    r"^The user query:\s*(?P<query>.*?)\nTask progress\s*(?P<progress>.*?)\s*$",
    re.DOTALL,
)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(?P<call>.*?)\s*</tool_call>", re.DOTALL)
REFERENCE_PATTERN = re.compile(
    r"Thought:\s*(?P<thought>.*?)\nAction:\s*(?P<action>.*?)\n<tool_call>", re.DOTALL
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def image_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Step:
    id: str
    group: str
    state: dict
    image: Path
    image_sha256: str
    arguments: dict
    reference: dict


def task_id(step_id: str) -> str:
    # Trailing provenance markers after the step number belong to the batch, not
    # the task: 42_App_step5__from0208_batch and 42_App_step5 must share a group.
    return re.sub(r"_step\d+.*$", "", step_id) or step_id


def point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in value
    ):
        return None
    return float(value[0]), float(value[1])


def swipe_direction(arguments: dict) -> str | None:
    start, end = point(arguments.get("coordinate")), point(arguments.get("coordinate2"))
    if start is None or end is None:
        return None
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) == abs(dy):
        return None
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def message_content(messages: list, role: str) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == role
            and isinstance(message.get("content"), str)
        ):
            return message["content"]
    return None


def reference(assistant: str, call: dict) -> dict:
    match = REFERENCE_PATTERN.search(assistant)
    return {
        "thought": match.group("thought").strip() if match else "",
        "action": match.group("action").strip() if match else "",
        "tool_call": call,
    }


def parse_step(record: dict, *, image_root: Path) -> tuple[Step | None, str | None]:
    """Return the parsed step, or (None, exclusion reason); never raises on bad input."""
    if not isinstance(record, dict):
        return None, "unparsable_state"
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None, "unparsable_state"
    if any(not isinstance(message, dict) for message in messages):
        return None, "unparsable_state"
    roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    ]
    users, assistants = roles.count("user"), roles.count("assistant")
    if not users and not assistants:
        return None, "unparsable_state"
    if users != 1 or assistants != 1:
        return None, "multi_turn"
    user = message_content(messages, "user")
    assistant = message_content(messages, "assistant")
    if user is None or assistant is None:
        return None, "unparsable_state"
    match = STATE_PATTERN.match(re.sub(r"<image>\s*$", "", user).strip())
    if match is None:
        return None, "unparsable_state"
    call_match = TOOL_CALL_PATTERN.search(assistant)
    if call_match is None:
        return None, "missing_tool_call"
    try:
        call = json.loads(call_match.group("call"))
    except (json.JSONDecodeError, RecursionError):
        return None, "missing_tool_call"
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return None, "missing_tool_call"
    if call.get("name") != "mobile_use":
        return None, "unknown_tool"
    step_id = record.get("id")
    if not isinstance(step_id, str) or not step_id:
        return None, "missing_id"
    arguments = call["arguments"]
    action = arguments.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        return None, "unknown_action"
    button = arguments.get("button")
    if action == "system_button" and (not isinstance(button, str) or button not in BUTTONS):
        return None, "invalid_button"
    if action == "swipe" and swipe_direction(arguments) is None:
        return None, "invalid_swipe"
    images = record.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0]
    ):
        return None, "multi_image"
    relative = Path(images[0])
    if relative.is_absolute() or ".." in relative.parts:
        return None, "missing_image"
    image = Path(image_root) / relative
    try:
        if not image.is_file():
            return None, "missing_image"
        with Image.open(image) as handle:
            handle.convert("RGB")
        image_sha256 = image_digest(image)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None, "missing_image"
    return (
        Step(
            id=step_id,
            group="task:" + task_id(step_id),
            state={
                "user_query": match.group("query"),
                "task_progress": match.group("progress"),
            },
            image=image,
            image_sha256=image_sha256,
            arguments=arguments,
            reference=reference(assistant, call),
        ),
        None,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: PASS(27 passed)

- [ ] **Step 5: 格式化、lint、typecheck**

Run: `pdm run format && pdm run lint && pdm run typecheck`
Expected: 无输出、退出码 0

- [ ] **Step 6: 提交**

```bash
git add src/dohnuts/gui_data.py tests/test_gui_data.py
git commit -m "Add GUI step parsing for direct-decision data" -- src/dohnuts/gui_data.py tests/test_gui_data.py
```

---

## Task 2: 行派生与分桶(`rows_for_step`、`split_for`)

**Files:**
- Modify: `src/dohnuts/gui_data.py`(追加函数)
- Test: `tests/test_gui_data.py`(追加测试)

- [ ] **Step 1: 写失败测试**

先把 `tests/test_gui_data.py` 顶部的 `from dohnuts.gui_data import parse_step` 替换为:

```python
from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    SWIPE_DIRECTIONS,
    parse_step,
    rows_for_step,
    split_for,
)
```

(ruff 的 `E402` 会把"非顶部 import"判为错误,所以后续任务新增的 import 一律改顶部区块。)

再在 `tests/test_gui_data.py` 末尾追加:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: FAIL —`ImportError: cannot import name 'rows_for_step'`

- [ ] **Step 3: 实现最小代码**

在 `src/dohnuts/gui_data.py` 的 `parse_step` 之后追加:

```python
def one_hot(width: int, index: int) -> list[float]:
    return [float(position == index) for position in range(width)]


def split_for(group: str) -> str:
    bucket = int(digest(f"{SPLIT_SEED}:{group}")[:8], 16) % 100
    for name, limit in SPLIT_LIMITS:
        if bucket < limit:
            return name
    return "train"


def rows_for_step(step: Step, image_path: str) -> list[dict]:
    """One action row per step, plus button, complete, and swipe_dir rows.

    `step` must come from `parse_step` (it guarantees the arguments this function
    relies on); `image_path` is the repository-root relative path of the stored
    image copy, not `step.image`. Rows come back in the order action, button,
    complete, swipe_dir. Every row of a step shares `state` and `reference` **by
    reference**: treat row values as read-only.
    """
    arguments = step.arguments
    action = arguments["action"]

    def row(name: str, question: dict, target: list[float]) -> dict:
        return {
            "id": f"{step.id}:{name}",
            "dataset": DATASETS[name],
            "group": step.group,
            "aliases": ["image-bytes:" + step.image_sha256],
            "split": split_for(step.group),
            "state": step.state,
            "image": image_path,
            "question": question,
            "target": target,
            "reference": step.reference,
        }

    rows = [
        row(
            "action",
            {
                "type": "choice",
                "instructions": INSTRUCTIONS["action"],
                "criteria": dict(ACTIONS),
            },
            one_hot(len(ACTIONS), list(ACTIONS).index(action)),
        )
    ]
    if action == "system_button":
        buttons = list(BUTTONS)
        rows.append(
            row(
                "button",
                {
                    "type": "choice",
                    "instructions": INSTRUCTIONS["button"],
                    "criteria": dict(BUTTONS),
                },
                one_hot(len(buttons), buttons.index(arguments["button"])),
            )
        )
    rows.append(
        row(
            "complete",
            {
                "type": "noul",
                "instructions": INSTRUCTIONS["complete"],
                "criteria": dict(COMPLETE_CRITERIA),
            },
            [0.0, 1.0] if action == "terminate" else [1.0, 0.0],
        )
    )
    if action == "swipe":
        directions = list(SWIPE_DIRECTIONS)
        direction = swipe_direction(arguments)
        if direction is None:
            raise ValueError(
                "rows_for_step requires distinct swipe axes; parse_step rejects the rest"
            )
        rows.append(
            row(
                "swipe_dir",
                {
                    "type": "choice",
                    "instructions": INSTRUCTIONS["swipe_dir"],
                    "criteria": dict(SWIPE_DIRECTIONS),
                },
                one_hot(len(directions), directions.index(direction)),
            )
        )
    return rows
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: PASS(32 passed)

- [ ] **Step 5: 格式化、lint、typecheck**

Run: `pdm run format && pdm run lint && pdm run typecheck`
Expected: 退出码 0

- [ ] **Step 6: 提交**

```bash
git add src/dohnuts/gui_data.py tests/test_gui_data.py
git commit -m "Derive GUI decision rows and task-hashed splits" -- src/dohnuts/gui_data.py tests/test_gui_data.py
```

---

## Task 3: 并查隔离与自检(`isolate`、`validate_rows`)

**Files:**
- Modify: `src/dohnuts/gui_data.py`(追加函数 + `Counter`/`Iterable` 导入)
- Test: `tests/test_gui_data.py`(追加测试)

- [ ] **Step 1: 写失败测试**

先改顶部 import 区块,把文件开头的 import 部分整体替换为:

```python
import hashlib
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
```

再在 `tests/test_gui_data.py` 末尾追加:

```python
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


def test_validate_rows_rejects_group_across_splits(image_root):
    step = step_for(image_root, "demo_step1", {"action": "wait", "time": 1})
    rows = rows_for_step(step, str(step.image))
    rows[1]["split"] = "test" if rows[0]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="span multiple splits"):
        validate_rows(rows)


def test_validate_rows_accepts_converted_rows(image_root):
    step = step_for(image_root, "demo_step1", {"action": "system_button", "button": "Home"})
    validate_rows(rows_for_step(step, str(step.image)))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: FAIL —`ImportError: cannot import name 'isolate'`

- [ ] **Step 3: 实现最小代码**

在 `src/dohnuts/gui_data.py` 顶部导入区补上:

```python
from collections import Counter
from collections.abc import Iterable
```

在文件末尾追加:

```python
def isolate(rows: Iterable[dict], audit: Counter, dropped: list[dict[str, str]]) -> list[dict]:
    """Resolve screenshot collisions across splits, then drop duplicates.

    Returns the kept rows and appends one `{id, reason, detail, stage}` entry per
    dropped row to `dropped`, so the CLI can report both in excluded.jsonl.

    A screenshot (identical image bytes, carried as an alias) may only live in one
    split: for every alias the highest-priority split among the rows carrying it
    wins (`train < calibration < dev < test`), and the rows carrying that alias in
    lower-priority splits are dropped as `cross_split_group`. A task's other rows
    stay in the task's own split and `group` is never rewritten. Reason precedence
    is cross-split, then duplicate row id, then duplicate content. Among mutually
    duplicate rows the first in input order survives, so callers must pass rows in
    a fixed order (the CLI sorts its input files).
    """
    pending = list(rows)
    best: dict[str, int] = {}
    for row in pending:
        for alias in row["aliases"]:
            best[alias] = max(best.get(alias, 0), SPLIT_ORDER[row["split"]])
    seen, seen_ids, kept = set(), set(), []
    for row in pending:
        if any(SPLIT_ORDER[row["split"]] < best[alias] for alias in row["aliases"]):
            audit[f"{row['dataset']}:{row['split']}:cross_split_group"] += 1
            dropped.append(
                {"id": row["id"], "reason": "cross_split_group", "detail": "", "stage": "isolate"}
            )
            continue
        if row["id"] in seen_ids:
            # Two records minted the same row id: exclude the later one instead of
            # letting validate_rows abort the whole conversion.
            audit[f"{row['dataset']}:{row['split']}:duplicate_input"] += 1
            dropped.append(
                {
                    "id": row["id"],
                    "reason": "duplicate_input",
                    "detail": "row id already seen",
                    "stage": "isolate",
                }
            )
            continue
        seen_ids.add(row["id"])
        key = digest(
            json.dumps(
                [row["dataset"], row["group"], row["state"], row["question"]], sort_keys=True
            )
        )
        if key in seen:
            audit[f"{row['dataset']}:{row['split']}:duplicate_input"] += 1
            dropped.append(
                {"id": row["id"], "reason": "duplicate_input", "detail": "", "stage": "isolate"}
            )
            continue
        seen.add(key)
        kept.append(row)
    return kept


def validate_rows(rows: Iterable[dict], *, root: Path | None = None) -> None:
    """Self-check ids, splits, targets, candidate counts, images, group isolation.

    Every failure is a `ValueError` naming the offending row, so callers can
    abort with one actionable line. `root` resolves the stored image paths
    (they are repository-root relative) and defaults to the working directory.
    """
    root = Path.cwd() if root is None else root
    seen_ids = set()
    splits_by_group: dict[str, set] = {}
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate row id: {row['id']}")
        seen_ids.add(row["id"])
        if row["split"] not in SPLIT_ORDER:
            raise ValueError(f"Unknown split: {row['id']} ({row['split']})")
        question = row["question"]
        width = 2 if question["type"] == "noul" else len(question["criteria"])
        if not 2 <= width <= 128:
            raise ValueError(f"Candidate count out of range: {row['id']}")
        if len(row["target"]) != width:
            raise ValueError(f"Target width does not match candidates: {row['id']}")
        if min(row["target"]) < 0 or abs(sum(row["target"]) - 1) > 1e-4:
            raise ValueError(f"Target is not a distribution: {row['id']}")
        try:
            with Image.open(root / row["image"]) as image:
                image.convert("RGB")
        except OSError as error:
            raise ValueError(
                f"Image is not readable: {row['id']} ({row['image']}): {error}"
            ) from error
        splits_by_group.setdefault(row["group"], set()).add(row["split"])
    leaked = [group for group, splits in splits_by_group.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Groups span multiple splits: {sorted(leaked)[:5]}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: PASS(48 passed)

- [ ] **Step 5: 格式化、lint、typecheck**

Run: `pdm run format && pdm run lint && pdm run typecheck`
Expected: 退出码 0

- [ ] **Step 6: 提交**

```bash
git add src/dohnuts/gui_data.py tests/test_gui_data.py
git commit -m "Return isolated rows eagerly and unify the self-check failure type" -- src/dohnuts/gui_data.py tests/test_gui_data.py
```

---

## Task 4: CLI 转换器(`scripts/prepare_gui_data.py`)

**Files:**
- Create: `scripts/prepare_gui_data.py`
- Test: `tests/test_gui_data.py`(追加 CLI 测试)

- [ ] **Step 1: 写失败测试**

先在顶部的标准库区块加一行 `import importlib.util`(isort 顺序:`importlib.util` 在
`import json` 之前),再在 `tests/test_gui_data.py` 末尾追加:

```python
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
```

注意:CLI 测试从仓库根运行(pytest 的工作目录),`--input`/`--output` 用 `tmp_path` 绝对路径;
行内 `image` 是相对仓库根的路径(可能是 `../../tmp/...`),`Image.open` 仍可解析。

- [ ] **Step 2: 运行测试确认失败**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: FAIL —`FileNotFoundError: scripts/prepare_gui_data.py`(importlib 加载失败)

- [ ] **Step 3: 实现最小代码**

创建 `scripts/prepare_gui_data.py`:

```python
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
        except (OSError, ValueError) as error:
            # JSONDecodeError and UnicodeDecodeError are both ValueError; a torn or
            # non-UTF-8 file must not abort the batch with a traceback.
            raise SystemExit(
                f"Unreadable step record file: {path} ({type(error).__name__}: {error})"
            ) from error
        if not isinstance(data, list):
            raise SystemExit(f"Unreadable step record file: {path} (expected a JSON array)")
        yield from data


def store_image(step, output: Path) -> Path:
    target = output / "images" / (step.image_sha256 + ".png")
    if not target.exists() or digest_file(target) != step.image_sha256:
        # Re-copy a target whose content does not match its name: a killed
        # previous run can leave a truncated file that later runs would trust.
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(step.image, target)
    return target


def convert(source: Path, output: Path, *, processor=None) -> dict:
    root = repository_root()
    if Path.cwd().resolve() != root:
        raise SystemExit(f"Run from the repository root: {root}")
    source = Path(source)
    image_root = source if source.is_dir() else source.parent
    paths = input_files(source)
    if not paths:
        raise SystemExit(f"No step record *.json found under {source}")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"Cannot create the output directory {output}: {error}") from error
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
    counts: Counter = Counter()
    classes: dict[str, Counter] = {}
    try:
        handles = {split: (output / f"{split}.jsonl").open("w") for split in SPLITS}
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
    except OSError as error:
        raise SystemExit(f"Cannot write the split files under {output}: {error}") from error
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
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
```

注意:Task 5 会给 `main` 接上 `--model`/`--no-token-check` 与 token 检查,本任务先不接;
`if __name__ == "__main__": main()` 守卫不能省,否则 `python scripts/prepare_gui_data.py ...` 什么都不做。

- [ ] **Step 4: 运行测试确认通过**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: PASS(66 passed — 本机同时具备 `Qwen/Qwen3.5-0.8B` 与 `example-data/` 时;缺任一项则相应用例 skip,数量减少)

- [ ] **Step 5: 格式化、lint、typecheck**

Run: `pdm run format && pdm run lint && pdm run typecheck`
Expected: 退出码 0

- [ ] **Step 6: 提交**

```bash
git add scripts/prepare_gui_data.py tests/test_gui_data.py
git commit -m "Add GUI decision data converter CLI" -- scripts/prepare_gui_data.py tests/test_gui_data.py
```

---

## Task 5: Token 预算检查接线

**Files:**
- Modify: `scripts/prepare_gui_data.py`
- Test: `tests/test_gui_data.py`(追加 stub 驱动测试)

- [ ] **Step 1: 写失败测试**

在 `tests/test_gui_data.py` 末尾追加:

```python
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
    prompt, _ = prepare.render_question(
        prepare.render(row["state"]), row["question"], has_image=True
    )
    # An 8x8 screenshot resizes to 532x532: 19x19 = 361 patches at factor 28, minus
    # the single placeholder token already present in the rendered prompt.
    assert prepare.token_length(StubProcessor(), row) == len(prompt.split()) + 360
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pdm run pytest tests/test_gui_data.py -q -k "token_length or token_budget"`
Expected: 选中的三个用例都失败(`-k token` 同时命中 collator 一致性用例)—— 后两个报
`AttributeError: module 'prepare_gui_data' has no attribute 'render_question'`(导入缺失);
`test_token_budget_excludes_whole_record` 则在断言处失败(`assert {} == {'parse:token_budget': 1}`),
因为 `convert` 此时还忽略 `processor`。

- [ ] **Step 3: 实现最小代码**

在 `scripts/prepare_gui_data.py` 的导入区补上下面这组(注意 isort 顺序:第三方
`PIL`/`transformers` 在前、`dohnuts.*` 在后,所以这几行会分别落到现有导入块的两段里,
不是连成一块):

```python
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH
```

在 `store_image` 之后追加:

```python
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
```

在 `convert` 中,把以 `for record in records(paths):` 开头的整个循环替换为下面这段
(唯一新增的是 `if processor is not None:` 分支,其余不变;下面按模块级缩进书写以便 ruff
检查,落盘时要整体缩进 4 格作为 `convert` 的函数体,并且**必须保留**
`images_written.add(stored.name)` —— 否则 `manifest["images"]` 会永远是空列表。
注意 `lengths = [...]` 那一行在模块缩进下恰好 100 字符、缩进 4 格后是 104 字符,所以落盘
后再跑 `pdm run format` 会把它折成三行 —— 这是预期的,折行后的形态才是仓库里的最终形态):

```python
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
            lengths = [token_length(processor, row) for row in rows_for_step(step, str(step.image))]
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
```

并把 `main` 改成:

```python
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
        try:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot load the token-check model {args.model}: {error}") from error
    convert(args.input, args.output, processor=processor)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pdm run pytest tests/test_gui_data.py -q`
Expected: PASS(71 passed — 缺本地 `Qwen/Qwen3.5-0.8B` 快照时 collator 一致性用例 skip,数量减少)

- [ ] **Step 4b: 用真实处理器核对长度估算(必须一致)**

stub 测试只钉住算术形状;训练侧 `DecisionCollator` 在任何一行超预算时直接抛
`Token budget exceeded`,所以估算必须与真实处理器逐一相等。跑:

```bash
pdm run python - <<'PY'
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.gui_data import parse_step, rows_for_step
from dohnuts.predictor import render, render_question
from dohnuts.recipe import IMAGE_PIXELS

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
factor = processor.image_processor.patch_size * processor.image_processor.merge_size
record = json.loads(Path("example-data/raw_data.json").read_text())[0]
step, reason = parse_step(record, image_root=Path("example-data"))
assert reason is None
row = rows_for_step(step, "example-data/raw_images/screenshot_step3.png")[0]
prompt, _ = render_question(render(row["state"]), row["question"], has_image=True)
text_tokens = len(processor.tokenizer(prompt, truncation=False)["input_ids"])
with Image.open(row["image"]) as image:
    width, height = smart_resize(
        image.height, image.width, factor=factor, min_pixels=IMAGE_PIXELS, max_pixels=IMAGE_PIXELS
    )[::-1]
estimated = text_tokens + (height // factor) * (width // factor) - 1
with Image.open(row["image"]) as image:
    inputs = processor.image_processor(
        images=[image.convert("RGB")],
        return_tensors="pt",
        size={"shortest_edge": IMAGE_PIXELS, "longest_edge": IMAGE_PIXELS},
    )
replacement = processor.replace_image_token(inputs, 0)
expanded = len(
    processor.tokenizer(prompt.replace(processor.image_token, replacement, 1))["input_ids"]
)
print({"estimated": estimated, "training_equivalent": expanded})
assert estimated == expanded, "converter token estimate disagrees with the training processor"
PY
```

Expected(本机 `Qwen/Qwen3.5-0.8B` + example-data,已预先验证):`{'estimated': 473,
'training_equivalent': 473}`。其它截图数值不同,但两者必须相等。真实处理器是
`patch_size=16`、`merge_size=2`(factor 32),与 stub 测试里的 factor 28 无关。

- [ ] **Step 5: 格式化、lint、typecheck**

Run: `pdm run format && pdm run lint && pdm run typecheck`
Expected: 退出码 0

- [ ] **Step 6: 提交**

```bash
git add scripts/prepare_gui_data.py tests/test_gui_data.py
git commit -m "Enforce the token budget when converting GUI steps" -- scripts/prepare_gui_data.py tests/test_gui_data.py
```

---

## Task 6: 文档小节

**Files:**
- Modify: `docs/data-and-evaluation.md`(追加到文件末尾)

- [ ] **Step 1: 追加文档**

在 `docs/data-and-evaluation.md` 末尾追加:

````markdown
## GUI step conversion

`scripts/prepare_gui_data.py` converts step-level GUI agent trajectories (user
query, completed-step history, screenshot, and the ground-truth tool call) into
decision rows. It needs no candidate list, element tree, or model output: every
row follows from the tool call alone.

One step yields two or three rows that share state and image, one question per
row:

| Question | Type | Candidates | Target |
| --- | --- | --- | --- |
| `action` | choice | click, long_press, swipe, type, answer, system_button, wait, terminate | the tool call's action |
| `button` | choice | Back, Home, Menu, Enter | the pressed button (system_button steps only) |
| `complete` | noul | false, true | whether the next action terminates the task |
| `swipe_dir` | choice | up, down, left, right | dominant axis of the swipe (swipe steps only) |

Fixed rules:

- `state` keeps the source `user_query` and `task_progress` verbatim; thought and
  action text are kept in `reference` for provenance and never enter model input.
  Records must hold exactly one user and one assistant message; anything else is
  excluded instead of guessed.
- Task ids strip the `_step<N>` suffix and form the isolation group. Split
  buckets are `int(sha256("doh-gui-split-2026:" + group)[:8], 16) % 100`:
  calibration < 10, dev < 20, test < 30, train otherwise. Image bytes join groups
  before the split priority (`train < calibration < dev < test`) is resolved, and
  a merged group keeps the lexicographically smallest task name (aliases never
  become group names). Task groups never straddle splits.
- Dataset names split by question (`gui_action`, `gui_button`, `gui_complete`,
  `gui_swipe`) so macro-F1 stays within one fixed candidate vocabulary. The
  uniform dataset sampler therefore gives each question family roughly equal
  weight, which relatively upweights button and swipe rows.
- Swipes whose axes tie on absolute delta are excluded instead of guessed.
- The token budget is checked when `--model` names a local snapshot (skip with
  `--no-token-check`). The estimate mirrors the training collator: rendered text
  tokens plus the expanded image placeholders at `IMAGE_PIXELS`. Over-budget rows
  are excluded whole, before their screenshot is stored.
- Exclusions are audited, never silent. The parse stage drops whole records
  (`unparsable_state`, `multi_turn`, `missing_tool_call`, `unknown_tool`,
  `missing_id`, `unknown_action`, `invalid_button`, `invalid_swipe`,
  `multi_image`, `missing_image`, `token_budget`, `unexpected`); the isolate
  stage drops rows (`cross_split_group`, `duplicate_input`). Both land in
  `excluded.jsonl` with `{id, reason, detail, stage}` and in the manifest's
  `exclusions` counts (`parse:<reason>` and `<dataset>:<split>:<reason>`).
- Manifest `images` lists the files this run wrote (content-deduplicated); an
  image can outlive rows that were later dropped by isolation, so it is not the
  set of images the dataset references. Re-running into the same `--output`
  reproduces the same four hashes; a different `--output` legitimately changes
  them because rows embed the stored image path.

Coordinates and typed text are payloads for the orchestrator, not decisions:
this model answers what to do, which button to press, in which direction to
swipe, and whether to stop. Region detection stays outside the converter.

Known limits: only one screenshot per step; symbol links inside the input root
can still resolve outside it; tasks that share a screen with another task are not
detected as near-duplicates.

```bash
# Run from the repository root. --input is your own directory of step-record
# *.json arrays; --model points at a local snapshot, or the token check is skipped.
pdm run python scripts/prepare_gui_data.py \
  --input data/raw/gui --output data/processed/gui-v1 --model Qwen/Qwen3.5-0.8B
```
````

- [ ] **Step 2: 构建文档确认没有语法/引用错误**

若未安装文档依赖,先跑 `pdm install -G docs`。然后:

Run: `pdm run docs`
Expected: 退出码 0(`-W` 会把警告当错误)

- [ ] **Step 3: 提交**

```bash
git add docs/data-and-evaluation.md
git commit -m "Document GUI step conversion rules" -- docs/data-and-evaluation.md
```

---

## Task 7: 端到端冒烟(需要 GPU 与本地 Qwen3.5-0.8B)

**Files:** 无仓库改动(产物在 `/tmp`)

- [ ] **Step 1: 生成 120 个合成任务**

Run:

```bash
pdm run python - <<'PY'
import json
import random
from pathlib import Path

from PIL import Image

root = Path("/tmp/gui-smoke/raw")
(root / "raw_images").mkdir(parents=True, exist_ok=True)
random.seed(7)
actions = [
    {"action": "click", "coordinate": [500, 900]},
    {"action": "long_press", "coordinate": [300, 700], "time": 2},
    {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]},
    {"action": "swipe", "coordinate": [200, 500], "coordinate2": [800, 500]},
    {"action": "type", "text": "task.html"},
    {"action": "answer", "text": "done"},
    {"action": "system_button", "button": "Back"},
    {"action": "system_button", "button": "Home"},
    {"action": "wait", "time": 2},
]
steps = []
for task in range(120):
    name = f"smoke{task:03d}_SyntheticApp"
    for index, arguments in enumerate([*random.sample(actions, 3), {"action": "terminate", "status": "success"}]):
        shot = f"raw_images/{name}_step{index + 1}.png"
        Image.new("RGB", (64, 64), (task * 7 % 256, index * 40 % 256, 128)).save(root / shot)
        steps.append({
            "id": f"{name}_step{index + 1}",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"The user query: Finish synthetic task {task}.\n"
                        "Task progress (You have done the following operation on the current device): "
                        f"Step {index}: previous actions.; .\n<image>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Thought: plan.\nAction: act.\n<tool_call>\n"
                    + json.dumps({"name": "mobile_use", "arguments": arguments})
                    + "\n</tool_call>",
                },
            ],
            "images": [shot],
            "bbox": None,
        })
(root / "steps.json").write_text(json.dumps(steps))
print(len(steps), "steps")
PY
```

Expected: `480 steps`

注意:每个合成任务必须用**不同的图片字节**(脚本按 task/index 变化颜色正是为此)。
若所有任务共用同一张截图,并查会把它们并成一个组、只保留最高优先级分区,四个 split 里
会有三个为空,后面的"四个 split 均非空"断言会以与 bug 无关的原因失败。

- [ ] **Step 2: 转换并检查每个 split 非空**

Run:

```bash
pdm run python scripts/prepare_gui_data.py --input /tmp/gui-smoke/raw --output /tmp/gui-smoke/data --model Qwen/Qwen3.5-0.8B
pdm run python - <<'PY'
import json
from collections import Counter
from pathlib import Path

manifest = json.loads(Path("/tmp/gui-smoke/data/manifest.json").read_text())
per_split = Counter()
for entry in manifest["counts"]:
    per_split[entry["split"]] += entry["n"]
print(dict(per_split))
print("exclusions:", manifest["exclusions"])
print("token_check:", manifest["token_check"])
PY
```

Expected: 四个 split 计数均 > 0;`exclusions` 为空;`token_check: enabled`。若某个 split 为 0
(概率极低),把生成脚本里的 `range(120)` 加大到 200 后重跑本节。

- [ ] **Step 3: 训练 1–2 步**

Run:

```bash
pdm run python - <<'PY'
import json
from pathlib import Path

from dohnuts.recipe import training_recipe
from dohnuts.rlcd import RLCDConfig

recipe = training_recipe(
    model="Qwen/Qwen3.5-0.8B",
    data=Path("/tmp/gui-smoke/data"),
    seed=42,
    steps=2,
    rlcd=RLCDConfig(sigma=0.3, ce_weight=1.0),
)
Path("/tmp/gui-smoke/recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
PY
pdm run python -m dohnuts.train train --config /tmp/gui-smoke/recipe.json --run /tmp/gui-smoke/run
```

Expected: 输出含 `{"kind": "dev", ...}` 与 `{"kind": "train_complete", ...}`,退出码 0。

- [ ] **Step 4: 校准、评估、导出**

注意:重试 Step 3–5 时必须换一个全新的 `--run` 目录(否则会因
`Existing run requires --resume` 退出);checkpoint 的路径见 Step 5(Run 的兄弟目录)。

Run:

```bash
pdm run python -m dohnuts.train evaluate --config /tmp/gui-smoke/recipe.json --run /tmp/gui-smoke/run
pdm run python - <<'PY'
import json
from pathlib import Path

run = Path("/tmp/gui-smoke/run")
report = json.loads((run / "evaluation.json").read_text())
print("temperatures:", json.loads((run / "temperatures.json").read_text()))
print("selected_step:", report["selected_step"])
print(
    "calibrated:",
    {k: round(v["accuracy"], 3) for k, v in report["calibrated"].items() if isinstance(v, dict)},
)
print("slices:", [(s["primitive"], s["candidates"], round(s["accuracy"], 3)) for s in report["primitive_candidate_slices"]])
PY
```

Expected: 退出码 0;`primitive_candidate_slices` 出现 `("choice", 8)`、`("noul", 2)`
等条目(合成数据上用 2 步训练,准确率没有意义,只要是有限值即可)。

- [ ] **Step 5: 用导出的 checkpoint 推理一条**

`train.py` 把导出写进 **`run` 的兄弟目录** `run.parent / "checkpoint"`(与仓库里
`runs/notebook/checkpoint` 的惯例一致),所以 `--run /tmp/gui-smoke/run` 对应的 checkpoint
是 `/tmp/gui-smoke/checkpoint`,不是 `run/checkpoint`。

Run:

```bash
pdm run python - <<'PY'
import json
from pathlib import Path

from PIL import Image

from dohnuts.predictor import Predictor
from dohnuts.training_data import load_records

predictor = Predictor.from_checkpoint("/tmp/gui-smoke/checkpoint")
groups = load_records(Path("/tmp/gui-smoke/data/train.jsonl"))
record = groups[sorted(groups)[0]][0]
state = {**record["state"], "image": Image.open(record["image"])}
answer = predictor.predict(state, {"q": record["question"]})["answers"]["q"]
print(record["dataset"], "->", answer.get("choice", answer.get("noul")), round(answer["confidence"], 4))
PY
```

Expected: 打印 dataset 名与一个合法答案(choice 标签或 noul 概率),无异常。

---

## Task 8: 真实数据转换与交付检查

**Files:** 无仓库改动;产物在 `data/processed/`(已被 gitignore)

- [ ] **Step 1: 转换真实数据**

用户提供批量数据目录后运行(路径以实际为准):

```bash
pdm run python scripts/prepare_gui_data.py \
  --input <真实数据目录> --output data/processed/gui-v1 --model Qwen/Qwen3.5-0.8B
```

- [ ] **Step 2: 检查清单**

Run:

```bash
pdm run python - <<'PY'
import json
from collections import Counter
from pathlib import Path

manifest = json.loads(Path("data/processed/gui-v1/manifest.json").read_text())
per_split = Counter()
for entry in manifest["counts"]:
    per_split[(entry["dataset"], entry["split"])] += entry["n"]
for key in sorted(per_split):
    print(key, per_split[key])
print("exclusions:", manifest["exclusions"])
print("action_classes:", manifest["action_classes"])
PY
```

逐项确认:

1. `gui_complete` 在 dev / calibration / test 每个 split 都非空,且 calibration 的
   `gui_action`(choice)与 `gui_complete`(noul)行数各 ≥ 10,否则 `fit_temperatures`
   会保持对应类型温度 1.0(不报错,但失去校准)。
2. 排除计数逐条可解释。**重点看 `*:cross_split_group`**:截图字节相同的步骤会被并查
   成一个隔离组、只保留最高优先级 split,其余按此原因丢弃——若该计数占输入比例很大
   (真实数据里首屏/锁屏/初始状态重复很常见),说明大量任务被合并,必须用
   `excluded.jsonl` 里 `stage=isolate` 的行 id 追查受影响的任务,再决定是接受损失还是
   先剔除这些重复截图步骤(后者会改变输出哈希,必须在训练前做)。
3. `action_classes` 各 split 的类别分布与整体相近;若某 split 缺关键类别(如
   `system_button`),记录在案并考虑扩大数据或调整分桶比例常量 `SPLIT_LIMITS`。
4. `token_check` 应为 `"enabled"`(即确实传了 `--model`);若存在 `parse:token_budget`
   计数,逐条确认那些行确实超预算,而不是估算与训练侧不一致(训练侧 `DecisionCollator`
   一旦遇到超预算行会直接中止,不会静默截断)。
5. 转换后不要再改 `data/processed/gui-v1` 下的 jsonl(`train.py` 会比对 SHA-256)。
6. 首次训练前,把 `train.ipynb` 的 `DATA_DIR` 从 `example-data` 改为 `data/processed/gui-v1`,
   并把 `SMOKE_DATA` 改为 `False`(真实数据无跨 split 泄漏);`TOKEN_CHECK_ROWS` 保留,
   它会用真实 collator 再核一遍 token 预算。

---

## 验收(全部任务完成后)

1. `pdm run test` 全绿(不加载模型权重)。
2. `pdm run check` 通过(lint + format-check + typecheck)。
3. `pdm run python scripts/prepare_gui_data.py --input /tmp/gui-smoke/raw --output /tmp/gui-smoke/data --model Qwen/Qwen3.5-0.8B` 自检通过、四个 split 非空。
4. Task 7 的 train → evaluate → predict 冒烟跑通。
5. 真实数据 manifest 通过 Task 8 检查清单。
