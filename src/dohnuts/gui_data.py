"""Deterministic conversion of GUI step records into Dohnuts decision rows.

One step record yields two to four rows that share state and image. Rows are
built only from the ground-truth tool call: no candidate list, element tree, or
model output is required. See
docs/superpowers/specs/2026-09-27-gui-direct-decision-data-design.md.
"""

import hashlib
import json
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
    return re.sub(r"_step\d+$", "", step_id) or step_id


def point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
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
    for message in messages:
        if message.get("role") == role and isinstance(message.get("content"), str):
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
    """Return the parsed step, or (None, exclusion reason)."""
    user = message_content(record.get("messages") or [], "user")
    assistant = message_content(record.get("messages") or [], "assistant")
    if user is None or assistant is None:
        return None, "unparsable_state"
    match = STATE_PATTERN.match(user.replace("<image>", "").strip())
    if match is None:
        return None, "unparsable_state"
    call_match = TOOL_CALL_PATTERN.search(assistant)
    if call_match is None:
        return None, "missing_tool_call"
    try:
        call = json.loads(call_match.group("call"))
    except json.JSONDecodeError:
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
    if action not in ACTIONS:
        return None, "unknown_action"
    if action == "system_button" and arguments.get("button") not in BUTTONS:
        return None, "invalid_button"
    if action == "swipe" and swipe_direction(arguments) is None:
        return None, "invalid_swipe"
    images = record.get("images") or []
    if len(images) != 1:
        return None, "multi_image"
    image = Path(image_root) / str(images[0])
    if not image.is_file():
        return None, "missing_image"
    try:
        with Image.open(image) as handle:
            handle.convert("RGB")
    except OSError:
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
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            arguments=arguments,
            reference=reference(assistant, call),
        ),
        None,
    )
