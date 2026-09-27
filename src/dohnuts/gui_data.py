"""Deterministic conversion of GUI step records into Dohnuts decision rows.

One step record yields two or three rows that share state and image. Rows are
built only from the ground-truth tool call: no candidate list, element tree, or
model output is required. See
docs/superpowers/specs/2026-09-27-gui-direct-decision-data-design.md.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
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


def isolate(rows: Iterable[dict], audit: Counter, dropped: list[dict[str, str]]) -> list[dict]:
    """Union groups sharing image bytes, keep the top partition, drop duplicates.

    Returns the kept rows and appends one `{id, reason, detail, stage}` entry per
    dropped row to `dropped`, so the CLI can report both in excluded.jsonl.

    This mutates every input row (dropped ones included): `row["group"]` becomes
    the canonical union root, and deduplication deliberately runs after that
    merge so merged tasks dedup against each other. Reason precedence is
    cross-split, then duplicate row id, then duplicate content. Among mutually
    duplicate rows the first in input order survives, so callers must pass rows
    in a fixed order (the CLI sorts its input files).
    """
    pending = list(rows)
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    # Aliases never become union nodes: they only record which group a screenshot
    # was first seen with, so a merged group keeps a task-group name instead of an
    # image hash.
    first_group: dict[str, str] = {}
    for row in pending:
        for alias in row["aliases"]:
            if alias in first_group:
                left, right = find(first_group[alias]), find(row["group"])
                if left != right:
                    # Name a merged group after its lexicographically smallest
                    # member so the result never depends on input shard order.
                    parents[max(left, right)] = min(left, right)
            else:
                first_group[alias] = row["group"]
    priority: dict[str, int] = {}
    for row in pending:
        root = find(row["group"])
        priority[root] = max(priority.get(root, 0), SPLIT_ORDER[row["split"]])
    seen, seen_ids, kept = set(), set(), []
    for row in pending:
        row["group"] = find(row["group"])
        if SPLIT_ORDER[row["split"]] != priority[row["group"]]:
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
