"""Deterministic parsing of Android Control episodes into Dohnuts decision rows.

An Android Control episode is one directory: `metadata_{episode_id}.json` lists
the steps of the episode and every step points at a screenshot plus a
`step_NNN_a11y.json` accessibility forest. This module holds the pure rules that
turn that raw format into the vocabulary the gui-v1 pipeline already uses: the
metadata structure check, the action mapping, the clickable element list, the
ground-truth hit test, and the candidate choice constraints. The element rules
are a port of the dataset's own `utils/representation_utils.py` onto plain JSON,
with no protobuf and no cv2. Row building, screenshot marking, and the
conversion CLI live in the callers, not here.
"""

import json
import math
from typing import TypeGuard

from dohnuts.gui_data import ACTIONS

# Android Control adds `open_app` to the gui-v1 vocabulary. It is appended last
# so indices 0..7 keep their gui-v1 meaning and one-hot targets stay comparable
# across the two pipelines.
AC_ACTIONS = {**ACTIONS, "open_app": "open an app by name"}

# Android Control records where the finger goes, gui-v1 records where the
# content goes: scrolling down reveals content below the fold, which means the
# finger moves up. Flip the alignment here and nowhere else.
SCROLL_TO_SWIPE_DIRECTION = {"down": "up", "up": "down", "left": "right", "right": "left"}

SYSTEM_BUTTONS = {"navigate_back": "Back", "navigate_home": "Home"}

# Choice questions need at least two candidates and at most 128, the same
# limits `dohnuts.gui_data.validate_rows` enforces on every written row.
MIN_CANDIDATES = 2
MAX_CANDIDATES = 128


def is_integer(value: object) -> TypeGuard[int]:
    """True for a JSON integer; booleans are not ids, pixels, or indices."""
    return isinstance(value, int) and not isinstance(value, bool)


def coordinate(value: object) -> int | float | None:
    """Read a screen coordinate: any finite number, but never a bool.

    The value comes back unchanged, so integer pixels stay integers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value):
            return None
    except OverflowError:
        return None
    return value


def integer_field(value: object) -> int:
    """Read a proto3 JSON integer field; absent or malformed values read as 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return int(value)
    except (ValueError, OverflowError):
        return 0


def read_text(value: object) -> str:
    """Read a proto3 JSON string field; absent and non-string values read as ""."""
    return value if isinstance(value, str) else ""


def read_bounds(node: dict) -> list[int]:
    """Read `boundsInScreen` as `[x_min, y_min, x_max, y_max]` in pixels.

    proto3 JSON omits default values, so a missing key, or a whole missing
    `boundsInScreen`, reads as 0.
    """
    raw = node.get("boundsInScreen") if isinstance(node, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    return [
        integer_field(raw.get("left")),
        integer_field(raw.get("top")),
        integer_field(raw.get("right")),
        integer_field(raw.get("bottom")),
    ]


def map_action(raw: dict) -> tuple[str, dict] | None:
    """Map one Android Control action to `(dohnuts action, tool-call arguments)`.

    Returns None for an unsupported `action_type` or a field of the wrong type,
    which callers report as `unknown_action`. Coordinate fields accept any
    finite int or float, never a bool, and are passed through unchanged; keys
    the mapped action does not use are ignored.
    """
    if not isinstance(raw, dict):
        return None
    action_type = raw.get("action_type")
    if not isinstance(action_type, str):
        return None
    if action_type in ("click", "long_press"):
        x, y = coordinate(raw.get("x")), coordinate(raw.get("y"))
        if x is None or y is None:
            return None
        return action_type, {"action": action_type, "coordinate": [x, y]}
    if action_type == "scroll":
        direction = raw.get("direction")
        if not isinstance(direction, str) or direction not in SCROLL_TO_SWIPE_DIRECTION:
            return None
        return "swipe", {"action": "swipe", "direction": SCROLL_TO_SWIPE_DIRECTION[direction]}
    if action_type == "input_text":
        text = raw.get("text")
        if not isinstance(text, str):
            return None
        return "type", {"action": "type", "text": text}
    if action_type == "wait":
        return "wait", {"action": "wait"}
    if action_type == "open_app":
        app_name = raw.get("app_name")
        if not isinstance(app_name, str):
            return None
        return "open_app", {"action": "open_app", "app_name": app_name}
    if action_type in SYSTEM_BUTTONS:
        return "system_button", {"action": "system_button", "button": SYSTEM_BUTTONS[action_type]}
    return None


def validate_element(node: dict, *, screen_size: tuple[int, int]) -> bool:
    """Port of the dataset's `validate_ui_element` onto one accessibility node.

    `screen_size` is the screenshot size as (width, height) in pixels. Invisible
    elements are invalid, as are elements whose pixel box is degenerate or lies
    entirely off screen. Bounds come from `boundsInScreen` with missing keys as
    0, so an element without a usable box never validates. Clickability is not
    checked here; `extract_elements` filters on it.
    """
    if not isinstance(node, dict) or node.get("isVisibleToUser") is not True:
        return False
    width, height = screen_size
    x_min, y_min, x_max, y_max = read_bounds(node)
    return not (
        x_min >= x_max
        or x_min >= width
        or x_max <= 0
        or y_min >= y_max
        or y_min >= height
        or y_max <= 0
    )


def extract_elements(a11y: dict, *, screen_size: tuple[int, int]) -> list[dict]:
    """List the clickable, visible UI elements of one accessibility forest.

    Windows are visited in file order and nodes in list order. A node is kept
    when it is clickable and passes `validate_element`, so clickable containers
    with children survive and nothing is deduplicated. `index` is the position
    in the returned list and runs contiguously from 0: dropped nodes never
    consume an index, so it cannot be read off the raw node list.

    Elements carry only `index`, `text`, `content_description`, and `bounds`
    (as `[x_min, y_min, x_max, y_max]`); node flags are deliberately dropped so
    they cannot leak into a prompt later. Malformed containers are skipped
    instead of raising, and a forest that cannot be walked yields an empty list,
    which `resolve_element_choice` reports as `too_few_candidates`.
    """
    elements = []
    windows = a11y.get("windows") if isinstance(a11y, dict) else None
    if not isinstance(windows, list):
        return elements
    for window in windows:
        tree = window.get("tree") if isinstance(window, dict) else None
        nodes = tree.get("nodes") if isinstance(tree, dict) else None
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict) or node.get("isClickable") is not True:
                continue
            if not validate_element(node, screen_size=screen_size):
                continue
            elements.append(
                {
                    "index": len(elements),
                    "text": read_text(node.get("text")),
                    "content_description": read_text(node.get("contentDescription")),
                    "bounds": read_bounds(node),
                }
            )
    return elements


def element_bounds(element: dict) -> tuple[float, float, float, float] | None:
    """Read an element's `[x_min, y_min, x_max, y_max]`, or None when unusable."""
    bounds = element.get("bounds") if isinstance(element, dict) else None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    x_min, y_min, x_max, y_max = bounds
    return x_min, y_min, x_max, y_max


def hit_test(elements: list[dict], x: float, y: float) -> int | None:
    """Position of the smallest element containing (x, y), or None on a miss.

    The result is a list position into the very `elements` list passed in, so
    callers can index it directly; `extract_elements` numbers its elements so
    that the position and the `index` field agree, but only the position is
    returned here. An input that is not a list, an element without a usable
    box, and a point that is not a finite number are all ignored.

    Bounds are closed on both ends. Equal areas go to the earlier position, so
    nested and identical boxes resolve deterministically.
    """
    if not isinstance(elements, list) or coordinate(x) is None or coordinate(y) is None:
        return None
    best: tuple[float, int] | None = None
    for position, element in enumerate(elements):
        bounds = element_bounds(element)
        if bounds is None:
            continue
        x_min, y_min, x_max, y_max = bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        area = (x_max - x_min) * (y_max - y_min)
        if best is None or (area, position) < best:
            best = (area, position)
    return None if best is None else best[1]


def resolve_element_choice(
    elements: list[dict], x: float, y: float
) -> tuple[int | None, str | None]:
    """Resolve the ground-truth element for a click at pixel (x, y).

    The returned position indexes the same `elements` list the caller passed in.
    The count checks run first, so an unusable candidate list, including one
    that is not a list at all, is reported even when the point would hit
    nothing. Otherwise the point must land inside an element and a miss
    excludes the step as `no_target_element`.
    """
    if not isinstance(elements, list) or len(elements) < MIN_CANDIDATES:
        return None, "too_few_candidates"
    if len(elements) > MAX_CANDIDATES:
        return None, "too_many_candidates"
    position = hit_test(elements, x, y)
    if position is None:
        return None, "no_target_element"
    return position, None


def element_description(index: int, element: dict) -> str:
    """Render the prompt option for one candidate: `UI element {index}: {payload}`.

    `index` is the number to print; for `extract_elements` output it is also the
    element's position in the list. The `UI element {index}:` prefix is the
    prompt label. The payload that
    follows it is JSON holding only a non-empty `text` and/or
    `content_description`, in that key order, and `{}` when the element has
    neither: node flags such as `is_clickable` never reach the payload, so they
    cannot leak into a prompt, and non-ASCII text stays readable.
    """
    element = element if isinstance(element, dict) else {}
    text = read_text(element.get("text"))
    description = read_text(element.get("content_description"))
    payload = {}
    if text:
        payload["text"] = text
    if description:
        payload["content_description"] = description
    return f"UI element {index}: {json.dumps(payload, ensure_ascii=False)}"


def validate_step(step: object) -> bool:
    """True when a metadata step entry carries every field the callers read.

    All five keys must be present: `step_id` is an int, `screenshot` and
    `accessibility_tree` are non-empty file names, `action` is a dict or null,
    and `step_instruction` is a string or null. The last step of every episode
    has a null action and stays valid; an `action_type` that `map_action` does
    not know is not a structure problem either.
    """
    if not isinstance(step, dict):
        return False
    if not is_integer(step.get("step_id")):
        return False
    for key in ("screenshot", "accessibility_tree"):
        value = step.get(key)
        if not isinstance(value, str) or not value:
            return False
    if "action" not in step or "step_instruction" not in step:
        return False
    action = step["action"]
    if action is not None and not isinstance(action, dict):
        return False
    instruction = step["step_instruction"]
    return instruction is None or isinstance(instruction, str)


def parse_metadata(record: object) -> tuple[dict | None, str | None]:
    """Validate one `metadata_{episode_id}.json` document; never raises.

    Returns the record itself, or `(None, "unparsable_metadata")` when the
    episode envelope or any step entry has the wrong shape.
    """
    if not isinstance(record, dict):
        return None, "unparsable_metadata"
    if not is_integer(record.get("episode_id")):
        return None, "unparsable_metadata"
    if not isinstance(record.get("goal"), str):
        return None, "unparsable_metadata"
    steps = record.get("steps")
    if not isinstance(steps, list) or not all(validate_step(step) for step in steps):
        return None, "unparsable_metadata"
    return record, None
