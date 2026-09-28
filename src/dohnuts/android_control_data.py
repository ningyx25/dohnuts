"""Deterministic parsing of Android Control episodes into Dohnuts decision rows.

An Android Control episode is one directory: `metadata_{episode_id}.json` lists
the steps of the episode and every step points at a screenshot plus a
`step_NNN_a11y.json` accessibility forest. This module holds the pure rules that
turn that raw format into the vocabulary the gui-v1 pipeline already uses: the
metadata structure check, the action mapping, the clickable element list, the
ground-truth hit test, the candidate choice constraints, and the two or three
decision rows every step produces. The element rules are a port of the dataset's
own `utils/representation_utils.py` onto plain JSON, with no protobuf and no
cv2. Screenshot marking and the conversion CLI live in the callers, not here.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from PIL import Image

from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    COMPLETE_CRITERIA,
    INSTRUCTIONS,
    SWIPE_DIRECTIONS,
    image_digest,
    one_hot,
    split_for,
)

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

DATASETS = {
    "action": "gui_action",
    "complete": "gui_complete",
    "element": "screenshot_choice",
    "swipe_dir": "gui_swipe",
    "button": "gui_button",
}

# The element family asks which candidate to act on, so it reuses the prompt of
# the row the dataset itself ships for that question.
ELEMENT_INSTRUCTION = "Which action should be taken next to complete the user's task?"

# Every task_progress starts with this phrase; the gui-v1 state template uses
# the same wording for the operations the agent has already performed.
TASK_PROGRESS_PREFIX = "(You have done the following operation on the current device): "


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
    """Read an element's `[x_min, y_min, x_max, y_max]`, or None when unusable.

    Unusable covers more than a missing or mistyped field: a non-finite
    coordinate (`inf`, `nan`) and an inverted box (`x_min >= x_max` or
    `y_min >= y_max`, including the degenerate zero-area one) are refused too,
    because nothing downstream can act on them. This is the one place that
    decides, so `hit_test` never has to reason about a box it cannot compare
    against and the marking code never hands a reversed rectangle to PIL.
    """
    bounds = element.get("bounds") if isinstance(element, dict) else None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    x_min, y_min, x_max, y_max = bounds
    if not all(math.isfinite(value) for value in bounds):
        return None
    if x_min >= x_max or y_min >= y_max:
        return None
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


def read_screenshot(episode_dir: Path, name: object) -> tuple[Path, str, tuple[int, int]] | None:
    """Open one step screenshot; None when it is unusable, never raises.

    `name` is the file name the metadata step points at. Absolute paths and
    paths that escape `episode_dir` are rejected before any IO, so a record can
    never read outside its own episode. The returned size is the pixel size
    (width, height) the element rules measure bounds against, and the digest is
    of the file bytes on disk, which is what the row aliases carry.

    Only the header is validated here: `Image.open` reads the size and rejects
    decompression bombs, and `verify` adds whatever container check the format
    plugin provides -- a real chunk and CRC walk for PNG, whose plugin overrides
    it, and nothing at all for the formats whose plugin does not. A JPEG with a
    cut scan, or a PNG whose compressed data is corrupt while its CRCs stay
    consistent, is therefore accepted here, which is deliberate: decoding every
    pixel would cost a full decode whose result is thrown away, because the
    stored copy is what gets decoded again. Residual decode failures surface
    later instead -- as a per-step `unexpected` exclusion when the marking step
    decodes the screenshot, or as a `validate_rows` self-check failure at the
    end of a run -- never as a silently wrong row. That backstop is only sound
    because the stored copy is byte-identical to the source: the conversion CLI
    copies the screenshot bytes with `shutil.copyfile` instead of re-encoding.
    """
    if not isinstance(name, str) or not name:
        return None
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    image = Path(episode_dir) / relative
    try:
        if not image.is_file():
            return None
        with Image.open(image) as handle:
            size = handle.size
            handle.verify()
        return image, image_digest(image), size
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def read_a11y(episode_dir: Path, name: object) -> dict | None:
    """Read one `step_NNN_a11y.json`; None when it is unusable, never raises.

    The same path safety as `read_screenshot` applies. A file that is absent, is
    not JSON, or is not a `{"windows": [...]}` forest reads as unusable rather
    than as an empty forest, so a click step whose candidates cannot be known is
    excluded instead of asked about with no candidates at all.
    """
    if not isinstance(name, str) or not name:
        return None
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = Path(episode_dir) / relative
    try:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(document, dict) or not isinstance(document.get("windows"), list):
        return None
    return document


def task_progress(instructions: list[str | None]) -> str:
    """Render the history prefix of one step from its completed instructions.

    Callers pass the `step_instruction` of every step before the current one, in
    episode order. None entries, and values that are not strings, are skipped
    and the kept ones are renumbered 1..n, so the numbering never has holes. An
    empty history renders as the bare prefix plus `.`.
    """
    parts = [
        f"Step {number}: {instruction};"
        for number, instruction in enumerate(
            (value for value in instructions if isinstance(value, str)), 1
        )
    ]
    return TASK_PROGRESS_PREFIX + " ".join([*parts, "."])


@dataclass(frozen=True)
class ACStep:
    """One parsed Android Control step; `rows_for_ac_step` turns it into rows.

    `elements` and `target_element` only carry values when the mapped action is
    `click` or `long_press`; every other step, including the terminal one, has
    `elements == []` and `target_element is None`. `target_element` is a
    position into `elements`, never the `index` field of an element, and
    `arguments` are the mapped mobile_use-shaped arguments, not the raw ones.
    """

    id: str
    group: str
    state: dict
    image: Path
    image_sha256: str
    action: str
    arguments: dict
    ac_action: dict | None
    instruction: str | None
    elements: list[dict]
    target_element: int | None
    reference: dict


def parse_step(
    metadata: dict, index: int, *, episode_dir: Path
) -> tuple[ACStep | None, str | None]:
    """Parse one step of an episode; never raises, returns an exclusion reason.

    `metadata` is a record already accepted by `parse_metadata` and `index` is
    the step to parse. The envelope is checked again here, so a caller that
    skips `parse_metadata` gets a reason string instead of a traceback.

    Reasons: `unparsable_metadata` (the envelope, the index, or the step entry
    is structurally unusable). The step entry must satisfy the whole
    `validate_step` schema, including the fields this function never reads
    (`step_id`, and `accessibility_tree` on the actions that need no forest), so
    that the exclusion stays interpretable whenever it fires. Then
    `missing_image` (the screenshot is missing, unreadable, or outside the
    episode directory), `unknown_action` (`map_action` does not know the
    action), `missing_a11y` (a click or long_press step without a readable
    accessibility forest), and `resolve_element_choice`'s `too_few_candidates`,
    `too_many_candidates` and `no_target_element`. Any element reason excludes
    the whole step: a step is never partially converted. The forest is opened
    only for `click` and `long_press` steps.

    The last step of an episode has a null action; it becomes a `terminate` step
    with an empty candidate list and no instruction.
    """
    if not isinstance(metadata, dict):
        return None, "unparsable_metadata"
    steps = metadata.get("steps")
    episode_id = metadata.get("episode_id")
    goal = metadata.get("goal")
    if not is_integer(episode_id) or not isinstance(goal, str):
        return None, "unparsable_metadata"
    if not isinstance(steps, list) or not is_integer(index) or not 0 <= index < len(steps):
        return None, "unparsable_metadata"
    step = steps[index]
    # The same structure check `parse_metadata` runs, applied here so both entry
    # points agree by construction. It matters most for `action`: a step that
    # dropped the key would otherwise read as the terminal step and mint a
    # confident `terminate` answer for a step nobody knows the action of.
    if not validate_step(step):
        return None, "unparsable_metadata"
    screenshot = read_screenshot(episode_dir, step.get("screenshot"))
    if screenshot is None:
        return None, "missing_image"
    image, image_sha256, size = screenshot

    raw = step.get("action")
    instruction = step.get("step_instruction")
    instruction = instruction if isinstance(instruction, str) else None
    elements: list[dict] = []
    target_element = None
    ac_action: dict | None = None
    if raw is None:
        # The last step of an episode: no action to take, and the instruction
        # that would describe the next one does not exist either.
        action, arguments = "terminate", {"action": "terminate"}
        ac_action = None
        instruction = None
    else:
        if not isinstance(raw, dict):
            return None, "unknown_action"
        mapped = map_action(raw)
        if mapped is None:
            return None, "unknown_action"
        action, arguments = mapped
        ac_action = raw
        if action in ("click", "long_press"):
            a11y = read_a11y(episode_dir, step.get("accessibility_tree"))
            if a11y is None:
                return None, "missing_a11y"
            elements = extract_elements(a11y, screen_size=size)
            # map_action passes the raw x/y through unchanged, and this branch
            # only runs when it accepted both as finite coordinates.
            x, y = arguments["coordinate"]
            position, reason = resolve_element_choice(elements, x, y)
            if reason is not None:
                return None, reason
            target_element = position

    past = [
        entry.get("step_instruction") if isinstance(entry, dict) else None
        for entry in steps[:index]
    ]
    return (
        ACStep(
            id=f"android_control_{episode_id}_step{index}",
            group=f"task:android_control_{episode_id}",
            state={"user_query": goal, "task_progress": task_progress(past)},
            image=image,
            image_sha256=image_sha256,
            action=action,
            arguments=arguments,
            ac_action=ac_action,
            instruction=instruction,
            elements=elements,
            target_element=target_element,
            reference={
                "thought": "",
                "action": instruction or "",
                "tool_call": {"name": "mobile_use", "arguments": arguments},
                "ac_action": ac_action,
                "element_position": target_element,
            },
        ),
        None,
    )


def rows_for_ac_step(
    step: ACStep, image_path: str, marked: tuple[str, str] | None = None
) -> list[dict]:
    """Every decision row of one parsed step: action, complete, then its family.

    `step` must come from `parse_step` (it guarantees the action vocabulary and
    the candidate list this function relies on). `image_path` is the repository
    root relative path of the stored copy of the raw screenshot, not
    `step.image`. `marked` is the relative path and sha256 of the marked copy and
    is only read by the element family; it is ignored by the other families.

    Rows come back in the order action, complete, element or button or
    swipe_dir. Every row of a step shares `state`, `group` and `reference`
    **by reference** (treat those values as read-only) and carries its own copy
    of the step's alias list.

    Three programming errors raise a `ValueError`, with the same posture as
    `gui_data.rows_for_step`'s swipe guard: an element step without `marked`,
    and an element step whose `target_element` is missing or does not name one
    of its candidates (neither of which `parse_step` can produce).
    """
    element_row = step.action in ("click", "long_press")
    if element_row and marked is None:
        raise ValueError(
            "rows_for_ac_step requires the marked screenshot of a click or long_press step"
        )
    marked_path = image_path
    aliases = ["image-bytes:" + step.image_sha256]
    if element_row and marked is not None:  # the guard above rules out marked=None
        marked_path = marked[0]
        aliases.append("image-bytes:" + marked[1])
    split = split_for(step.group)

    def row(name: str, question: dict, target: list[float], image: str) -> dict:
        return {
            "id": f"{step.id}:{name}",
            "dataset": DATASETS[name],
            "group": step.group,
            "aliases": list(aliases),
            "split": split,
            "state": step.state,
            "image": image,
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
                "criteria": dict(AC_ACTIONS),
            },
            one_hot(len(AC_ACTIONS), list(AC_ACTIONS).index(step.action)),
            image_path,
        ),
        row(
            "complete",
            {
                "type": "noul",
                "instructions": INSTRUCTIONS["complete"],
                "criteria": dict(COMPLETE_CRITERIA),
            },
            [0.0, 1.0] if step.action == "terminate" else [1.0, 0.0],
            image_path,
        ),
    ]
    if element_row:
        if step.target_element is None or not 0 <= step.target_element < len(step.elements):
            raise ValueError(
                "rows_for_ac_step requires an element target that names a candidate;"
                " parse_step rejects the rest"
            )
        rows.append(
            row(
                "element",
                {
                    "type": "choice",
                    "instructions": ELEMENT_INSTRUCTION,
                    "criteria": {
                        f"r{position}": element_description(position, element)
                        for position, element in enumerate(step.elements)
                    },
                },
                one_hot(len(step.elements), step.target_element),
                marked_path,
            )
        )
    elif step.action == "system_button":
        buttons = list(BUTTONS)
        rows.append(
            row(
                "button",
                {
                    "type": "choice",
                    "instructions": INSTRUCTIONS["button"],
                    "criteria": dict(BUTTONS),
                },
                one_hot(len(buttons), buttons.index(step.arguments["button"])),
                image_path,
            )
        )
    elif step.action == "swipe":
        directions = list(SWIPE_DIRECTIONS)
        rows.append(
            row(
                "swipe_dir",
                {
                    "type": "choice",
                    "instructions": INSTRUCTIONS["swipe_dir"],
                    "criteria": dict(SWIPE_DIRECTIONS),
                },
                one_hot(len(directions), directions.index(step.arguments["direction"])),
                image_path,
            )
        )
    return rows
