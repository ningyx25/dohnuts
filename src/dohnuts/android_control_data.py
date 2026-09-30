"""Android Control steps as mobile-jev shaped decision rows.

An Android Control episode is one directory: `metadata_{episode_id}.json` lists
the steps of the episode and every step points at a screenshot plus a
`step_NNN_a11y.json` accessibility forest. This module turns that raw format
into decision rows whose prompt is the one `ClientMobileJev` would send for the
same screen and goal: the state and the questions come from
`dohnuts.mobile_jev_prompt`, the candidate space is derived from the same
accessibility tree, and the ground truth is the recorded action.

What the corpus can and cannot express decides which rows exist:

- `operation` exists for every step whose recorded action maps to an operation
  the screen offers.
- `tap_target` exists for a click whose point lands on a TAP candidate.
- `text_value` exists when the typed text is one of the goal's n-gram spans,
  which is the only text the agent offers.
- `app_target` exists when the opened app is offered and the question has at
  least two options.
- `scroll_target` is never produced: Android Control records a scroll direction
  but no coordinates, so the scrolled region cannot be identified.

The readers (metadata, screenshot, accessibility forest) keep the contract they
had before: they never raise, they refuse paths that escape the episode
directory, and they report one exclusion reason per step instead of aborting a
batch. Screenshot marking and the conversion CLI live in the callers, not here.
"""

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, TypeGuard

from PIL import Image

from dohnuts import mobile_jev_prompt as jev
from dohnuts.gui_data import image_digest, one_hot, split_for

# Choice questions need at least two candidates and at most 128, the same
# limits `dohnuts.gui_data.validate_rows` enforces on every written row. The
# agent allows 255, so a screen with 129..255 candidates loses its target row
# rather than the whole step.
MIN_CANDIDATES = 2
MAX_CANDIDATES = 128

DATASETS = {
    "operation": "jev_operation",
    "tap_target": "jev_tap_target",
    "scroll_target": "jev_scroll_target",
    "text_value": "jev_text_value",
    "app_target": "jev_app_target",
}

# The mapped actions (`map_action` output) that stand for one agent operation
# each. Scrolls, the terminal step and the system buttons are handled in
# `step_operation`, which needs the raw action to tell them apart. The scroll
# mapping is the identity on purpose: AC's `scroll: down` reveals content below
# the fold and mobile-jev's SCROLL_DOWN is the same gesture. The finger
# vocabulary used by gui-v1 lives in `SCROLL_TO_SWIPE_DIRECTION` below and must
# not be reused here.
STEP_OPERATIONS = {
    "click": "TAP",
    "long_press": "TAP",
    "type": "TYPE_TEXT",
    "wait": "WAIT",
    "open_app": "OPEN_APP",
}
SCROLL_OPERATIONS = {
    "down": "SCROLL_DOWN",
    "up": "SCROLL_UP",
    "left": "SCROLL_LEFT",
    "right": "SCROLL_RIGHT",
}

# Android Control records where the content goes, gui-v1 where the finger goes:
# `scroll: down` reveals content below the fold, so the finger moves up. The
# inversion was checked against the corpus -- on 138 real vertical scroll steps
# the content moved up in 28 coherent cases against 10 moving down, and the step
# instructions agree ("Swipe up for Product details" on `scroll: down` steps).
# This only shapes the `tool_call` provenance kept in `reference` and the mapped
# arguments; the operation rows use the identity mapping above.
SCROLL_TO_SWIPE_DIRECTION = {"down": "up", "up": "down", "left": "right", "right": "left"}

SYSTEM_BUTTONS = {"navigate_back": "Back", "navigate_home": "Home"}

# The keyboard window is never the foreground app a goal is about.
IME_PACKAGES = frozenset(("com.google.android.inputmethod.latin",))

# Why a step or one of its families produced no row. Step-level reasons exclude
# the whole step; family reasons drop a single row and leave the operation row in
# place.
OPERATION_NOT_OFFERED = "operation_not_offered"
NO_TARGET_ELEMENT = "no_target_element"
TOO_FEW_CANDIDATES = "too_few_candidates"
TOO_MANY_CANDIDATES = "too_many_candidates"
TEXT_NOT_A_GOAL_SPAN = "text_not_a_goal_span"
APP_NOT_OFFERED = "app_not_offered"


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
    """Read `boundsInScreen` as `[left, top, right, bottom]` in pixels.

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


def step_problem(step: object) -> str | None:
    """Name the first field a metadata step entry gets wrong, or None when it is valid.

    The fields are checked in the order the callers read them: `step_id` is an
    int, `screenshot` and `accessibility_tree` are non-empty file names, `action`
    is a dict or null, and `step_instruction` is a string or null. The last step
    of every episode has a null action and stays valid; an `action_type` that
    `map_action` does not know is not a structure problem either. The returned
    string is phrased to follow `step {index}:` and never quotes a value, so a
    malformed record cannot smuggle its content into a log line.
    """
    if not isinstance(step, dict):
        return "step entry is not an object"
    if not is_integer(step.get("step_id")):
        return "key 'step_id' is not an integer"
    for key in ("screenshot", "accessibility_tree"):
        value = step.get(key)
        if not isinstance(value, str) or not value:
            return f"key '{key}' is not a non-empty file name"
    if "action" not in step:
        return "missing key 'action'"
    if "step_instruction" not in step:
        return "missing key 'step_instruction'"
    action = step["action"]
    if action is not None and not isinstance(action, dict):
        return "key 'action' is neither an object nor null"
    instruction = step["step_instruction"]
    if instruction is not None and not isinstance(instruction, str):
        return "key 'step_instruction' is neither a string nor null"
    return None


def validate_step(step: object) -> bool:
    """True when a metadata step entry carries every field the callers read.

    The rules live in `step_problem`, which names the field that failed; this is
    the same walk with the name thrown away.
    """
    return step_problem(step) is None


def metadata_detail(record: object) -> str:
    """Explain the first structural failure of a metadata document, or "" when valid.

    The envelope is checked before the steps, and the steps in list order, so
    the detail names the first thing a reader would have to fix: `key 'goal' is
    not a string`, then `step 1: missing key 'action'` and the like. It is the
    single source of truth behind `parse_metadata`, so the reason and the detail
    can never disagree about whether a document is usable.
    """
    if not isinstance(record, dict):
        return "document is not an object"
    if not is_integer(record.get("episode_id")):
        return "key 'episode_id' is not an integer"
    if not isinstance(record.get("goal"), str):
        return "key 'goal' is not a string"
    steps = record.get("steps")
    if not isinstance(steps, list):
        return "key 'steps' is not a list"
    for index, step in enumerate(steps):
        problem = step_problem(step)
        if problem is not None:
            return f"step {index}: {problem}"
    return ""


def parse_metadata(record: object) -> tuple[dict | None, str | None]:
    """Validate one `metadata_{episode_id}.json` document; never raises.

    Returns the record itself, or `(None, "unparsable_metadata")` when the
    episode envelope or any step entry has the wrong shape; `metadata_detail`
    names the field that failed.

    The type check repeats what `metadata_detail` already refuses, so that the
    caller gets a `dict` rather than an `object` back.
    """
    if not isinstance(record, dict) or metadata_detail(record):
        return None, "unparsable_metadata"
    return record, None


def read_screenshot(episode_dir: Path, name: object) -> tuple[Path, str, tuple[int, int]] | None:
    """Open one step screenshot; None when it is unusable, never raises.

    `name` is the file name the metadata step points at. Absolute paths and
    paths that escape `episode_dir` are rejected before any IO, so a record can
    never read outside its own episode. The returned size is the pixel size
    (width, height) the element rules measure bounds against, and the digest is
    of the file bytes on disk, which is what the row aliases carry.

    The whole image is decoded here, not just its header. A container check is
    not enough: a PNG whose chunk CRCs are consistent over a compressed stream
    that does not decompress, or that decompresses to scanlines the decoder
    rejects, passes `Image.verify` and then fails at the first real decode --
    which, for the steps nothing else decodes, used to be the `validate_rows`
    self-check at the very end of a batch, aborting a whole run over one bad
    file. Decoding per step instead turns that into this step's own
    `missing_image`, and the copy stored alongside the rows is byte-identical to
    what was decoded here, so the failure cannot come back later.
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
            handle.convert("RGB")
        return image, image_digest(image), size
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def read_a11y(episode_dir: Path, name: object) -> dict | None:
    """Read one `step_NNN_a11y.json`; None when it is unusable, never raises.

    The same path safety as `read_screenshot` applies. A file that is absent, is
    not JSON, or is not a `{"windows": [...]}` forest reads as unusable rather
    than as an empty forest: every row now carries the screen it was decided on,
    so a step whose screen cannot be known would otherwise be converted into a
    prompt describing a blank device.
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


def iter_nodes(document: Any):
    """Yield every accessibility node in window order, then node order.

    This is the order the agent flattens `state.ui_elements` in, so the position
    of a node here is the element id a prompt refers to. Malformed containers are
    skipped rather than raising: a forest that cannot be walked yields nothing.
    """
    windows = document.get("windows") if isinstance(document, dict) else None
    if not isinstance(windows, list):
        return
    for window in windows:
        tree = window.get("tree") if isinstance(window, dict) else None
        nodes = tree.get("nodes") if isinstance(tree, dict) else None
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if isinstance(node, dict):
                yield node


def clamped_bounds(node: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The node's pixel box clamped to the screen, or None when unusable.

    The agent clamps the same way before it decides whether an element exists at
    all, so an element that is entirely off screen, or whose box is degenerate
    after clamping, never reaches a prompt and never consumes an element id.
    """
    x_min, y_min, x_max, y_max = read_bounds(node)
    left, top = max(0, x_min), max(0, y_min)
    right, bottom = min(width, x_max), min(height, y_max)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def foreground_package(document: Any) -> str:
    """The package of the largest visible application window, IME excluded.

    Android Control records every window of the screen, so the foreground app
    has to be inferred: among `TYPE_APPLICATION` windows the largest area wins,
    ties go to the window that comes first in the file, and a window is
    represented by the package most of its nodes carry. A screen with no
    application window falls back to the same rule over every window type, which
    keeps a prompt's `app` field filled instead of empty.
    """
    best: tuple[tuple[int, int], str] | None = None
    fallback: tuple[tuple[int, int], str] | None = None
    windows = document.get("windows") if isinstance(document, dict) else None
    if not isinstance(windows, list):
        return ""
    for order, window in enumerate(windows):
        if not isinstance(window, dict):
            continue
        packages = Counter(
            node.get("packageName")
            for node in iter_nodes({"windows": [window]})
            if isinstance(node.get("packageName"), str) and node.get("packageName")
        )
        if not packages:
            continue
        package = packages.most_common(1)[0][0]
        if package in IME_PACKAGES:
            continue
        left, top, right, bottom = read_bounds(window)
        area = max(0, right - left) * max(0, bottom - top)
        key = (area, -order)
        if fallback is None or key > fallback[0]:
            fallback = (key, package)
        if window.get("windowType") == "TYPE_APPLICATION" and (best is None or key > best[0]):
            best = (key, package)
    chosen = best or fallback
    return chosen[1] if chosen else ""


def summarize_observation(
    document: Any,
    *,
    screen_size: tuple[int, int],
    package_name: str,
) -> jev.Observation:
    """Flatten one accessibility forest into an agent-shaped observation.

    The filter chain is the one `ClientMobileJev.summarize_state` applies to
    `state.ui_elements`: invisible nodes are dropped, nodes whose clamped box is
    empty are dropped, and a node with no text, no description and no
    interactive flag is dropped as decoration. Everything that survives keeps
    its position in the flattened node list as its element id, which is what a
    prompt's indices refer to.

    Node flags stay inside: only `editable`, `scrollable`, `checked` and
    `selected` ever reach a prompt, and only through the element entries
    `mobile_jev_prompt.build_questions` builds.
    """
    width, height = int(screen_size[0]), int(screen_size[1])
    elements: list[jev.Element] = []
    for index, node in enumerate(iter_nodes(document)):
        if node.get("isVisibleToUser") is not True:
            continue
        bounds = clamped_bounds(node, width, height)
        if bounds is None:
            continue
        text = read_text(node.get("text"))
        label = read_text(node.get("contentDescription"))
        class_name = read_text(node.get("className"))
        clickable = node.get("isClickable") is True
        editable = node.get("isEditable") is True or class_name in jev.EDITABLE_CLASS_NAMES
        scrollable = node.get("isScrollable") is True
        if not (text or label or clickable or editable or scrollable):
            continue
        elements.append(
            jev.Element(
                id=str(index),
                bounds=bounds,
                text=text,
                label=label,
                hint=read_text(node.get("hintText")),
                resource_id=read_text(node.get("viewIdResourceName")),
                class_name=class_name,
                clickable=clickable,
                editable=editable,
                scrollable=scrollable,
                enabled=node.get("isEnabled") is not False,
                focused=node.get("isFocused") is True,
                checkable=node.get("isCheckable") is True,
                checked=node.get("isChecked") is True,
                selected=node.get("isSelected") is True,
            )
        )
    return jev.make_observation(
        jev.phone_from(package_name, elements), (width, height), elements
    )


def element_bounds(element: Any) -> tuple[float, float, float, float] | None:
    """Read an element's `(left, top, right, bottom)`, or None when unusable.

    Unusable covers more than a missing or mistyped field: a non-finite
    coordinate (`inf`, `nan`) and an inverted box (`left >= right` or
    `top >= bottom`, including the degenerate zero-area one) are refused too,
    because nothing downstream can act on them. This is the one place that
    decides, so the hit rules never have to reason about a box they cannot
    compare against and the marking code never hands a reversed rectangle to
    PIL. Every extent that gets through is positive and finite, but their
    product is not bounded by that -- `element_target_weights` owns the range
    checks on the area itself.
    """
    bounds = getattr(element, "bounds", None)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    left, top, right, bottom = bounds
    try:
        if not all(math.isfinite(value) for value in bounds):
            return None
    except OverflowError:  # an int too large to convert to a float
        return None
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def element_hits(elements: list, x: float, y: float) -> list[int]:
    """Ascending positions of every element whose bounds contain (x, y).

    The positions index the very `elements` list passed in, so callers can look
    the elements up directly; an element without a usable box, and a point that
    is not a finite number, are both ignored. Bounds are closed on both ends, so
    a point on an edge or a corner counts as inside every box that shares it.
    """
    if not isinstance(elements, list) or coordinate(x) is None or coordinate(y) is None:
        return []
    hits = []
    for position, element in enumerate(elements):
        bounds = element_bounds(element)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        if left <= x <= right and top <= y <= bottom:
            hits.append(position)
    return hits


def element_target_weights(elements: list, x: float, y: float) -> list[float] | None:
    """The ground-truth distribution of a click at pixel (x, y), or None on a miss.

    Every element whose bounds contain the point is ground truth, weighted by
    inverse box area and normalized over the hits, so a smaller box -- the more
    specific target -- carries the larger share. The returned list has one
    weight per element of the very list passed in, `0.0` off the hits, and sums
    to 1. A single hit degenerates to the one-hot of that position, and a point
    inside nothing is a miss rather than a zero vector.

    Areas are computed in float on purpose: `element_bounds` bounds each
    coordinate, not the product, so an area can leave the float range. One that
    overflows to `inf` inverts to `0.0`, which is its true share against any box
    that fits; if every hit overflows the total would be `0.0 / 0.0`, and if an
    area underflows to `0.0` the ratio would be a division by zero. Both cases
    mean the areas cannot be ranked at all, so the fallback is a uniform
    distribution over the hits -- never a zero vector with a hit in it, and a
    lone overflowing hit still comes back as its own one-hot.
    """
    hits = element_hits(elements, x, y)
    if not hits:
        return None
    areas: dict[int, float] = {}
    for position in hits:
        bounds = element_bounds(elements[position])
        if bounds is None:  # not reachable: element_hits only reports usable boxes
            continue
        left, top, right, bottom = bounds
        areas[position] = (float(right) - float(left)) * (float(bottom) - float(top))
    uniform = [1.0 / len(areas) if position in areas else 0.0 for position in range(len(elements))]
    if any(area == 0.0 for area in areas.values()):
        return uniform
    inverse = {
        position: (1.0 / area if math.isfinite(area) else 0.0) for position, area in areas.items()
    }
    total = sum(inverse.values())
    if total == 0.0 or not math.isfinite(total):
        return uniform
    return [inverse.get(position, 0.0) / total for position in range(len(elements))]


def step_operation(action: str, raw: dict | None) -> str | None:
    """The agent operation a mapped Android Control action stands for.

    `terminate` is the null action of an episode's last step, which the agent
    expresses as DONE. Scrolls map by identity (see `STEP_OPERATIONS`), and a
    scroll with a direction the corpus never writes maps to None.
    """
    if action == "terminate":
        return "DONE"
    if action == "swipe":
        direction = raw.get("direction") if isinstance(raw, dict) else None
        return SCROLL_OPERATIONS.get(direction) if isinstance(direction, str) else None
    if action == "system_button":
        action_type = raw.get("action_type") if isinstance(raw, dict) else None
        if action_type == "navigate_back":
            return "BACK"
        return "HOME" if action_type == "navigate_home" else None
    return STEP_OPERATIONS.get(action)


def question_criteria(request: jev.Request, question_id: str) -> dict | None:
    """The criteria map of one question, or None when the screen offers none."""
    question = request.questions.get(question_id)
    return question["criteria"] if question else None


def tap_candidates(request: jev.Request, observation: jev.Observation) -> list[jev.Element]:
    """The TAP candidates in the order the `tap_target` criteria list them."""
    criteria = question_criteria(request, "tap_target")
    if criteria is None:
        return []
    elements = []
    for index in criteria:
        candidate = request.space.tap.get(index)
        element = observation.element_by_id(candidate.action.element_id) if candidate else None
        if element is not None:
            elements.append(element)
    return elements


def scroll_region_candidates(request: jev.Request, observation: jev.Observation) -> list[jev.Element]:
    """The SCROLL regions in the order the `scroll_target` criteria list them."""
    criteria = question_criteria(request, "scroll_target")
    if criteria is None:
        return []
    elements = []
    for index in criteria:
        actions = request.space.scroll.get(index) or {}
        candidate = next(iter(actions.values()), None)
        element = observation.element_by_id(candidate.action.region_id) if candidate else None
        if element is not None:
            elements.append(element)
    return elements


def tap_target_weights(
    request: jev.Request, observation: jev.Observation, raw: dict | None
) -> list[float] | None:
    """The click distribution over the TAP candidates, or None when unusable."""
    if not isinstance(raw, dict):
        return None
    x, y = coordinate(raw.get("x")), coordinate(raw.get("y"))
    if x is None or y is None:
        return None
    elements = tap_candidates(request, observation)
    if not MIN_CANDIDATES <= len(elements) <= MAX_CANDIDATES:
        return None
    return element_target_weights(elements, x, y)


def operation_label(
    operation: str,
    observation: jev.Observation,
    request: jev.Request,
    raw: dict | None,
    arguments: dict,
) -> str:
    """The history label the agent would have recorded for this operation.

    The agent labels an executed decision with `describe_action`, so a history
    entry the model reads looks like the one it would see online. Two details
    cannot be recovered from the corpus and are approximated deliberately: a
    scroll's region is not recorded, so the lowest-indexed scroll candidate
    stands in (the gesture JSON then carries that region id, or omits it when
    the screen offers none), and a tap that missed every candidate falls back to
    the raw tool call.
    """
    if operation == "TAP":
        weights = tap_target_weights(request, observation, raw)
        if weights is not None:
            best = max(range(len(weights)), key=weights.__getitem__)
            chosen = tap_candidates(request, observation)[best]
            return jev.describe_action(
                jev.Action("tap_element", element_id=chosen.id), observation
            )
        return jev.canonical_json(arguments)
    if operation == "TYPE_TEXT":
        text = raw.get("text") if isinstance(raw, dict) else None
        return jev.canonical_json(
            jev.Action(
                "type",
                text=text if isinstance(text, str) else "",
                element_id=observation.phone.input_element_id,
            ).as_dict()
        )
    if operation.startswith("SCROLL_"):
        regions = scroll_region_candidates(request, observation)
        return jev.describe_action(
            jev.Action(
                "scroll",
                region_id=regions[0].id if regions else None,
                direction=operation.removeprefix("SCROLL_").lower(),
            ),
            observation,
        )
    if operation == "OPEN_APP":
        name = raw.get("app_name") if isinstance(raw, dict) else None
        return jev.describe_action(jev.Action("open_app", app_name=name or ""), observation)
    if operation == "WAIT":
        return jev.WAIT_LABEL
    if operation == "DONE":
        # A DONE step is the episode's last one, so this entry is never read
        # back by a later step; it stays honest rather than pretending to be an
        # action the agent performed.
        return jev.canonical_json({"action": "terminate"})
    key = {"BACK": "back", "HOME": "home", "ENTER": "enter"}.get(operation)
    if key is None:
        return jev.canonical_json(arguments)
    return jev.canonical_json(
        jev.Action(
            "key" if operation == "ENTER" else "global",
            name=key,
            element_id=observation.phone.input_element_id if operation == "ENTER" else None,
        ).as_dict()
    )


@dataclasses.dataclass(frozen=True)
class Target:
    """One target row's ground truth: its question family, key, and weights."""

    family: str
    key: str
    weights: list[float]

    @property
    def positions(self) -> list[int]:
        return [position for position, weight in enumerate(self.weights) if weight > 0]


@dataclasses.dataclass(frozen=True)
class ACStep:
    """One parsed step: the agent request plus what its rows need.

    `request` is exactly what the agent would post for this screen and goal, so
    every row of the step shares one `state`. `target` is None when the step's
    action cannot be expressed as a target question; the operation row exists
    either way. `history_entry` is the decision this step contributes to the
    next step's `recentActions`, with `screen_changed` filled in once the next
    screen is known.
    """

    id: str
    group: str
    index: int
    image: Path
    image_sha256: str
    action: str
    operation: str
    arguments: dict
    ac_action: dict | None
    instruction: str | None
    observation: jev.Observation
    request: jev.Request
    target: Target | None
    history_entry: jev.HistoryEntry
    reference: dict


def exclusion(step_id: str, reason: str, detail: str = "") -> dict:
    """One `excluded.jsonl` entry for a step that produced no row."""
    return {"id": step_id, "reason": reason, "detail": detail, "stage": "parse"}


def family_exclusion(step_id: str, family: str, reason: str) -> dict:
    """One `excluded.jsonl` entry for a family the step could not express."""
    return {"id": f"{step_id}:{family}", "reason": reason, "detail": "", "stage": "family"}


TARGET_FAMILIES = {
    "TAP": "tap_target",
    "TYPE_TEXT": "text_value",
    "OPEN_APP": "app_target",
}


def target_family(operation: str) -> str | None:
    """The question family an operation's ground truth belongs to, if any."""
    return TARGET_FAMILIES.get(operation)


def width_reason(criteria: dict) -> str | None:
    """Why a criteria map cannot be a row, or None when it can.

    Dohnuts rows allow 2..128 options while the agent allows 255, so a screen
    whose question is wider loses that row rather than the whole step. The
    operation question never gets here: it has at most thirteen options.
    """
    if len(criteria) < MIN_CANDIDATES:
        return TOO_FEW_CANDIDATES
    if len(criteria) > MAX_CANDIDATES:
        return TOO_MANY_CANDIDATES
    return None


def resolve_target(
    step_id: str,
    operation: str,
    observation: jev.Observation,
    request: jev.Request,
    raw: dict | None,
) -> tuple[Target | None, dict | None]:
    """Resolve a step's target row, or the family-level exclusion that drops it.

    Returns `(target, exclusion-or-None)`. An operation that carries no target
    question -- WAIT, DONE, BACK, HOME, ENTER and the scrolls -- resolves to
    `(None, None)`: there is nothing to ask, which is not a failure.

    Every target family is checked against the row limits first, because the
    corpus can produce questions Dohnuts cannot ask: a long goal offers more
    text spans than a row may carry, and a corpus-wide app inventory is wider
    than 128 names.
    """
    if operation == "TAP":
        weights = tap_target_weights(request, observation, raw)
        if weights is not None:
            return Target("tap_target", "", weights), None
        candidates = len(tap_candidates(request, observation))
        if candidates < MIN_CANDIDATES:
            reason = TOO_FEW_CANDIDATES
        elif candidates > MAX_CANDIDATES:
            reason = TOO_MANY_CANDIDATES
        else:
            reason = NO_TARGET_ELEMENT
        return None, family_exclusion(step_id, "tap_target", reason)
    if operation == "TYPE_TEXT":
        criteria = question_criteria(request, "text_value")
        text = raw.get("text") if isinstance(raw, dict) else None
        key = next((key for key, value in (criteria or {}).items() if value == text), None)
        reason = width_reason(criteria) if criteria is not None else TOO_FEW_CANDIDATES
        if reason is not None:
            return None, family_exclusion(step_id, "text_value", reason)
        if key is None:
            return None, family_exclusion(step_id, "text_value", TEXT_NOT_A_GOAL_SPAN)
        return Target("text_value", key, one_hot(len(criteria), list(criteria).index(key))), None
    if operation == "OPEN_APP":
        criteria = question_criteria(request, "app_target")
        label = raw.get("app_name") if isinstance(raw, dict) else None
        key = next((key for key, value in (criteria or {}).items() if value == label), None)
        reason = width_reason(criteria) if criteria is not None else TOO_FEW_CANDIDATES
        if reason is not None:
            return None, family_exclusion(step_id, "app_target", reason)
        if key is None:
            return None, family_exclusion(step_id, "app_target", APP_NOT_OFFERED)
        return Target("app_target", key, one_hot(len(criteria), list(criteria).index(key))), None
    return None, None


def parse_episode(
    metadata: dict, *, episode_dir: Path, apps: tuple[str, ...] = ()
) -> tuple[list[ACStep], list[dict]]:
    """Parse one episode into agent-shaped steps; never raises.

    Returns the steps that produced a prompt and one exclusion entry per step or
    family that did not. A step is excluded when its structure is wrong, its
    screenshot or accessibility forest cannot be read, its action is unknown,
    the screen offers no such operation, or its request would exceed the payload
    the agent refuses to send. Family-level failures -- a click that hit no
    candidate, a typed value that is not a goal span, an app that is not offered
    -- drop one target row and leave the operation row in place.

    `apps` is the offered inventory, narrowed per goal by
    `mobile_jev_prompt.build_request` exactly as the agent narrows the installed
    apps it reads from the device.

    History covers the steps that were parsed: a step excluded for a missing
    screen or an unknown action never executed in this data and so contributes
    nothing, and one whose operation the screen did not offer is dropped with
    its rows. A step that only lost its target row stays in history, because the
    action did happen.
    """
    if not isinstance(metadata, dict):
        return [], [exclusion("", "unparsable_metadata")]
    episode_id = metadata.get("episode_id")
    goal = metadata.get("goal")
    raw_steps = metadata.get("steps")
    if not is_integer(episode_id) or not isinstance(goal, str) or not isinstance(raw_steps, list):
        return [], [exclusion("", "unparsable_metadata")]
    steps: list[ACStep] = []
    exclusions: list[dict] = []
    history: list[jev.HistoryEntry] = []
    pending: tuple[jev.HistoryEntry, str] | None = None
    for index, raw_step in enumerate(raw_steps):
        step_id = f"android_control_{episode_id}_step{index}"
        if not validate_step(raw_step):
            exclusions.append(
                exclusion(step_id, "unparsable_metadata", step_problem(raw_step) or "")
            )
            continue
        screenshot = read_screenshot(episode_dir, raw_step.get("screenshot"))
        if screenshot is None:
            exclusions.append(exclusion(step_id, "missing_image"))
            continue
        image, image_sha256, size = screenshot
        document = read_a11y(episode_dir, raw_step.get("accessibility_tree"))
        if document is None:
            exclusions.append(exclusion(step_id, "missing_a11y"))
            continue
        observation = summarize_observation(
            document, screen_size=size, package_name=foreground_package(document)
        )
        # The previous decision's `screenChanged` is only knowable once this
        # screen has been read, which is why its entry is finalized here.
        if pending is not None:
            entry, fingerprint = pending
            history.append(
                dataclasses.replace(entry, screen_changed=fingerprint != observation.fingerprint)
            )
            pending = None
        raw_action = raw_step.get("action")
        instruction = raw_step.get("step_instruction")
        instruction = instruction if isinstance(instruction, str) else None
        if raw_action is None:
            action = "terminate"
            arguments: dict = {"action": "terminate"}
            ac_action = None
            operation = "DONE"
            instruction = None
        else:
            mapped = map_action(raw_action)
            if mapped is None:
                exclusions.append(exclusion(step_id, "unknown_action"))
                continue
            action, arguments = mapped
            ac_action = raw_action
            operation = step_operation(action, raw_action)
            if operation is None:
                exclusions.append(exclusion(step_id, "unknown_action"))
                continue
        try:
            request = jev.build_request(goal, observation, history, apps)
        except jev.PayloadTooLargeError:
            exclusions.append(exclusion(step_id, "payload_too_large"))
            continue
        if operation not in request.questions["operation"]["criteria"]:
            exclusions.append(exclusion(step_id, OPERATION_NOT_OFFERED))
            continue
        target, family_skip = resolve_target(step_id, operation, observation, request, raw_action)
        if family_skip is not None:
            exclusions.append(family_skip)
        text = raw_action.get("text") if isinstance(raw_action, dict) else None
        entry = jev.HistoryEntry(
            operation=operation,
            label=operation_label(operation, observation, request, raw_action, arguments),
            text=text if isinstance(text, str) else None,
        )
        pending = (entry, observation.fingerprint)
        steps.append(
            ACStep(
                id=step_id,
                group=f"task:android_control_{episode_id}",
                index=index,
                image=image,
                image_sha256=image_sha256,
                action=action,
                operation=operation,
                arguments=arguments,
                ac_action=ac_action,
                instruction=instruction,
                observation=observation,
                request=request,
                target=target,
                history_entry=entry,
                reference={
                    "thought": "",
                    "action": instruction or "",
                    "tool_call": {"name": "mobile_use", "arguments": arguments},
                    "ac_action": ac_action,
                    "element_positions": target.positions if target is not None else [],
                },
            )
        )
    return steps, exclusions


def rows_for_ac_step(
    step: ACStep, image_path: str, marked: tuple[str, str] | None = None
) -> list[dict]:
    """Every decision row of one parsed step: operation, then its target family.

    `step` must come from `parse_episode` (it guarantees the operation is in the
    criteria and that a resolved target lines up with its question). `image_path`
    is the repository root relative path of the stored copy of the raw
    screenshot, not `step.image`. `marked` is the relative path and sha256 of
    the marked copy and is only read by `tap_target`, whose criteria keys are the
    numbers the marked screenshot draws.

    Rows come back in the order operation, then tap_target, text_value or
    app_target. Every row of a step shares `state` **by reference** (treat it as
    read-only); `reference` is copied per row so the row can name its question.
    """
    question = step.request.questions
    aliases = ["image-bytes:" + step.image_sha256]
    if step.target is not None and step.target.family == "tap_target":
        if marked is None:
            raise ValueError(
                "rows_for_ac_step requires the marked screenshot of a tap_target step"
            )
        aliases.append("image-bytes:" + marked[1])

    def row(name: str, target: list[float], image: str) -> dict:
        return {
            "id": f"{step.id}:{name}",
            "dataset": DATASETS[name],
            "group": step.group,
            "aliases": list(aliases),
            "split": split_for(step.group),
            "state": step.request.state,
            "image": image,
            "question": question[name],
            "target": target,
            "reference": {**step.reference, "question_id": name},
        }

    criteria = list(question["operation"]["criteria"])
    rows = [row("operation", one_hot(len(criteria), criteria.index(step.operation)), image_path)]
    if step.target is not None:
        image = image_path
        if step.target.family == "tap_target":
            # The guard above rules out marked=None for exactly this family.
            image = marked[0] if marked is not None else image_path
        rows.append(row(step.target.family, list(step.target.weights), image))
    return rows


def app_vocabulary(documents: Any) -> list[str]:
    """The offered app inventory, collected from the corpus's `open_app` actions.

    Android Control records the display name of every app it opens but never the
    device's installed apps, so the agent's inventory is approximated by the
    apps the corpus itself opens. The order is `(casefold, name)` so the list is
    stable across runs and machines, and the result feeds `_named_apps` exactly
    like a device's app list would.
    """
    names = set()
    for document in documents:
        if not isinstance(document, dict):
            continue
        for step in document.get("steps") or []:
            action = step.get("action") if isinstance(step, dict) else None
            if not isinstance(action, dict) or action.get("action_type") != "open_app":
                continue
            name = action.get("app_name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return sorted(names, key=lambda name: (name.casefold(), name))
