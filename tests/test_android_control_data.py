"""Android Control conversion: readers, the agent-shaped observation, the rows.

Everything here is synthetic: an episode is written to `tmp_path` with a PIL
screenshot and a hand-written accessibility forest, so the tests stay
independent of the example-data corpus. The prompt shapes themselves are pinned
in `tests/test_mobile_jev_prompt.py`; what is checked here is that the corpus is
read into exactly those shapes and that the ground truth lines up with them.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from dohnuts import mobile_jev_prompt as jev
from dohnuts.android_control_data import (
    DATASETS,
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    NO_TARGET_ELEMENT,
    OPERATION_NOT_OFFERED,
    TOO_FEW_CANDIDATES,
    app_vocabulary,
    clamped_bounds,
    element_bounds,
    element_hits,
    element_target_weights,
    foreground_package,
    iter_nodes,
    map_action,
    metadata_detail,
    parse_episode,
    parse_metadata,
    read_a11y,
    read_screenshot,
    rows_for_ac_step,
    scroll_direction,
    step_operation,
    step_problem,
    summarize_observation,
    validate_step,
)
from dohnuts.gui_data import image_digest, split_for, validate_rows
from dohnuts.recipe import MAX_LENGTH

SCREENSHOT = (100, 200)
SCREEN = SCREENSHOT
GOAL = "Open the Zoho Meet app , view the scheduled meetings ."
PACKAGE = "com.example.app"
RAW_IMAGE = "images/raw.png"
MARKED_IMAGE = "images/marked.png"


def node(**overrides):
    """One accessibility node with the fields the summarizer reads."""
    base = {
        "className": "android.widget.Button",
        "text": "",
        "contentDescription": "",
        "boundsInScreen": {"left": 0, "top": 0, "right": 10, "bottom": 10},
        "isClickable": True,
        "isVisibleToUser": True,
    }
    base.update(overrides)
    return base


def window(*nodes, window_type="TYPE_APPLICATION", package=PACKAGE, bounds=None):
    return {
        "windowType": window_type,
        "layer": 1,
        "boundsInScreen": bounds
        or {"left": 0, "top": 0, "right": SCREEN[0], "bottom": SCREEN[1]},
        "tree": {"nodes": [{"packageName": package, **item} for item in nodes]},
    }


def forest(*nodes, package=PACKAGE):
    return {"windows": [window(*nodes, package=package)]}


def write_png(path: Path, *, size=SCREENSHOT, color="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_a11y(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def ac_step(index, action, instruction, *, directory: Path, nodes=None, color="white"):
    """One metadata step entry, with the files it points at already on disk.

    `nodes=None` writes no accessibility file at all, so a test can prove the
    step reported it instead of summarizing a blank screen.
    """
    stem = f"step_{index:03d}"
    write_png(directory / f"{stem}_screenshot.png", color=color)
    if nodes is not None:
        write_a11y(directory / f"{stem}_a11y.json", forest(*nodes))
    return {
        "step_id": index,
        "screenshot": f"{stem}_screenshot.png",
        "accessibility_tree": f"{stem}_a11y.json",
        "action": action,
        "step_instruction": instruction,
    }


def raw_step(index, action, instruction):
    """One metadata step entry with no files behind it."""
    stem = f"step_{index:03d}"
    return {
        "step_id": index,
        "screenshot": f"{stem}_screenshot.png",
        "accessibility_tree": f"{stem}_a11y.json",
        "action": action,
        "step_instruction": instruction,
    }


def ac_episode(*steps, goal=GOAL, episode_id=7):
    return {"episode_id": episode_id, "goal": goal, "steps": list(steps)}


def parse(record, *, directory, apps=()):
    return parse_episode(record, episode_dir=directory, apps=apps)


def only_step(record, *, directory, apps=()):
    steps, exclusions = parse(record, directory=directory, apps=apps)
    assert exclusions == []
    assert len(steps) == 1
    return steps[0]


def element(id="0", bounds=(0, 0, 10, 10), **overrides):
    """One observation element, exactly as the summarizer would build it."""
    return jev.Element(id=id, bounds=tuple(bounds), **overrides)


def click_nodes():
    """Two clickable nodes; a click at (75, 75) resolves to the second."""
    return [
        node(text="Cancel", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50}),
        node(
            text="JOIN A MEETING",
            contentDescription="Join the next meeting",
            boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100},
        ),
        node(
            text="Decorative",
            isClickable=False,
            boundsInScreen={"left": 0, "top": 150, "right": 100, "bottom": 180},
        ),
    ]


# ---------------------------------------------------------------------------
# Readers: metadata, screenshots, accessibility forests.
# ---------------------------------------------------------------------------


def test_metadata_detail_names_the_first_structural_failure():
    assert metadata_detail({"episode_id": 0, "goal": "g", "steps": []}) == ""
    assert metadata_detail([]) == "document is not an object"
    assert metadata_detail({"goal": "g", "steps": []}) == "key 'episode_id' is not an integer"
    assert metadata_detail({"episode_id": 0, "goal": 1, "steps": []}) == "key 'goal' is not a string"
    assert (
        metadata_detail({"episode_id": 0, "goal": "g", "steps": {}}) == "key 'steps' is not a list"
    )
    assert (
        metadata_detail({"episode_id": 0, "goal": "g", "steps": [{}]})
        == "step 0: key 'step_id' is not an integer"
    )


def test_parse_metadata_accepts_valid_records_and_reports_reasons():
    record = ac_episode(raw_step(0, {"action_type": "wait"}, "wait"))
    assert parse_metadata(record) == (record, None)
    assert parse_metadata({"episode_id": 0, "goal": "g", "steps": []})[1] is None
    assert parse_metadata(None) == (None, "unparsable_metadata")
    assert parse_metadata([1]) == (None, "unparsable_metadata")


def test_step_problem_checks_fields_in_reading_order():
    base = {
        "step_id": 0,
        "screenshot": "a.png",
        "accessibility_tree": "a.json",
        "action": None,
        "step_instruction": None,
    }
    assert validate_step(base)
    assert step_problem({**base, "step_id": "0"}) == "key 'step_id' is not an integer"
    assert (
        step_problem({**base, "screenshot": ""}) == "key 'screenshot' is not a non-empty file name"
    )
    assert step_problem({**base, "action": 1}) == "key 'action' is neither an object nor null"
    assert (
        step_problem({**base, "step_instruction": 2})
        == "key 'step_instruction' is neither a string nor null"
    )
    assert step_problem({k: v for k, v in base.items() if k != "action"}) == "missing key 'action'"


def test_read_screenshot_refuses_paths_that_escape_the_episode(tmp_path):
    episode = tmp_path / "episode"
    write_png(episode / "step_000_screenshot.png")
    write_png(tmp_path / "outside.png")
    assert read_screenshot(episode, "step_000_screenshot.png") is not None
    assert read_screenshot(episode, "../outside.png") is None
    assert read_screenshot(episode, "/etc/hostname") is None
    assert read_screenshot(episode, "") is None
    assert read_screenshot(episode, None) is None
    assert read_screenshot(episode, "missing.png") is None


def test_read_screenshot_reports_size_and_digest(tmp_path):
    episode = tmp_path / "episode"
    path = write_png(episode / "step_000_screenshot.png", size=(64, 32))
    image, sha256, size = read_screenshot(episode, "step_000_screenshot.png")
    assert image == path
    assert size == (64, 32)
    assert sha256 == image_digest(path)


def test_read_screenshot_rejects_a_png_that_cannot_be_decoded(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir(parents=True)
    (episode / "torn.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"not a png")
    assert read_screenshot(episode, "torn.png") is None
    (episode / "text.png").write_text("hello")
    assert read_screenshot(episode, "text.png") is None


def test_read_a11y_requires_a_window_list(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir(parents=True)
    write_a11y(episode / "ok.json", forest(node()))
    write_a11y(episode / "not_a_forest.json", {"windows": {}})
    write_a11y(episode / "bad.json", [1, 2])
    (episode / "torn.json").write_text("{")
    assert read_a11y(episode, "ok.json") is not None
    assert read_a11y(episode, "not_a_forest.json") is None
    assert read_a11y(episode, "bad.json") is None
    assert read_a11y(episode, "torn.json") is None
    assert read_a11y(episode, "../ok.json") is None
    assert read_a11y(episode, "missing.json") is None


def test_iter_nodes_walks_windows_then_nodes_and_skips_junk():
    document = {
        "windows": [
            {"tree": {"nodes": [node(text="a"), "junk", 3]}},
            "not a window",
            {"tree": {"nodes": [node(text="b")]}},
            {"tree": {}},
        ]
    }
    assert [item["text"] for item in iter_nodes(document)] == ["a", "b"]
    assert list(iter_nodes({"windows": "no"})) == []


# ---------------------------------------------------------------------------
# Action mapping and the operation vocabulary.
# ---------------------------------------------------------------------------


def test_map_action_maps_the_corpus_vocabulary():
    assert map_action({"action_type": "click", "x": 5, "y": 6}) == (
        "click",
        {"action": "click", "coordinate": [5, 6]},
    )
    assert map_action({"action_type": "long_press", "x": 5, "y": 6})[0] == "long_press"
    # The finger vocabulary inverts the content direction; the operation mapping
    # below does not (see `step_operation`).
    assert map_action({"action_type": "scroll", "direction": "down"}) == (
        "swipe",
        {"action": "swipe", "direction": "up"},
    )
    assert map_action({"action_type": "input_text", "text": "hi"}) == (
        "type",
        {"action": "type", "text": "hi"},
    )
    assert map_action({"action_type": "wait"}) == ("wait", {"action": "wait"})
    assert map_action({"action_type": "open_app", "app_name": "Clock"}) == (
        "open_app",
        {"action": "open_app", "app_name": "Clock"},
    )
    assert map_action({"action_type": "navigate_back"}) == (
        "system_button",
        {"action": "system_button", "button": "Back"},
    )


def test_map_action_rejects_unknown_or_malformed_actions():
    assert map_action({"action_type": "answer", "text": "x"}) is None
    assert map_action({"action_type": "click", "x": True, "y": 1}) is None
    assert map_action({"action_type": "click", "x": float("inf"), "y": 1}) is None
    assert map_action({"action_type": "scroll", "direction": "sideways"}) is None
    assert map_action({"action_type": "input_text", "text": 3}) is None
    assert map_action("click") is None


def test_step_operation_merges_the_scrolls_and_the_last_step_is_done():
    assert step_operation("click", None) == "TAP"
    assert step_operation("long_press", None) == "TAP"
    # `map_action` names the mapped action "type"; `step_operation` reads that.
    assert step_operation("type", None) == "TYPE_TEXT"
    assert step_operation("wait", None) == "WAIT"
    assert step_operation("open_app", None) == "OPEN_APP"
    assert step_operation("input_text", None) is None
    assert step_operation("system_button", {"action_type": "navigate_back"}) == "BACK"
    assert step_operation("system_button", {"action_type": "navigate_home"}) == "HOME"
    assert step_operation("terminate", None) == "DONE"
    # Every scroll is one SCROLL operation now; the direction is the answer to
    # the separate question, and it keeps AC's own meaning (see
    # `SCROLL_TO_SWIPE_DIRECTION` for the gui-v1 finger vocabulary, which is a
    # different mapping).
    assert step_operation("swipe", {"direction": "down"}) == "SCROLL"
    assert step_operation("swipe", {"direction": "up"}) == "SCROLL"
    assert scroll_direction({"direction": "down"}) == "DOWN"
    assert scroll_direction({"direction": "up"}) == "UP"
    assert scroll_direction({"direction": "left"}) == "LEFT"
    assert scroll_direction({"direction": "right"}) == "RIGHT"
    assert step_operation("swipe", {"direction": "diagonal"}) is None
    assert scroll_direction({"direction": "diagonal"}) is None


# ---------------------------------------------------------------------------
# The observation: filtering, ids, focus, and the foreground package.
# ---------------------------------------------------------------------------


def test_summarize_observation_keeps_what_the_agent_keeps():
    document = forest(
        node(text="Wi-Fi", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 20}),
        node(
            className="android.widget.EditText",
            isEditable=True,
            isFocused=True,
            hintText="Search",
            isClickable=False,
            boundsInScreen={"left": 0, "top": 30, "right": 100, "bottom": 60},
        ),
        node(
            text="List",
            isScrollable=True,
            isClickable=False,
            boundsInScreen={"left": 0, "top": 70, "right": 100, "bottom": 190},
        ),
        node(
            text="decor",
            isClickable=False,
            isVisibleToUser=False,
            boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10},
        ),
        node(
            text="",
            contentDescription="",
            isClickable=False,
            isScrollable=False,
            boundsInScreen={"left": 0, "top": 62, "right": 10, "bottom": 66},
        ),
    )
    observation = summarize_observation(document, screen_size=SCREEN, package_name=PACKAGE)
    assert [item.id for item in observation.elements] == ["0", "1", "2"]
    assert [item.text or item.label for item in observation.elements] == ["Wi-Fi", "", "List"]
    assert observation.elements[1].editable
    assert observation.elements[1].hint == "Search"
    assert observation.phone.is_editable
    assert observation.phone.input_element_id == "1"
    assert observation.phone.package_name == PACKAGE
    assert observation.screen == SCREEN


def test_summarize_observation_keeps_source_indices_of_dropped_nodes():
    """A dropped node still consumes an id: the id is the position, not a rank."""
    document = forest(
        node(
            text="",
            contentDescription="",
            isClickable=False,
            isVisibleToUser=False,
            boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10},
        ),
        node(text="kept", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 20}),
    )
    observation = summarize_observation(document, screen_size=SCREEN, package_name=PACKAGE)
    assert [(item.id, item.text) for item in observation.elements] == [("1", "kept")]


def test_summarize_observation_clamps_bounds_and_drops_empty_boxes():
    document = forest(
        node(
            text="offscreen",
            boundsInScreen={"left": 500, "top": 500, "right": 600, "bottom": 600},
        ),
        node(text="clipped", boundsInScreen={"left": -10, "top": -10, "right": 20, "bottom": 20}),
        node(text="inverted", boundsInScreen={"left": 60, "top": 60, "right": 40, "bottom": 40}),
    )
    observation = summarize_observation(document, screen_size=SCREEN, package_name=PACKAGE)
    assert [item.text for item in observation.elements] == ["clipped"]
    assert observation.elements[0].bounds == (0, 0, 20, 20)


def test_input_element_needs_a_unique_focus_or_a_single_input():
    def editable(focused=False, id="0"):
        return element(id=id, editable=True, focused=focused)

    assert jev.input_element([editable(focused=True, id="1"), editable(id="2")]).id == "1"
    assert jev.input_element([editable(id="2")]).id == "2"
    assert jev.input_element([editable(id="1"), editable(id="2")]) is None
    assert jev.input_element([element(editable=True, enabled=False)]) is None
    assert jev.input_element([editable(focused=True, id="1"), editable(focused=True, id="2")]) is None


def test_foreground_package_prefers_the_largest_application_window():
    document = {
        "windows": [
            window(node(text="status"), window_type="TYPE_SYSTEM", package="com.android.systemui"),
            window(node(text="app"), window_type="TYPE_APPLICATION", package=PACKAGE),
            window(
                node(text="keyboard"),
                window_type="TYPE_INPUT_METHOD",
                package="com.google.android.inputmethod.latin",
                bounds={"left": 0, "top": 120, "right": 100, "bottom": 200},
            ),
        ]
    }
    assert foreground_package(document) == PACKAGE


def test_foreground_package_falls_back_and_never_returns_the_keyboard():
    without_application = {
        "windows": [
            window(
                node(text="keyboard"),
                window_type="TYPE_INPUT_METHOD",
                package="com.google.android.inputmethod.latin",
            ),
            window(node(text="system"), window_type="TYPE_SYSTEM", package="com.android.systemui"),
        ]
    }
    assert foreground_package(without_application) == "com.android.systemui"
    assert foreground_package({"windows": []}) == ""
    assert foreground_package(None) == ""


def test_foreground_package_uses_the_majority_package_of_a_window():
    document = {
        "windows": [
            {
                "windowType": "TYPE_APPLICATION",
                "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 200},
                "tree": {
                    "nodes": [
                        {"packageName": "com.other"},
                        {"packageName": PACKAGE},
                        {"packageName": PACKAGE},
                    ]
                },
            }
        ]
    }
    assert foreground_package(document) == PACKAGE


# ---------------------------------------------------------------------------
# Element geometry and the click distribution.
# ---------------------------------------------------------------------------


def test_clamped_bounds_clamps_to_the_screen_and_refuses_empty_boxes():
    assert clamped_bounds(
        {"boundsInScreen": {"left": -5, "top": -5, "right": 500, "bottom": 500}}, 100, 200
    ) == (0, 0, 100, 200)
    assert (
        clamped_bounds(
            {"boundsInScreen": {"left": 200, "top": 0, "right": 300, "bottom": 10}}, 100, 200
        )
        is None
    )
    assert clamped_bounds({}, 100, 200) is None


def test_element_bounds_rejects_unusable_boxes():
    assert element_bounds(element(bounds=(1, 2, 3, 4))) == (1, 2, 3, 4)
    assert element_bounds(element(bounds=(3, 2, 1, 4))) is None
    assert element_bounds(element(bounds=(1, 2, 1, 4))) is None
    assert element_bounds(element(bounds=(1, 2, 3))) is None
    assert element_bounds(element(bounds=(1, 2, 3, float("nan")))) is None
    assert element_bounds(element(bounds=(1, 2, 3, float("inf")))) is None
    assert element_bounds(element(bounds=(True, 2, 3, 4))) is None
    assert element_bounds({"bounds": (1, 2, 3, 4)}) is None
    assert element_bounds(object()) is None


def test_element_hits_uses_closed_bounds_and_reports_positions():
    elements = [
        element(id="9", bounds=(0, 0, 10, 10)),
        element(id="3", bounds=(10, 10, 20, 20)),
        element(id="4", bounds=(30, 30, 40, 40)),
    ]
    assert element_hits(elements, 10, 10) == [0, 1]
    assert element_hits(elements, 25, 25) == []
    assert element_hits(elements, float("nan"), 0) == []
    assert element_hits("no", 0, 0) == []


def test_element_target_weights_degenerate_to_a_one_hot():
    elements = [element(bounds=(0, 0, 10, 10)), element(bounds=(20, 20, 30, 30))]
    assert element_target_weights(elements, 5, 5) == [1.0, 0.0]


def test_element_target_weights_split_multi_hits_by_inverse_area():
    elements = [
        element(bounds=(0, 0, 100, 100)),  # area 10_000
        element(bounds=(0, 0, 50, 50)),  # area 2_500, four times as specific
    ]
    weights = element_target_weights(elements, 10, 10)
    assert weights is not None
    assert weights[1] == pytest.approx(0.8)
    assert weights[0] == pytest.approx(0.2)
    assert sum(weights) == pytest.approx(1.0)


def test_element_target_weights_report_a_miss_and_zero_off_the_hits():
    elements = [element(bounds=(0, 0, 10, 10)), element(bounds=(50, 50, 60, 60))]
    assert element_target_weights(elements, 55, 55) == [0.0, 1.0]
    assert element_target_weights(elements, 25, 25) is None


def test_element_target_weights_survive_areas_that_leave_the_float_range():
    # The coordinates are finite; it is the product that overflows to inf.
    huge = element(bounds=(0.0, 0.0, 1e200, 1e200))
    tiny = element(bounds=(0.0, 0.0, 1.0, 1.0))
    # An overflowing hit carries no mass against a box that fits: its true share.
    assert element_target_weights([huge, tiny], 0, 0) == pytest.approx([0.0, 1.0])
    # ...but when nothing can be ranked the fallback is uniform, never a zero
    # vector with a hit in it, and a lone hit is still its own one-hot.
    assert element_target_weights([huge, element(bounds=(0.0, 0.0, 2e200, 2e200))], 0, 0) == (
        pytest.approx([0.5, 0.5])
    )
    assert element_target_weights([huge], 0, 0) == [1.0]


def test_element_target_weights_fall_back_to_uniform_when_an_area_underflows():
    underflow = element(bounds=(0.0, 0.0, 1e-200, 1e-200))
    other = element(bounds=(0.0, 0.0, 10.0, 10.0))
    assert element_target_weights([underflow, other], 0, 0) == pytest.approx([0.5, 0.5])


def test_element_target_weights_ignore_unusable_boxes():
    elements = [
        element(bounds=(0, 0, 10, 10)),
        element(bounds=(float("inf"), 0, 10, 10)),
    ]
    assert element_target_weights(elements, 5, 5) == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Parsing a step: the operation, the target, and the history entry.
# ---------------------------------------------------------------------------


def test_click_step_produces_the_agent_operation_and_a_tap_target(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap JOIN A MEETING",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    step = only_step(record, directory=directory)
    assert step.id == "android_control_7_step0"
    assert step.group == "task:android_control_7"
    assert step.action == "click"
    assert step.operation == "TAP"
    assert step.arguments == {"action": "click", "coordinate": [75, 75]}
    assert step.ac_action == {"action_type": "click", "x": 75, "y": 75}
    assert step.instruction == "Tap JOIN A MEETING"
    assert step.image == directory / "step_000_screenshot.png"
    assert step.image_sha256 == image_digest(directory / "step_000_screenshot.png")
    assert step.target is not None
    assert step.target.family == "tap_target"
    assert step.target.weights == [0.0, 1.0]
    assert step.target.positions == [1]
    assert step.reference["element_positions"] == [1]
    assert step.reference["tool_call"] == {
        "name": "mobile_use",
        "arguments": {"action": "click", "coordinate": [75, 75]},
    }
    assert step.history_entry.operation == "TAP"
    assert step.history_entry.label == "Tap JOIN A MEETING / Join the next meeting."


def test_state_is_the_agent_state_with_the_goal_and_the_screen(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap JOIN A MEETING",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    step = only_step(record, directory=directory, apps=("Clock", "Notes"))
    state = step.request.state
    assert list(state) == [
        "goal",
        "app",
        "isEditable",
        "textSource",
        "textEntryAvailableAfterFocus",
        "visibleText",
        "elements",
        "availableApps",
        "recentActions",
    ]
    assert state["goal"] == GOAL
    assert state["app"] == PACKAGE
    assert state["isEditable"] is False
    assert state["textSource"] == "goal"
    assert state["visibleText"] == [
        "Cancel",
        "JOIN A MEETING",
        "Join the next meeting",
        "Decorative",
    ]
    assert state["elements"] == [
        {
            "index": "1",
            "label": "Cancel",
            "editable": False,
            "scrollable": False,
            "operations": ["TAP"],
        },
        {
            "index": "2",
            "label": "JOIN A MEETING / Join the next meeting",
            "editable": False,
            "scrollable": False,
            "operations": ["TAP"],
        },
    ]
    assert state["availableApps"] == [
        {"index": "1", "label": "Clock"},
        {"index": "2", "label": "Notes"},
    ]
    assert list(step.request.questions) == ["operation", "app_target", "tap_target"]
    assert list(step.request.questions["operation"]["criteria"]) == [
        "OPEN_APP",
        "TAP",
        "BACK",
        "HOME",
        "WAIT",
        "DONE",
        "BLOCKED",
    ]
    assert step.request.questions["tap_target"]["criteria"] == {
        "1": "[1] Cancel",
        "2": "[2] JOIN A MEETING / Join the next meeting",
    }


def test_scroll_step_produces_a_direction_row(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "scroll", "direction": "down"},
            "Swipe up for details",
            directory=directory,
            nodes=[
                node(
                    text="List",
                    isScrollable=True,
                    isClickable=False,
                    boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 190},
                ),
            ],
        )
    )
    step = only_step(record, directory=directory)
    assert step.action == "swipe"
    assert step.operation == "SCROLL"
    assert step.arguments == {"action": "swipe", "direction": "up"}
    assert step.target is not None
    assert step.target.family == "scroll_direct"
    assert step.target.key == "DOWN"
    assert step.target.weights == [1.0, 0.0, 0.0, 0.0]
    criteria = step.request.questions["operation"]["criteria"]
    assert "SCROLL" in criteria
    assert not [name for name in criteria if name.startswith("SCROLL_")]
    # The agent's region question is still built; the conversion just never
    # writes a row for it.
    assert "scroll_target" in step.request.questions
    rows = rows_for_ac_step(step, RAW_IMAGE)
    assert [row["dataset"] for row in rows] == ["jev_operation", "jev_scroll_direct"]
    operation, direction = rows
    assert direction["question"]["criteria"] == {
        "DOWN": "Scroll down to reveal more content in that direction.",
        "UP": "Scroll up to reveal more content in that direction.",
        "LEFT": "Scroll left to reveal more content in that direction.",
        "RIGHT": "Scroll right to reveal more content in that direction.",
    }
    assert direction["target"] == [1.0, 0.0, 0.0, 0.0]
    assert operation["target"][list(operation["question"]["criteria"]).index("SCROLL")] == 1.0
    assert direction["reference"]["element_positions"] == []


def test_typed_step_keeps_type_text_as_an_operation_without_a_target_row(tmp_path):
    directory = tmp_path / "episode"
    nodes = [
        node(
            className="android.widget.EditText",
            isEditable=True,
            isFocused=True,
            isClickable=False,
            hintText="Search",
            boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 40},
        )
    ]
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "input_text", "text": "Zoho Meet"},
            "type",
            directory=directory,
            nodes=nodes,
        )
    )
    step = only_step(record, directory=directory)
    assert step.operation == "TYPE_TEXT"
    assert step.target is None
    # The agent's question is still built and the operation is still offered;
    # only the row is not written, whatever the typed value was.
    assert "TYPE_TEXT" in step.request.questions["operation"]["criteria"]
    assert "Zoho Meet" in step.request.questions["text_value"]["criteria"].values()
    assert step.history_entry.text == "Zoho Meet"
    assert step.history_entry.label == jev.canonical_json(
        {"element_id": "0", "text": "Zoho Meet", "type": "type"}
    )
    assert [row["dataset"] for row in rows_for_ac_step(step, RAW_IMAGE)] == ["jev_operation"]

    outside = ac_episode(
        ac_step(
            0,
            {"action_type": "input_text", "text": "unrelated"},
            "type",
            directory=directory,
            nodes=nodes,
        )
    )
    steps, exclusions = parse(outside, directory=directory)
    assert [item.operation for item in steps] == ["TYPE_TEXT"]
    assert steps[0].target is None
    assert exclusions == []


def test_open_app_step_targets_the_offered_inventory(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "open_app", "app_name": "Zoho Meeting"},
            "Open the app",
            directory=directory,
            nodes=[node(text="home", boundsInScreen={"left": 0, "top": 0, "right": 20, "bottom": 20})],
        )
    )
    step = only_step(record, directory=directory, apps=("Clock", "Zoho Meeting"))
    assert step.operation == "OPEN_APP"
    assert step.target is not None
    assert step.target.family == "app_target"
    assert step.request.questions["app_target"]["criteria"][step.target.key] == "Zoho Meeting"
    assert step.history_entry.label == "Open Zoho Meeting"


def test_open_app_with_a_singleton_inventory_loses_its_row(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "open_app", "app_name": "Zoho Meeting"},
            "Open the app",
            directory=directory,
            nodes=[node(text="home")],
        )
    )
    steps, exclusions = parse(record, directory=directory, apps=("Zoho Meeting",))
    # The goal does not name the app, so the whole one-name inventory is offered;
    # a one-option question cannot be a row.
    assert steps[0].target is None
    assert exclusions[0]["reason"] == TOO_FEW_CANDIDATES
    assert exclusions[0]["id"].endswith(":app_target")
    # The state still carries the inventory, exactly as the agent would send it.
    assert steps[0].request.state["availableApps"] == [{"index": "1", "label": "Zoho Meeting"}]


def test_goal_named_apps_narrow_the_inventory(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "wait", directory=directory, nodes=[node(text="x")]),
        goal="Open Clock and then Notes",
    )
    step = only_step(record, directory=directory, apps=("Clock", "Notes", "Camera"))
    assert step.request.state["availableApps"] == [
        {"index": "1", "label": "Clock"},
        {"index": "2", "label": "Notes"},
    ]


def test_wait_and_terminal_steps_use_the_agent_operations(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "wait", directory=directory, nodes=[node(text="a")]),
        ac_step(1, None, None, directory=directory, nodes=[node(text="b")]),
    )
    steps, exclusions = parse(record, directory=directory)
    assert exclusions == []
    assert [step.operation for step in steps] == ["WAIT", "DONE"]
    assert steps[0].history_entry.label == jev.WAIT_LABEL
    assert steps[0].target is None
    assert steps[1].target is None
    assert steps[1].instruction is None
    assert steps[1].arguments == {"action": "terminate"}


def test_history_finalizes_screen_changed_from_the_next_screen(tmp_path):
    directory = tmp_path / "episode"
    same = [node(text="a", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50})]
    other = [node(text="b", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50})]
    changed = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 10, "y": 10}, "tap", directory=directory, nodes=same
        ),
        ac_step(
            1, {"action_type": "click", "x": 10, "y": 10}, "tap", directory=directory, nodes=other
        ),
    )
    steps, _ = parse(changed, directory=directory)
    assert steps[1].request.state["recentActions"][0]["screenChanged"] is True

    unchanged = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 10, "y": 10}, "tap", directory=directory, nodes=same
        ),
        ac_step(
            1, {"action_type": "click", "x": 10, "y": 10}, "tap", directory=directory, nodes=same
        ),
    )
    steps, _ = parse(unchanged, directory=directory)
    assert steps[1].request.state["recentActions"][0]["screenChanged"] is False


def test_recent_actions_keep_only_the_last_eight(tmp_path):
    directory = tmp_path / "episode"
    steps = [
        ac_step(index, {"action_type": "wait"}, "wait", directory=directory, nodes=[node(text="a")])
        for index in range(11)
    ]
    parsed, _ = parse(ac_episode(*steps), directory=directory)
    recent = parsed[-1].request.state["recentActions"]
    assert len(recent) == 8
    assert all(entry["operation"] == "WAIT" for entry in recent)


def test_a_step_is_excluded_when_the_screen_does_not_offer_the_operation(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "tap",
            directory=directory,
            nodes=[
                node(
                    text="nothing to tap",
                    isClickable=False,
                    boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50},
                )
            ],
        )
    )
    steps, exclusions = parse(record, directory=directory)
    assert steps == []
    assert exclusions == [
        {
            "id": "android_control_7_step0",
            "reason": OPERATION_NOT_OFFERED,
            "detail": "",
            "stage": "parse",
        }
    ]


def test_step_exclusions_cover_the_readers_and_the_payload_cap(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "wait", directory=directory, nodes=None),
        ac_step(
            1,
            {"action_type": "answer", "text": "x"},
            "answer",
            directory=directory,
            nodes=[node(text="a")],
        ),
        ac_step(2, {"action_type": "wait"}, "wait", directory=directory,
                nodes=[node(text="t" * 200_000)]),
    )
    (directory / "step_001_screenshot.png").unlink()
    (directory / "step_001_a11y.json").unlink()
    steps, exclusions = parse(record, directory=directory)
    assert steps == []
    assert [entry["reason"] for entry in exclusions] == [
        "missing_a11y",
        "missing_image",
        "payload_too_large",
    ]
    assert [entry["id"] for entry in exclusions] == [
        "android_control_7_step0",
        "android_control_7_step1",
        "android_control_7_step2",
    ]


def test_tap_family_drops_report_why(tmp_path):
    directory = tmp_path / "episode"
    # A single candidate: the question exists but cannot be a row.
    single = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 5, "y": 5},
            "tap",
            directory=directory,
            nodes=[
                node(
                    text="only",
                    boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10},
                )
            ],
        )
    )
    steps, exclusions = parse(single, directory=directory)
    assert steps[0].target is None
    assert exclusions == [
        {
            "id": "android_control_7_step0:tap_target",
            "reason": TOO_FEW_CANDIDATES,
            "detail": "",
            "stage": "family",
        }
    ]

    miss = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 500, "y": 500},
            "tap",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    _, exclusions = parse(miss, directory=directory)
    assert exclusions[0]["reason"] == NO_TARGET_ELEMENT

    crowded = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 5, "y": 5},
            "tap",
            directory=directory,
            nodes=[
                node(
                    text=f"n{index}",
                    boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10},
                )
                for index in range(MAX_CANDIDATES + 1)
            ],
        )
    )
    _, exclusions = parse(crowded, directory=directory)
    # MAX_CANDIDATES is the agent's own option limit now, so a screen past it is
    # refused while the question is built, before any family is resolved.
    assert exclusions[0]["reason"] == "payload_too_large"
    assert exclusions[0]["stage"] == "parse"


def test_parse_episode_reports_unreadable_metadata():
    steps, exclusions = parse_episode(
        {"episode_id": 7, "goal": "g", "steps": "no"}, episode_dir=Path("/nonexistent")
    )
    assert steps == []
    assert exclusions == [{"id": "", "reason": "unparsable_metadata", "detail": "", "stage": "parse"}]


def test_parse_episode_never_raises_on_junk():
    for record in (None, [], 3, {"episode_id": "7"}, {"episode_id": 7, "goal": "g"}):
        steps, exclusions = parse_episode(record, episode_dir=Path("/nonexistent"))
        assert steps == []
        assert exclusions[0]["reason"] == "unparsable_metadata"


# ---------------------------------------------------------------------------
# Rows.
# ---------------------------------------------------------------------------


def test_rows_share_the_state_and_copy_the_reference(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 75, "y": 75}, "tap", directory=directory,
            nodes=click_nodes()
        )
    )
    step = only_step(record, directory=directory)
    marked = (MARKED_IMAGE, image_digest(write_png(tmp_path / MARKED_IMAGE)))
    write_png(tmp_path / RAW_IMAGE)
    rows = rows_for_ac_step(step, RAW_IMAGE, marked)
    assert [row["id"] for row in rows] == [
        "android_control_7_step0:operation",
        "android_control_7_step0:tap_target",
    ]
    assert [row["dataset"] for row in rows] == ["jev_operation", "jev_tap_target"]
    assert {row["group"] for row in rows} == {"task:android_control_7"}
    assert {row["split"] for row in rows} == {split_for("task:android_control_7")}
    operation, tap = rows
    assert operation["state"] is tap["state"]
    assert operation["reference"] is not tap["reference"]
    assert operation["reference"]["question_id"] == "operation"
    assert tap["reference"]["question_id"] == "tap_target"
    assert operation["image"] == RAW_IMAGE
    assert tap["image"] == MARKED_IMAGE
    # Both copies are aliased by every row of the step, so isolation keeps the
    # raw frame and its mark in the same split.
    assert operation["aliases"] == [
        "image-bytes:" + step.image_sha256,
        "image-bytes:" + marked[1],
    ]
    assert tap["aliases"] == operation["aliases"]
    assert operation["question"] is step.request.questions["operation"]
    assert sum(operation["target"]) == pytest.approx(1.0)
    assert sum(tap["target"]) == pytest.approx(1.0)
    assert len(tap["target"]) == len(tap["question"]["criteria"])
    validate_rows(rows, root=tmp_path)
    assert DATASETS["tap_target"] == "jev_tap_target"
    assert MIN_CANDIDATES == 2


def test_rows_for_a_tap_target_need_the_marked_screenshot(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 75, "y": 75}, "tap", directory=directory,
            nodes=click_nodes()
        )
    )
    step = only_step(record, directory=directory)
    with pytest.raises(ValueError, match="requires the marked screenshot"):
        rows_for_ac_step(step, RAW_IMAGE)


def test_tap_target_rows_carry_a_distribution_over_the_criteria(tmp_path):
    directory = tmp_path / "episode"
    nested = [
        node(text="row", boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 100}),
        node(text="cell", boundsInScreen={"left": 40, "top": 40, "right": 60, "bottom": 60}),
    ]
    record = ac_episode(
        ac_step(
            0, {"action_type": "long_press", "x": 50, "y": 50}, "press", directory=directory,
            nodes=nested
        )
    )
    step = only_step(record, directory=directory)
    marked = (MARKED_IMAGE, image_digest(write_png(tmp_path / MARKED_IMAGE)))
    write_png(tmp_path / RAW_IMAGE)
    rows = rows_for_ac_step(step, RAW_IMAGE, marked)
    _, tap = rows
    assert sum(1 for weight in tap["target"] if weight > 0) == 2
    assert tap["target"][1] > tap["target"][0]  # the smaller box is the better target
    assert step.reference["element_positions"] == [0, 1]


# ---------------------------------------------------------------------------
# The app inventory.
# ---------------------------------------------------------------------------


def test_app_vocabulary_collects_open_app_names_in_a_stable_order():
    documents = [
        {"steps": [{"action": {"action_type": "open_app", "app_name": "clock"}}]},
        {"steps": [{"action": {"action_type": "open_app", "app_name": "Clock"}}]},
        {"steps": [{"action": {"action_type": "open_app", "app_name": "  Notes  "}}]},
        {"steps": [{"action": {"action_type": "click", "x": 1, "y": 1}}]},
        {"steps": [{"action": {"action_type": "open_app", "app_name": ""}}]},
        {"steps": ["junk", {"action": None}]},
        None,
        "junk",
    ]
    assert app_vocabulary(documents) == ["Clock", "clock", "Notes"]


# ---------------------------------------------------------------------------
# The CLI: discovery, storage, isolation, and the manifest.
# ---------------------------------------------------------------------------


def load_script():
    name = "prepare_android_control_data"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts/prepare_android_control_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    # A forked worker unpickles the job function by the module it is defined in,
    # so the dynamically loaded script has to be registered under its own name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script()
ROOT = prepare.repository_root()


def write_corpus(source: Path, episodes=2):
    """A tiny corpus: a click step, an open_app step and a terminal step."""
    for episode in range(episodes):
        directory = source / str(episode)
        steps = [
            ac_step(
                0,
                {"action_type": "click", "x": 75, "y": 75},
                "Tap JOIN A MEETING",
                directory=directory,
                nodes=click_nodes(),
                color=(episode + 1, 0, 0),
            ),
            ac_step(
                1,
                {"action_type": "open_app", "app_name": "Zoho Meeting"},
                "Open the app",
                directory=directory,
                nodes=click_nodes(),
                color=(0, episode + 1, 0),
            ),
            ac_step(
                2,
                None,
                None,
                directory=directory,
                nodes=click_nodes(),
                color=(0, 0, episode + 3),
            ),
        ]
        (directory / f"metadata_{episode}.json").write_text(
            json.dumps(ac_episode(*steps, episode_id=episode))
        )
    return source


def run_conversion(tmp_path, workers=1):
    source = write_corpus(tmp_path / f"corpus{workers}")
    output = tmp_path / f"out{workers}"
    manifest = prepare.convert(source, output, workers=workers)
    return output, manifest


def test_conversion_writes_splits_a_manifest_and_exclusions(tmp_path):
    output, manifest = run_conversion(tmp_path)
    assert (output / "manifest.json").is_file()
    assert (output / "excluded.jsonl").is_file()
    for split in ("train", "dev", "calibration", "test"):
        assert (output / f"{split}.jsonl").is_file()
    assert manifest["schema_version"] == 2
    assert manifest["source"]["episodes"] == 2
    assert manifest["source"]["metadata_files_hashed"] == 2
    assert manifest["app_inventory"]["apps"] == 1
    assert manifest["app_inventory"]["offered_cap"] == jev.MAX_APPS
    assert {entry["dataset"] for entry in manifest["counts"]} == {
        "jev_operation",
        "jev_tap_target",
    }
    assert manifest["vocabularies"]["rules"] == jev.RULES
    assert manifest["vocabularies"]["limits"]["max_candidates_per_row"] == MAX_CANDIDATES
    assert manifest["family_coverage"]["operation"]["rows"] == 6
    assert manifest["family_coverage"]["tap_target"] == {
        "basis": "rows are pre-isolation; steps_asking counts parsed steps",
        "dataset": "jev_tap_target",
        "steps_asking": 2,
        "rows": 2,
        "rate": 1.0,
        "dropped": {},
    }
    assert manifest["family_coverage"]["app_target"] == {
        "basis": "rows are pre-isolation; steps_asking counts parsed steps",
        "dataset": "jev_app_target",
        "steps_asking": 2,
        "rows": 0,
        "rate": 0.0,
        "dropped": {TOO_FEW_CANDIDATES: 2},
    }
    rows = [
        json.loads(line)
        for split in ("train", "dev", "calibration", "test")
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]
    validate_rows(rows, root=ROOT)
    assert {row["dataset"] for row in rows} == {"jev_operation", "jev_tap_target"}
    for row in rows:
        if row["dataset"] == "jev_operation":
            assert list(row["question"]["instructions"]) == ["goal", "rules"]
            assert row["reference"]["question_id"] == "operation"


def read_rows(output):
    return [
        json.loads(line)
        for split in ("train", "dev", "calibration", "test")
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]


def test_conversion_is_identical_across_workers(tmp_path):
    """Same input, same workers or not: same rows, in the same order.

    The two runs write into different directories, and a row names its stored
    image relative to the repository root, so the paths are normalized to their
    file names before comparing; everything else has to match byte for byte.
    """
    serial_out, serial = run_conversion(tmp_path, workers=1)
    parallel_out, parallel = run_conversion(tmp_path, workers=2)

    def normalize(rows):
        return [{**row, "image": Path(row["image"]).name} for row in rows]

    assert normalize(read_rows(serial_out)) == normalize(read_rows(parallel_out))
    assert serial["counts"] == parallel["counts"]
    assert serial["exclusions"] == parallel["exclusions"]
    assert serial["family_coverage"] == parallel["family_coverage"]
    assert serial["family_stats"] == parallel["family_stats"]
    assert serial["images"] == parallel["images"]
    assert serial["operation_classes"] == parallel["operation_classes"]


def test_conversion_stores_content_addressed_images(tmp_path):
    output, manifest = run_conversion(tmp_path)
    assert len(manifest["images"]) == len(set(manifest["images"])) == 8
    for name in manifest["images"]:
        path = output / "images" / name
        assert path.is_file()
        assert image_digest(path) == name.removesuffix(".png")


def test_exclusions_of_a_broken_episode_are_written(tmp_path):
    source = write_corpus(tmp_path / "corpus")
    (source / "1" / "metadata_1.json").write_text("{not json")
    output = tmp_path / "out"
    manifest = prepare.convert(source, output, workers=1)
    assert manifest["source"]["metadata_files_hashed"] == 2
    assert manifest["exclusions"]["parse:unparsable_metadata"] == 1
    lines = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    assert any(entry["reason"] == "unparsable_metadata" for entry in lines)


class StubTokenizer:
    """One token per whitespace-separated piece, which is all the gate needs."""

    def __call__(self, text, truncation=False):
        del truncation
        return {"input_ids": list(range(len(text.split())))}


class StubImageProcessor:
    patch_size = 16
    merge_size = 2


class StubProcessor:
    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.image_processor = StubImageProcessor()


def test_token_gate_excludes_an_over_budget_step(tmp_path):
    source = write_corpus(tmp_path / "corpus")
    metadata = json.loads((source / "1" / "metadata_1.json").read_text())
    # Long enough that the prompt's four copies of the goal pass the budget,
    # short enough that the request stays under the agent's payload limit.
    metadata["goal"] = " ".join(["word"] * (MAX_LENGTH // 2))
    (source / "1" / "metadata_1.json").write_text(json.dumps(metadata))
    output = tmp_path / "out"
    manifest = prepare.convert(source, output, processor=StubProcessor(), workers=1)
    # Every step of the long-goal episode is over budget; the other episode is
    # untouched, and nothing of the excluded steps was written.
    assert manifest["exclusions"]["parse:token_budget"] == 3
    assert sum(entry["n"] for entry in manifest["counts"]) == 4
    entries = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    over = [entry for entry in entries if entry["reason"] == "token_budget"]
    assert {entry["id"] for entry in over} == {
        "android_control_1_step0",
        "android_control_1_step1",
        "android_control_1_step2",
    }
    assert all(int(entry["detail"]) > MAX_LENGTH for entry in over)


def test_workers_argument_is_validated(tmp_path):
    with pytest.raises(SystemExit):
        prepare.main(
            ["--input", str(tmp_path), "--output", str(tmp_path / "out"), "--workers", "0"]
        )
