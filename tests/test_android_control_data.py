"""Android Control step parsing, action mapping, element extraction, and row rules."""

import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from dohnuts.android_control_data import (
    AC_ACTIONS,
    DATASETS,
    ELEMENT_INSTRUCTION,
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    SCROLL_TO_SWIPE_DIRECTION,
    element_bounds,
    element_description,
    element_hits,
    element_target_weights,
    extract_elements,
    map_action,
    metadata_detail,
    parse_metadata,
    parse_step,
    resolve_element_target,
    rows_for_ac_step,
    validate_element,
)
from dohnuts.gui_data import (
    ACTIONS,
    BUTTONS,
    COMPLETE_CRITERIA,
    INSTRUCTIONS,
    SPLIT_SEED,
    SWIPE_DIRECTIONS,
    image_digest,
    split_for,
    validate_rows,
)
from dohnuts.recipe import MAX_LENGTH

SCREEN = (1080, 2400)


def node(**overrides):
    """One accessibility node with the fields the extractor reads."""
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


def forest(*nodes):
    return {"windows": [{"windowType": "TYPE_APPLICATION", "tree": {"nodes": list(nodes)}}]}


def element(index, bounds, text="", content_description=""):
    """One extracted element, shaped exactly like `extract_elements` output."""
    return {
        "index": index,
        "text": text,
        "content_description": content_description,
        "bounds": list(bounds),
    }


def metadata(**overrides):
    record = {
        "episode_id": 0,
        "goal": "Open the Zoho Meet app , view the scheduled meetings .",
        "steps": [
            {
                "step_id": 0,
                "screenshot": "step_000_screenshot.png",
                "accessibility_tree": "step_000_a11y.json",
                "action": {"action_type": "open_app", "app_name": "Zoho Meeting"},
                "step_instruction": "Open the Zoho Meet app",
            },
            {
                "step_id": 1,
                "screenshot": "step_001_screenshot.png",
                "accessibility_tree": "step_001_a11y.json",
                "action": None,
                "step_instruction": None,
            },
        ],
    }
    record.update(overrides)
    return record


def step(**overrides):
    entry = {
        "step_id": 0,
        "screenshot": "step_000_screenshot.png",
        "accessibility_tree": "step_000_a11y.json",
        "action": {"action_type": "wait"},
        "step_instruction": "Wait for the app to load",
    }
    entry.update(overrides)
    return entry


def test_ac_actions_appends_open_app_to_the_frozen_vocabulary():
    assert len(AC_ACTIONS) == 9
    assert list(AC_ACTIONS) == [*ACTIONS, "open_app"]
    assert [AC_ACTIONS[name] for name in ACTIONS] == list(ACTIONS.values())
    assert AC_ACTIONS["open_app"]


def test_map_action_maps_click_and_long_press():
    assert map_action({"action_type": "click", "x": 540, "y": 390}) == (
        "click",
        {"action": "click", "coordinate": [540, 390]},
    )
    assert map_action({"action_type": "long_press", "x": 0, "y": 2400}) == (
        "long_press",
        {"action": "long_press", "coordinate": [0, 2400]},
    )


def test_map_action_inverts_the_scroll_direction():
    # Android Control records the finger, gui-v1 records the content: scrolling
    # down reveals content below, so the finger moves up.
    assert map_action({"action_type": "scroll", "direction": "down"}) == (
        "swipe",
        {"action": "swipe", "direction": "up"},
    )
    assert map_action({"action_type": "scroll", "direction": "up"})[1]["direction"] == "down"
    assert map_action({"action_type": "scroll", "direction": "left"})[1]["direction"] == "right"
    assert map_action({"action_type": "scroll", "direction": "right"})[1]["direction"] == "left"
    assert SCROLL_TO_SWIPE_DIRECTION == {
        "down": "up",
        "up": "down",
        "left": "right",
        "right": "left",
    }


def test_map_action_maps_text_wait_and_system_buttons():
    assert map_action({"action_type": "input_text", "text": "hello"}) == (
        "type",
        {"action": "type", "text": "hello"},
    )
    assert map_action({"action_type": "wait"}) == ("wait", {"action": "wait"})
    assert map_action({"action_type": "navigate_back"}) == (
        "system_button",
        {"action": "system_button", "button": "Back"},
    )
    assert map_action({"action_type": "navigate_home"}) == (
        "system_button",
        {"action": "system_button", "button": "Home"},
    )


def test_map_action_maps_open_app():
    assert map_action({"action_type": "open_app", "app_name": "Zoho Meeting"}) == (
        "open_app",
        {"action": "open_app", "app_name": "Zoho Meeting"},
    )


def test_map_action_ignores_keys_the_action_does_not_use():
    assert map_action({"action_type": "wait", "x": "junk", "time": 1}) == (
        "wait",
        {"action": "wait"},
    )
    assert map_action({"action_type": "click", "x": 5, "y": 6, "extra": 1}) == (
        "click",
        {"action": "click", "coordinate": [5, 6]},
    )
    assert map_action({"action_type": "open_app", "app_name": "Maps", "x": None})[1] == {
        "action": "open_app",
        "app_name": "Maps",
    }


def test_map_action_keeps_pixel_values_exact():
    assert map_action({"action_type": "click", "x": 540.5, "y": 390})[1]["coordinate"] == [
        540.5,
        390,
    ]


def test_map_action_rejects_unknown_types():
    assert map_action({"action_type": "double_click"}) is None
    assert map_action({"action_type": "null"}) is None
    assert map_action({"action_type": "open_app "}) is None
    assert map_action({"action_type": 7}) is None
    assert map_action({"action_type": ["click"], "x": 1, "y": 2}) is None
    assert map_action({}) is None
    assert map_action(None) is None
    assert map_action("click") is None
    assert map_action(["click"]) is None


def test_map_action_rejects_malformed_fields():
    malformed = [
        {"action_type": "click", "x": "540", "y": 390},
        {"action_type": "click", "x": 540},
        {"action_type": "click", "y": 390},
        {"action_type": "click", "x": None, "y": None},
        {"action_type": "click", "x": True, "y": 390},
        {"action_type": "click", "x": [540], "y": 390},
        {"action_type": "click", "x": float("nan"), "y": 390},
        {"action_type": "long_press", "x": 1, "y": "2"},
        {"action_type": "scroll"},
        {"action_type": "scroll", "direction": "forward"},
        {"action_type": "scroll", "direction": 3},
        {"action_type": "scroll", "direction": ["down"]},
        {"action_type": "input_text"},
        {"action_type": "input_text", "text": 42},
        {"action_type": "open_app"},
        {"action_type": "open_app", "app_name": None},
        {"action_type": "open_app", "app_name": {"name": "Maps"}},
    ]
    for raw in malformed:
        assert map_action(raw) is None, raw


def test_extract_elements_keeps_clickable_visible_nodes_in_order():
    elements = extract_elements(
        forest(
            node(text="First"),
            node(text="Second", contentDescription="Second tab"),
        ),
        screen_size=SCREEN,
    )
    assert [entry["text"] for entry in elements] == ["First", "Second"]
    assert elements[1]["content_description"] == "Second tab"
    assert [entry["index"] for entry in elements] == [0, 1]
    assert elements[0]["bounds"] == [0, 0, 10, 10]


def test_extract_elements_numbers_only_the_kept_nodes():
    # Non-clickable and invalid nodes must not consume an index: the kept nodes
    # are numbered 0..N-1 over the filtered list.
    elements = extract_elements(
        forest(
            node(text="kept"),
            node(text="not clickable", isClickable=False),
            node(text="invisible", isVisibleToUser=False),
            node(text="degenerate", boundsInScreen={"left": 5, "top": 5, "right": 5, "bottom": 6}),
            node(text="also kept"),
        ),
        screen_size=SCREEN,
    )
    assert [(entry["index"], entry["text"]) for entry in elements] == [
        (0, "kept"),
        (1, "also kept"),
    ]


def test_extract_elements_defaults_missing_booleans_to_false():
    # proto3 JSON omits defaults, so an absent isClickable/isVisibleToUser is false.
    elements = extract_elements(
        forest(
            {"boundsInScreen": {"left": 0, "top": 0, "right": 10, "bottom": 10}},
            node(isVisibleToUser="true"),
            node(isClickable="true"),
            node(isClickable=None, isVisibleToUser=None),
            node(text="kept"),
        ),
        screen_size=SCREEN,
    )
    assert [entry["text"] for entry in elements] == ["kept"]


def test_extract_elements_rejects_degenerate_and_offscreen_bounds():
    rejected = [
        {"left": 0, "top": 0, "right": 0, "bottom": 10},  # x_min >= x_max
        {"left": 10, "top": 0, "right": 5, "bottom": 10},  # x_min >= x_max
        {"left": 0, "top": 0, "right": 10, "bottom": 0},  # y_min >= y_max
        {"left": 1080, "top": 0, "right": 1200, "bottom": 10},  # x_min >= width
        {"left": -10, "top": 0, "right": 0, "bottom": 10},  # x_max <= 0
        {"left": 0, "top": 2400, "right": 10, "bottom": 2500},  # y_min >= height
        {"left": 0, "top": -10, "right": 10, "bottom": 0},  # y_max <= 0
    ]
    for bounds in rejected:
        elements = extract_elements(forest(node(boundsInScreen=bounds)), screen_size=SCREEN)
        assert elements == [], bounds
    touched = [
        {"left": 0, "top": 0, "right": 1080, "bottom": 2400},
        {"left": 1079, "top": 2399, "right": 1080, "bottom": 2400},
        {"left": 0, "top": 0, "right": 1, "bottom": 1},
    ]
    for bounds in touched:
        elements = extract_elements(forest(node(boundsInScreen=bounds)), screen_size=SCREEN)
        assert [entry["bounds"] for entry in elements] == [
            [bounds["left"], bounds["top"], bounds["right"], bounds["bottom"]]
        ], bounds


def test_extract_elements_defaults_missing_bounds_to_zero():
    # Absent keys read as 0, so a node without a usable box never validates.
    assert extract_elements(forest(node(boundsInScreen={"left": 5})), screen_size=SCREEN) == []
    assert extract_elements(forest(node(boundsInScreen=None)), screen_size=SCREEN) == []
    assert extract_elements(forest(node(boundsInScreen=[])), screen_size=SCREEN) == []
    flagless = {"isClickable": True, "isVisibleToUser": True}
    assert extract_elements(forest(flagless), screen_size=SCREEN) == []
    kept = extract_elements(
        forest(node(boundsInScreen={"right": 100, "bottom": 50})), screen_size=SCREEN
    )
    assert kept[0]["bounds"] == [0, 0, 100, 50]


def test_extract_elements_keeps_clickable_containers_and_duplicates():
    shared = {"left": 0, "top": 0, "right": 100, "bottom": 100}
    elements = extract_elements(
        forest(
            node(childIds=[1, 2, 3], boundsInScreen=shared),
            node(childIds=[], boundsInScreen=shared),
            node(boundsInScreen=shared),
        ),
        screen_size=SCREEN,
    )
    assert [entry["index"] for entry in elements] == [0, 1, 2]
    assert [entry["bounds"] for entry in elements] == [[0, 0, 100, 100]] * 3


def test_extract_elements_reads_text_fields_with_empty_defaults():
    elements = extract_elements(
        forest(
            node(text="Past", contentDescription="Past tab"),
            node(),
            node(text=5, contentDescription=None),
        ),
        screen_size=SCREEN,
    )
    assert elements[0]["text"] == "Past"
    assert elements[0]["content_description"] == "Past tab"
    assert elements[1]["text"] == ""
    assert elements[1]["content_description"] == ""
    assert elements[2]["text"] == ""
    assert elements[2]["content_description"] == ""


def test_extract_elements_payload_has_only_the_expected_keys():
    elements = extract_elements(forest(node(text="Past")), screen_size=SCREEN)
    assert set(elements[0]) == {"index", "text", "content_description", "bounds"}


def test_extract_elements_walks_windows_in_file_order():
    a11y = {
        "windows": [
            {"tree": {"nodes": [node(text="first")]}},
            {"tree": {"nodes": [node(text="second"), node(text="third")]}},
        ]
    }
    elements = extract_elements(a11y, screen_size=SCREEN)
    assert [entry["text"] for entry in elements] == ["first", "second", "third"]
    assert [entry["index"] for entry in elements] == [0, 1, 2]


def test_extract_elements_never_raises_on_malformed_forests():
    malformed = [
        None,
        [],
        "forest",
        7,
        {},
        {"windows": None},
        {"windows": {}},
        {"windows": [None, 1, "window"]},
        {"windows": [{"tree": None}]},
        {"windows": [{"tree": {"nodes": None}}]},
        {"windows": [{"tree": {"nodes": [None, 3, "node"]}}]},
        {"windows": [{"tree": {"nodes": [{"boundsInScreen": "box"}]}}]},
        {"windows": [{"tree": {"nodes": [node(boundsInScreen={"left": "x", "right": 10})]}}]},
    ]
    for a11y in malformed:
        assert extract_elements(a11y, screen_size=SCREEN) == [], a11y
    a11y = {"windows": [{"tree": {"nodes": ["junk", node(text="kept")]}}]}
    kept = extract_elements(a11y, screen_size=SCREEN)
    assert [(entry["index"], entry["text"]) for entry in kept] == [(0, "kept")]


def test_extract_elements_on_an_empty_forest():
    assert extract_elements({"windows": []}, screen_size=SCREEN) == []


def test_validate_element_ports_the_dataset_rules():
    assert validate_element(node(), screen_size=SCREEN)
    assert validate_element(node(isClickable=False), screen_size=SCREEN)
    assert not validate_element(node(isVisibleToUser=False), screen_size=SCREEN)
    assert not validate_element(
        node(boundsInScreen={"left": 10, "top": 10, "right": 10, "bottom": 20}), screen_size=SCREEN
    )
    assert not validate_element({}, screen_size=SCREEN)
    assert not validate_element("node", screen_size=SCREEN)


def test_element_bounds_rejects_unusable_boxes():
    assert element_bounds(element(0, (10, 20, 60, 80))) == (10, 20, 60, 80)
    assert element_bounds(element(0, (10.5, 20, 60, 80.25))) == (10.5, 20, 60, 80.25)
    # Inverted and zero-area boxes have no interior to act on, so nothing
    # downstream is allowed to see one.
    assert element_bounds(element(0, (10, 10, 5, 5))) is None
    assert element_bounds(element(0, (10, 10, 10, 10))) is None
    assert element_bounds(element(0, (0, 10, 10, 5))) is None
    # Nor does a coordinate that is not a finite number.
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert element_bounds(element(0, (bad, 0, 10, 10))) is None
        assert element_bounds(element(0, (0, 0, 10, bad))) is None
    # An int too large for a float is out too, rather than raising from
    # `math.isfinite`.
    for huge in (10**400, -(10**400)):
        assert element_bounds(element(0, (huge, 0, 10, 10))) is None
        assert element_bounds(element(0, (0, 0, 10, huge))) is None


def test_element_hits_lists_every_containing_element_in_ascending_order():
    elements = [
        element(0, (0, 0, 100, 100)),
        element(1, (10, 10, 40, 40)),
        element(2, (0, 0, 200, 200)),
    ]
    # Nested boxes are all ground truth: the point touched every one of them.
    assert element_hits(elements, 20, 20) == [0, 1, 2]
    assert element_hits(elements, 150, 150) == [2]
    assert element_hits(elements, 95, 95) == [0, 2]
    assert element_hits(elements, 300, 300) == []


def test_element_hits_reports_positions_not_the_index_field():
    # Only positions are returned, so a caller can index the list it passed in
    # even if the `index` fields disagree with the positions.
    elements = [element(7, (0, 0, 50, 50)), element(9, (100, 100, 200, 200))]
    assert element_hits(elements, 5, 5) == [0]
    assert element_hits(elements, 150, 150) == [1]
    assert element_hits(list(reversed(elements)), 5, 5) == [1]
    assert element_hits(list(reversed(elements)), 150, 150) == [0]


def test_element_hits_uses_closed_bounds():
    elements = [element(0, (10, 10, 20, 20))]
    assert element_hits(elements, 10, 10) == [0]
    assert element_hits(elements, 20, 20) == [0]
    assert element_hits(elements, 15, 15) == [0]
    assert element_hits(elements, 9.9, 15) == []
    assert element_hits(elements, 15, 20.1) == []


def test_element_hits_ignores_inverted_and_non_finite_element_bounds():
    elements = [
        {"bounds": [10, 10, 5, 5]},
        {"bounds": [float("nan"), 0, 10, 10]},
        {"bounds": [float("inf"), 0, 10, 10]},
        element(3, (0, 0, 100, 100)),
    ]
    assert element_hits(elements, 5, 5) == [3]
    # An unbounded box would otherwise swallow every point in the image.
    assert (
        element_hits([{"bounds": [float("-inf"), float("-inf"), float("inf"), float("inf")]}], 5, 5)
        == []
    )
    assert element_hits([{"bounds": [-(10**400), 0, 10**400, 10]}], 5, 5) == []


def test_element_hits_reports_misses_and_tolerates_malformed_input():
    assert element_hits([], 0, 0) == []
    assert element_hits([element(0, (0, 0, 10, 10))], 50, 50) == []
    assert element_hits([element(0, (0, 0, 10, 10))], float("nan"), 5) == []
    assert element_hits([element(0, (0, 0, 10, 10))], "5", 5) == []
    assert element_hits([element(0, (0, 0, 10, 10))], 5, None) == []
    unusable = [
        {"index": 0, "bounds": None},
        {"index": 1},
        {"index": 2, "bounds": [1, 2, 3]},
        {"index": 3, "bounds": [0, 0, "10", 10]},
        "junk",
        element(4, (0, 0, 100, 100)),
    ]
    assert element_hits(unusable, 5, 5) == [5]  # the position of the only usable element


def test_element_hits_reports_nothing_for_a_non_list():
    assert element_hits(None, 5, 5) == []
    assert element_hits(7, 5, 5) == []
    assert element_hits("elements", 5, 5) == []
    assert element_hits({"0": element(0, (0, 0, 10, 10))}, 5, 5) == []


def test_element_target_weights_degenerate_to_a_one_hot():
    pair = [element(0, (0, 0, 10, 10)), element(1, (100, 100, 200, 200))]
    assert element_target_weights(pair, 5, 5) == [1.0, 0.0]
    assert element_target_weights(pair, 150, 150) == [0.0, 1.0]
    # The distribution is over the list that was passed in, not a fixed id.
    assert element_target_weights(list(reversed(pair)), 5, 5) == [0.0, 1.0]


def test_element_target_weights_split_multi_hits_by_inverse_area():
    elements = [
        element(0, (0, 0, 100, 100)),  # area 10000
        element(1, (10, 10, 40, 40)),  # area 900, nested inside the first
    ]
    weights = element_target_weights(elements, 20, 20)
    assert weights is not None
    assert abs(sum(weights) - 1) <= 1e-9
    # The smaller box carries the larger share.
    assert 0 < weights[0] < weights[1] < 1
    assert weights[0] == pytest.approx(900 / 10900)
    assert weights[1] == pytest.approx(10000 / 10900)


def test_element_target_weights_are_zero_off_the_hits():
    elements = [
        element(0, (0, 0, 10, 10)),
        element(1, (0, 0, 100, 100)),
        element(2, (500, 500, 600, 600)),
    ]
    weights = element_target_weights(elements, 5, 5)
    assert weights is not None
    assert weights[2] == 0.0
    # The smaller box inside the larger one carries the larger share.
    assert 0 < weights[1] < weights[0]
    assert abs(sum(weights) - 1) <= 1e-9


def test_element_target_weights_report_a_miss_as_none():
    assert element_target_weights([element(0, (0, 0, 10, 10))], 50, 50) is None
    assert element_target_weights([], 0, 0) is None
    assert element_target_weights(None, 0, 0) is None
    assert element_target_weights(7, 0, 0) is None
    assert element_target_weights([element(0, (0, 0, 10, 10))], float("nan"), 5) is None
    assert element_target_weights([element(0, (0, 0, 10, 10))], 5, "5") is None


def test_element_target_weights_survive_an_area_that_leaves_the_float_range():
    # Two bounds either side of 1e154 multiply past the float maximum: an exact
    # int product would raise from `math.isfinite`, and every inverse would be
    # 0.0, so the shares have to fall back to uniform instead of 0.0 / 0.0.
    elements = [
        element(0, (0, 0, 2e154, 2e154)),
        element(1, (0, 0, 3e154, 3e154)),
    ]
    weights = element_target_weights(elements, 1, 1)
    assert weights == [0.5, 0.5]
    assert abs(sum(weights) - 1) <= 1e-9
    # A lone overflowing hit still degenerates to its own one-hot.
    assert element_target_weights([element(0, (0, 0, 2e154, 2e154))], 1, 1) == [1.0]
    assert element_target_weights([element(0, (10**400, 0, 2 * 10**400, 10))], 0, 0) is None


def test_element_target_weights_give_an_overflowing_hit_no_mass():
    elements = [
        element(0, (0, 0, 2e154, 2e154)),  # area inf: the share is 0.0
        element(1, (0, 0, 10, 10)),  # area 100
    ]
    assert element_hits(elements, 5, 5) == [0, 1]
    weights = element_target_weights(elements, 5, 5)
    assert weights == [0.0, 1.0]


def test_element_target_weights_fall_back_to_uniform_when_an_area_underflows():
    # Extents small enough to underflow the product leave no ratio to invert,
    # so every hit counts the same rather than the division blowing up.
    elements = [
        element(0, (0, 0, 5e-200, 5e-200)),
        element(1, (0, 0, 10, 10)),
    ]
    weights = element_target_weights(elements, 1e-201, 1e-201)
    assert weights == [0.5, 0.5]


def test_element_target_weights_ignore_unusable_boxes():
    elements = [{"index": 0, "bounds": None}, element(1, (0, 0, 10, 10))]
    assert element_target_weights(elements, 5, 5) == [0.0, 1.0]


def test_resolve_element_target_enforces_the_candidate_count():
    assert (MIN_CANDIDATES, MAX_CANDIDATES) == (2, 128)
    assert resolve_element_target([], 0, 0) == (None, "too_few_candidates")
    assert resolve_element_target([element(0, (0, 0, 10, 10))], 5, 5) == (
        None,
        "too_few_candidates",
    )
    too_many = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES + 1)]
    assert resolve_element_target(too_many, 5, 5) == (None, "too_many_candidates")


def test_resolve_element_target_checks_the_count_before_the_hit_test():
    too_many = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES + 1)]
    assert resolve_element_target(too_many, 5000, 5000) == (None, "too_many_candidates")
    full = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES)]
    assert resolve_element_target(full, 5000, 5000) == (None, "no_target_element")


def test_resolve_element_target_returns_the_weights_of_a_hit():
    pair = [element(0, (0, 0, 10, 10)), element(1, (100, 100, 200, 200))]
    assert resolve_element_target(pair, 5, 5) == ([1.0, 0.0], None)
    assert resolve_element_target(pair, 150, 150) == ([0.0, 1.0], None)
    assert resolve_element_target(pair, 50, 50) == (None, "no_target_element")
    # The result indexes the list that was passed in, not a fixed id.
    assert resolve_element_target(list(reversed(pair)), 150, 150) == ([1.0, 0.0], None)
    assert resolve_element_target(list(reversed(pair)), 5, 5) == ([0.0, 1.0], None)


def test_resolve_element_target_accepts_the_full_candidate_list():
    elements = [element(index, (index * 10, 0, index * 10 + 8, 8)) for index in range(128)]
    weights, reason = resolve_element_target(elements, 1005, 4)
    assert reason is None
    assert weights == [float(index == 100) for index in range(128)]


def test_resolve_element_target_reports_a_non_list_as_too_few():
    assert resolve_element_target(None, 5, 5) == (None, "too_few_candidates")
    assert resolve_element_target(7, 5, 5) == (None, "too_few_candidates")
    assert resolve_element_target({}, 5, 5) == (None, "too_few_candidates")


def test_extract_elements_numbers_every_element_with_its_position():
    forests = [
        forest(),
        forest(node(text="kept"), node(isClickable=False), node(text="also kept")),
        {
            "windows": [
                {"tree": {"nodes": [node(text="a"), node(isVisibleToUser=False)]}},
                {"tree": {"nodes": [node(text="b")]}},
            ]
        },
    ]
    for a11y in forests:
        elements = extract_elements(a11y, screen_size=SCREEN)
        assert [entry["index"] for entry in elements] == list(range(len(elements))), a11y


def test_extract_hits_and_description_agree_at_the_seam():
    # The seam Tasks 2 and 3 rely on: the weights, the description label and the
    # marked element all name positions in the very candidate list handed in.
    bounds = {"left": 100, "top": 200, "right": 300, "bottom": 260}
    a11y = forest(
        node(text="not clickable", isClickable=False, boundsInScreen=bounds),
        node(text="TARGET", boundsInScreen=bounds),
        node(text="elsewhere", boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10}),
    )
    elements = extract_elements(a11y, screen_size=SCREEN)
    weights, reason = resolve_element_target(elements, 200, 230)
    assert reason is None
    assert weights == [1.0, 0.0]
    assert element_hits(elements, 200, 230) == [0]
    position = element_hits(elements, 200, 230)[0]
    target = elements[position]
    assert target["index"] == position
    assert target["text"] == "TARGET"
    assert target["bounds"][0] <= 200 <= target["bounds"][2]
    description = element_description(position, target)
    assert description == 'UI element 0: {"text": "TARGET"}'
    assert json.loads(description.removeprefix("UI element 0: ")) == {"text": "TARGET"}


def test_element_description_renders_text_then_content_description():
    both = element_description(15, element(15, (0, 0, 1, 1), "Past", "Past tab"))
    assert both == 'UI element 15: {"text": "Past", "content_description": "Past tab"}'


def test_element_description_renders_each_field_alone():
    assert element_description(15, element(15, (0, 0, 1, 1), text="JOIN A MEETING")) == (
        'UI element 15: {"text": "JOIN A MEETING"}'
    )
    assert element_description(16, element(16, (0, 0, 1, 1), content_description="Back")) == (
        'UI element 16: {"content_description": "Back"}'
    )


def test_element_description_renders_empty_payloads_as_an_object():
    assert element_description(17, element(17, (0, 0, 1, 1))) == "UI element 17: {}"
    assert element_description(0, {}) == "UI element 0: {}"
    assert element_description(0, {"text": 5, "content_description": None}) == "UI element 0: {}"
    assert element_description(0, element(0, (0, 0, 1, 1), text="", content_description="")) == (
        "UI element 0: {}"
    )


def test_element_description_never_leaks_other_element_fields():
    entry = element(3, (0, 0, 1, 1), text="Past")
    entry.update(
        {"is_clickable": True, "is_scrollable": True, "class_name": "android.widget.TextView"}
    )
    description = element_description(3, entry)
    assert description == 'UI element 3: {"text": "Past"}'
    assert json.loads(description.removeprefix("UI element 3: ")) == {"text": "Past"}


def test_element_description_keeps_non_ascii_text_readable():
    assert element_description(0, element(0, (0, 0, 1, 1), text="设置")) == (
        'UI element 0: {"text": "设置"}'
    )


def test_parse_metadata_accepts_a_well_formed_episode():
    record = metadata()
    parsed, reason = parse_metadata(record)
    assert reason is None
    assert parsed is record
    assert parsed["steps"][-1]["action"] is None


def test_parse_metadata_accepts_empty_steps_and_unknown_action_types():
    parsed, reason = parse_metadata(metadata(steps=[]))
    assert (parsed is not None, reason) == (True, None)
    record = metadata()
    record["steps"][0]["action"] = {"action_type": "teleport"}
    parsed, reason = parse_metadata(record)
    assert reason is None
    assert parsed["steps"][0]["action"] == {"action_type": "teleport"}


def test_parse_metadata_rejects_unparsable_envelopes():
    malformed = [
        None,
        [],
        "episode",
        7,
        {},
        {"goal": "g", "steps": []},
        {"episode_id": 0, "steps": []},
        {"episode_id": "0", "goal": "g", "steps": []},
        {"episode_id": True, "goal": "g", "steps": []},
        {"episode_id": 0.5, "goal": "g", "steps": []},
        {"episode_id": 0, "goal": 5, "steps": []},
        {"episode_id": 0, "goal": None, "steps": []},
        {"episode_id": 0, "goal": ["goal"], "steps": []},
        {"episode_id": 0, "goal": "g"},
        {"episode_id": 0, "goal": "g", "steps": None},
        {"episode_id": 0, "goal": "g", "steps": {}},
        {"episode_id": 0, "goal": "g", "steps": "steps"},
        {"episode_id": 0, "goal": "g", "steps": [None]},
        {"episode_id": 0, "goal": "g", "steps": [1, 2]},
    ]
    for record in malformed:
        assert parse_metadata(record) == (None, "unparsable_metadata"), record


def test_parse_metadata_rejects_malformed_steps():
    malformed = [
        step(step_id="0"),
        step(step_id=True),
        step(step_id=0.0),
        step(step_id=None),
        step(screenshot=""),
        step(screenshot=None),
        step(screenshot=7),
        step(accessibility_tree=""),
        step(accessibility_tree=["step_000_a11y.json"]),
        step(action="click"),
        step(action=[]),
        step(step_instruction=5),
        step(step_instruction=["Go to the Past section"]),
        {},
    ]
    for entry in malformed:
        record = metadata(steps=[entry])
        assert parse_metadata(record) == (None, "unparsable_metadata"), entry
    assert (
        parse_metadata(metadata(steps=[step(action=None), step(step_instruction=None)]))[1] is None
    )


def test_parse_metadata_requires_the_step_keys_to_be_present():
    for key in ("action", "step_instruction"):
        entry = step()
        del entry[key]
        assert parse_metadata(metadata(steps=[entry])) == (None, "unparsable_metadata"), key


def test_parse_metadata_never_raises_on_malformed_records():
    malformed = [
        None,
        [],
        "record",
        0,
        {"episode_id": 0, "goal": "g", "steps": [{"step_id": 0}]},
        {"episode_id": 0, "goal": "g", "steps": [{"action": {"action_type": 1}}]},
        {"episode_id": 0, "goal": "g", "steps": [[{"step_id": 0}]]},
        {"episode_id": 0, "goal": "g", "steps": {"0": {}}},
        {"episode_id": 0, "goal": "g", "steps": [{"step_id": 0, "screenshot": object()}]},
    ]
    for record in malformed:
        parsed, reason = parse_metadata(record)
        assert parsed is None
        assert reason == "unparsable_metadata"


def test_metadata_detail_names_the_first_broken_field():
    assert metadata_detail(metadata()) == ""
    assert metadata_detail(metadata(steps=[])) == ""
    assert metadata_detail(None) == "document is not an object"
    assert metadata_detail([]) == "document is not an object"
    assert metadata_detail("episode") == "document is not an object"
    assert metadata_detail({}) == "key 'episode_id' is not an integer"
    assert metadata_detail({"episode_id": 0, "goal": 5, "steps": []}) == (
        "key 'goal' is not a string"
    )
    assert metadata_detail({"episode_id": 0, "goal": "g"}) == "key 'steps' is not a list"
    assert metadata_detail({"episode_id": 0, "goal": "g", "steps": {}}) == (
        "key 'steps' is not a list"
    )


def test_metadata_detail_names_the_first_broken_step():
    assert metadata_detail(metadata(steps=[None])) == "step 0: step entry is not an object"
    assert metadata_detail(metadata(steps=[step(step_id="0")])) == (
        "step 0: key 'step_id' is not an integer"
    )
    assert metadata_detail(metadata(steps=[step(screenshot="")])) == (
        "step 0: key 'screenshot' is not a non-empty file name"
    )
    assert metadata_detail(metadata(steps=[step(accessibility_tree=7)])) == (
        "step 0: key 'accessibility_tree' is not a non-empty file name"
    )
    assert metadata_detail(metadata(steps=[step(action="click")])) == (
        "step 0: key 'action' is neither an object nor null"
    )
    assert metadata_detail(metadata(steps=[step(step_instruction=5)])) == (
        "step 0: key 'step_instruction' is neither a string nor null"
    )
    dropped = step()
    del dropped["action"]
    assert metadata_detail(metadata(steps=[dropped])) == "step 0: missing key 'action'"
    dropped = step()
    del dropped["step_instruction"]
    assert metadata_detail(metadata(steps=[dropped])) == "step 0: missing key 'step_instruction'"
    # Only the first failure is reported, with the index of the step it is in.
    assert metadata_detail(metadata(steps=[step(), None, step(step_id=None)])) == (
        "step 1: step entry is not an object"
    )
    assert metadata_detail(metadata(steps=[step(step_id=0.5)])) == (
        "step 0: key 'step_id' is not an integer"
    )


def test_metadata_detail_agrees_with_parse_metadata():
    # The detail and the reason come from the same walk, so one can never accept
    # a document the other refuses.
    records = [
        metadata(),
        metadata(steps=[]),
        metadata(steps=[step(step_id="0")]),
        metadata(steps=[None]),
        {"episode_id": 0, "goal": "g"},
        {"episode_id": 0, "goal": "g", "steps": [{"step_id": 0}]},
        {},
        [],
        None,
    ]
    for record in records:
        parsed, reason = parse_metadata(record)
        assert (reason is None) == (metadata_detail(record) == ""), record
        assert (parsed is None) == (reason is not None)


# --- Task 2: per-step parsing and the five question families -------------------

GOAL = "Open the Zoho Meet app , view the scheduled meetings ."
SCREENSHOT = (100, 200)
RAW_IMAGE = "episode/step_000_screenshot.png"
MARKED_IMAGE = "episode/step_000_marked.png"


def write_png(path: Path, *, size=SCREENSHOT, color="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_a11y(path: Path, *nodes) -> Path:
    """Write the accessibility forest a click step points at."""
    path.write_text(json.dumps(forest(*nodes)))
    return path


def ac_step(index, action, instruction, *, directory: Path, nodes=None):
    """One metadata step entry, with the files it points at already on disk.

    `nodes=None` writes no accessibility file at all, so a test can prove the
    step never opened one.
    """
    stem = f"step_{index:03d}"
    write_png(directory / f"{stem}_screenshot.png")
    if nodes is not None:
        write_a11y(directory / f"{stem}_a11y.json", *nodes)
    return {
        "step_id": index,
        "screenshot": f"{stem}_screenshot.png",
        "accessibility_tree": f"{stem}_a11y.json",
        "action": action,
        "step_instruction": instruction,
    }


def ac_episode(*steps, goal=GOAL, episode_id=7):
    return {"episode_id": episode_id, "goal": goal, "steps": list(steps)}


def parse_ac_step(record, index, *, directory):
    step, reason = parse_step(record, index, episode_dir=directory)
    assert reason is None
    assert step is not None
    return step


def click_nodes():
    """Two clickable nodes; a click at (75, 75) resolves to position 1."""
    return [
        node(text="Cancel", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50}),
        node(
            text="JOIN A MEETING",
            contentDescription="Join the next meeting",
            boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100},
        ),
    ]


def test_parse_step_reads_a_click_step_and_resolves_the_target(tmp_path):
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
    step = parse_ac_step(record, 0, directory=directory)
    assert step.id == "android_control_7_step0"
    assert step.group == "task:android_control_7"
    assert step.action == "click"
    assert step.arguments == {"action": "click", "coordinate": [75, 75]}
    assert step.ac_action == {"action_type": "click", "x": 75, "y": 75}
    assert step.instruction == "Tap JOIN A MEETING"
    assert [entry["text"] for entry in step.elements] == ["Cancel", "JOIN A MEETING"]
    assert step.element_weights == [0.0, 1.0]
    assert step.image == directory / "step_000_screenshot.png"
    assert step.image_sha256 == image_digest(directory / "step_000_screenshot.png")
    assert step.state == {
        "user_query": GOAL,
        "task_progress": "(You have done the following operation on the current device): .",
    }
    assert step.reference == {
        "thought": "",
        "action": "Tap JOIN A MEETING",
        "tool_call": {
            "name": "mobile_use",
            "arguments": {"action": "click", "coordinate": [75, 75]},
        },
        "ac_action": {"action_type": "click", "x": 75, "y": 75},
        "element_positions": [1],
    }


def test_click_step_yields_action_complete_and_element_rows(tmp_path):
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
    step = parse_ac_step(record, 0, directory=directory)
    marked = (MARKED_IMAGE, image_digest(write_png(directory / "step_000_marked.png")))
    rows = rows_for_ac_step(step, RAW_IMAGE, marked)
    assert [row["id"] for row in rows] == [
        "android_control_7_step0:action",
        "android_control_7_step0:complete",
        "android_control_7_step0:element",
    ]
    assert [row["dataset"] for row in rows] == ["gui_action", "gui_complete", "screenshot_choice"]
    assert {row["group"] for row in rows} == {"task:android_control_7"}
    assert {row["split"] for row in rows} == {split_for("task:android_control_7")}
    # state and reference are shared by every row of the step; the alias list
    # is equal by content only, so annotating one row cannot touch the others.
    assert rows[0]["state"] is rows[2]["state"]
    assert rows[0]["reference"] is rows[2]["reference"]
    assert rows[0]["aliases"] == rows[2]["aliases"]
    assert rows[0]["aliases"] is not rows[2]["aliases"]
    choice_aliases = rows[2]["aliases"].copy()
    rows[2]["aliases"].append("image-bytes:annotated")
    assert rows[0]["aliases"] == choice_aliases
    rows[2]["aliases"][:] = choice_aliases
    action, complete, choice = rows
    assert action["question"]["type"] == "choice"
    assert action["question"]["instructions"] == INSTRUCTIONS["action"]
    assert list(action["question"]["criteria"]) == list(AC_ACTIONS)
    assert action["target"][list(AC_ACTIONS).index("click")] == 1.0
    assert sum(action["target"]) == 1.0
    assert complete["question"] == {
        "type": "noul",
        "instructions": INSTRUCTIONS["complete"],
        "criteria": dict(COMPLETE_CRITERIA),
    }
    assert complete["target"] == [1.0, 0.0]
    assert choice["question"]["type"] == "choice"
    assert choice["question"]["instructions"] == ELEMENT_INSTRUCTION
    assert list(choice["question"]["criteria"]) == ["r0", "r1"]
    assert choice["question"]["criteria"] == {
        f"r{index}": element_description(index, element)
        for index, element in enumerate(step.elements)
    }
    assert choice["target"] == [0.0, 1.0]
    assert choice["image"] == MARKED_IMAGE
    assert {row["image"] for row in rows[:2]} == {RAW_IMAGE}
    assert rows[0]["aliases"] == [
        "image-bytes:" + step.image_sha256,
        "image-bytes:" + marked[1],
    ]


def test_click_step_element_criteria_carry_only_text_and_description(tmp_path):
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
    step = parse_ac_step(record, 0, directory=directory)
    rows = rows_for_ac_step(step, RAW_IMAGE, (MARKED_IMAGE, "marked"))
    criteria = rows[2]["question"]["criteria"]
    assert criteria["r0"] == 'UI element 0: {"text": "Cancel"}'
    assert criteria["r1"] == (
        'UI element 1: {"text": "JOIN A MEETING", "content_description": "Join the next meeting"}'
    )
    for index, payload in enumerate(criteria.values()):
        assert set(json.loads(payload.removeprefix(f"UI element {index}: "))) <= {
            "text",
            "content_description",
        }


def test_scroll_step_yields_action_complete_and_swipe_dir_rows(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0, {"action_type": "scroll", "direction": "down"}, "Scroll down", directory=directory
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    assert step.arguments == {"action": "swipe", "direction": "up"}
    assert step.elements == []
    assert step.element_weights is None
    rows = rows_for_ac_step(step, RAW_IMAGE)
    assert [row["id"] for row in rows] == [
        "android_control_7_step0:action",
        "android_control_7_step0:complete",
        "android_control_7_step0:swipe_dir",
    ]
    assert [row["dataset"] for row in rows] == ["gui_action", "gui_complete", "gui_swipe"]
    assert rows[0]["target"][list(AC_ACTIONS).index("swipe")] == 1.0
    # Android Control records the finger, gui-v1 the content: scrolling down is
    # an upward swipe.
    swipe = rows[2]
    assert swipe["question"]["type"] == "choice"
    assert swipe["question"]["instructions"] == INSTRUCTIONS["swipe_dir"]
    assert list(swipe["question"]["criteria"]) == list(SWIPE_DIRECTIONS)
    assert swipe["target"] == [float(name == "up") for name in SWIPE_DIRECTIONS]
    assert swipe["image"] == RAW_IMAGE
    assert swipe["aliases"] == ["image-bytes:" + step.image_sha256]


def test_system_button_step_yields_action_complete_and_button_rows(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "navigate_back"}, "Press back", directory=directory)
    )
    step = parse_ac_step(record, 0, directory=directory)
    rows = rows_for_ac_step(step, RAW_IMAGE)
    assert [row["id"] for row in rows] == [
        "android_control_7_step0:action",
        "android_control_7_step0:complete",
        "android_control_7_step0:button",
    ]
    assert [row["dataset"] for row in rows] == ["gui_action", "gui_complete", "gui_button"]
    button = rows[2]
    assert button["target"] == [float(name == "Back") for name in BUTTONS]
    assert button["question"]["instructions"] == INSTRUCTIONS["button"]
    assert list(button["question"]["criteria"]) == list(BUTTONS)
    assert button["reference"]["tool_call"]["arguments"] == {
        "action": "system_button",
        "button": "Back",
    }


def test_open_app_step_target_is_the_last_action(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "open_app", "app_name": "Zoho Meeting"},
            "Open the app",
            directory=directory,
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    rows = rows_for_ac_step(step, RAW_IMAGE)
    assert [row["id"] for row in rows] == [
        "android_control_7_step0:action",
        "android_control_7_step0:complete",
    ]
    assert list(AC_ACTIONS).index("open_app") == 8
    assert rows[0]["target"][8] == 1.0
    assert rows[0]["target"][-1] == 1.0


def test_wait_and_type_steps_have_no_conditional_row(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "Wait", directory=directory),
        ac_step(
            1, {"action_type": "input_text", "text": "hello"}, "Type hello", directory=directory
        ),
    )
    for index, action in enumerate(("wait", "type")):
        step = parse_ac_step(record, index, directory=directory)
        assert step.action == action
        assert step.elements == []
        assert step.element_weights is None
        rows = rows_for_ac_step(step, RAW_IMAGE)
        assert [row["id"] for row in rows] == [
            f"android_control_7_step{index}:action",
            f"android_control_7_step{index}:complete",
        ]
        assert rows[0]["target"][list(AC_ACTIONS).index(action)] == 1.0
        assert rows[1]["target"] == [1.0, 0.0]


def test_terminal_step_yields_terminate_rows_without_an_accessibility_tree(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "Wait", directory=directory),
        ac_step(1, None, None, directory=directory),  # no a11y file on disk
    )
    step = parse_ac_step(record, 1, directory=directory)
    assert step.action == "terminate"
    assert step.arguments == {"action": "terminate"}
    assert step.ac_action is None
    assert step.instruction is None
    assert step.elements == []
    assert step.element_weights is None
    assert step.reference == {
        "thought": "",
        "action": "",
        "tool_call": {"name": "mobile_use", "arguments": {"action": "terminate"}},
        "ac_action": None,
        "element_positions": [],
    }
    rows = rows_for_ac_step(step, "episode/step_001_screenshot.png")
    assert [row["id"] for row in rows] == [
        "android_control_7_step1:action",
        "android_control_7_step1:complete",
    ]
    assert rows[0]["target"][list(AC_ACTIONS).index("terminate")] == 1.0
    assert rows[1]["target"] == [0.0, 1.0]
    assert rows[0]["aliases"] == ["image-bytes:" + step.image_sha256]


def test_rows_for_ac_step_requires_the_marked_screenshot_for_element_rows(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    with pytest.raises(ValueError, match="marked screenshot"):
        rows_for_ac_step(step, RAW_IMAGE)
    # A step without an element row never needs the marked copy: passing one is
    # ignored rather than an error.
    waited = parse_ac_step(
        ac_episode(ac_step(0, {"action_type": "wait"}, "Wait", directory=directory)),
        0,
        directory=directory,
    )
    assert rows_for_ac_step(waited, RAW_IMAGE, (MARKED_IMAGE, "marked"))[0]["aliases"] == [
        "image-bytes:" + waited.image_sha256
    ]


def test_rows_for_ac_step_rejects_element_weights_over_the_wrong_candidates(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    # A hand-built step must not mint an all-zeros target, which would silently
    # teach the model that no candidate is the answer, and it must describe the
    # very candidate list the row is asked about.
    for broken in (
        replace(step, element_weights=None),
        replace(step, element_weights=[0.0] * len(step.elements)),
        replace(step, element_weights=[0.0]),
        replace(step, element_weights=[1.0, 0.0, 0.0]),
        replace(step, elements=[]),
    ):
        with pytest.raises(ValueError, match="element weights"):
            rows_for_ac_step(broken, RAW_IMAGE, (MARKED_IMAGE, "marked"))
    assert rows_for_ac_step(step, RAW_IMAGE, (MARKED_IMAGE, "marked"))[2]["target"] == [0.0, 1.0]


def test_multi_hit_element_rows_are_soft_and_pass_validate_rows(tmp_path):
    directory = tmp_path / "episode"
    nodes = [
        node(text="Card", boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 100}),
        node(text="Button", boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100}),
        node(text="Elsewhere", boundsInScreen={"left": 10, "top": 150, "right": 20, "bottom": 160}),
    ]
    record = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 75, "y": 75}, "Tap", directory=directory, nodes=nodes
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    assert step.reference["element_positions"] == [0, 1]
    weights = step.element_weights
    assert weights is not None
    assert weights[2] == 0.0
    assert 0 < weights[0] < weights[1] < 1
    assert abs(sum(weights) - 1) <= 1e-9
    marked = (MARKED_IMAGE, image_digest(write_png(directory / "step_000_marked.png")))
    rows = rows_for_ac_step(step, RAW_IMAGE, marked)
    assert rows[2]["target"] == step.element_weights
    validate_rows(rows, root=tmp_path)


def test_element_positions_report_the_support_of_the_weights(tmp_path):
    directory = tmp_path / "episode"
    # The huge box contains the tap point as well, but its area is past the
    # float maximum, so it carries no weight and is not one of the answers the
    # row teaches: provenance follows the distribution, not the raw hit list.
    huge = node(boundsInScreen={"left": 0, "top": 0, "right": 2e154, "bottom": 2e154})
    small = node(text="TARGET", boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 100})
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap",
            directory=directory,
            nodes=[huge, small],
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    assert element_hits(step.elements, 75, 75) == [0, 1]
    assert step.element_weights == [0.0, 1.0]
    assert step.reference["element_positions"] == [1]


def test_state_template_is_empty_at_the_first_step(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "Wait for the app to load", directory=directory),
        ac_step(1, None, None, directory=directory),
    )
    step = parse_ac_step(record, 0, directory=directory)
    assert step.state == {
        "user_query": GOAL,
        "task_progress": "(You have done the following operation on the current device): .",
    }


def test_state_template_numbers_only_the_completed_instructions(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, "A", directory=directory),
        ac_step(1, {"action_type": "wait"}, "B", directory=directory),
        ac_step(2, {"action_type": "wait"}, "C", directory=directory),
    )
    step = parse_ac_step(record, 2, directory=directory)
    assert step.state["task_progress"] == (
        "(You have done the following operation on the current device): Step 1: A; Step 2: B; ."
    )
    # The current step's instruction is the answer and never reaches the state.
    assert step.instruction == "C"
    assert "C" not in step.state["task_progress"]


def test_state_template_skips_missing_instructions_without_renumbering(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "wait"}, None, directory=directory),
        ac_step(1, {"action_type": "wait"}, "A", directory=directory),
        ac_step(2, {"action_type": "wait"}, "B", directory=directory),
        ac_step(3, {"action_type": "wait"}, "C", directory=directory),
    )
    step = parse_ac_step(record, 3, directory=directory)
    assert step.state["task_progress"] == (
        "(You have done the following operation on the current device): Step 1: A; Step 2: B; ."
    )
    assert step.state["user_query"] == GOAL


def test_state_carries_the_goal_verbatim(tmp_path):
    directory = tmp_path / "episode"
    goal = "Book  a  table , then   call the office ."
    record = ac_episode(ac_step(0, {"action_type": "wait"}, "Wait", directory=directory), goal=goal)
    step = parse_ac_step(record, 0, directory=directory)
    assert step.state["user_query"] == goal


def test_reference_keeps_the_raw_action_and_the_mapped_arguments(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0, {"action_type": "scroll", "direction": "left"}, "Scroll left", directory=directory
        )
    )
    step = parse_ac_step(record, 0, directory=directory)
    assert step.reference == {
        "thought": "",
        "action": "Scroll left",
        "tool_call": {
            "name": "mobile_use",
            "arguments": {"action": "swipe", "direction": "right"},
        },
        "ac_action": {"action_type": "scroll", "direction": "left"},
        "element_positions": [],
    }


def test_parse_step_excludes_unknown_actions(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(ac_step(0, {"action_type": "teleport"}, "Teleport", directory=directory))
    assert parse_step(record, 0, episode_dir=directory) == (None, "unknown_action")
    record = ac_episode(
        ac_step(0, {"action_type": "click", "x": "540", "y": 390}, "Tap", directory=directory)
    )
    assert parse_step(record, 0, episode_dir=directory) == (None, "unknown_action")


def test_parse_step_excludes_missing_and_undecodable_screenshots(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(ac_step(0, {"action_type": "wait"}, "Wait", directory=directory))
    record["steps"][0]["screenshot"] = "gone.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")
    record["steps"][0]["screenshot"] = "../step_000_screenshot.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")
    record["steps"][0]["screenshot"] = "/etc/hostname"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")
    # A screenshot name that is not a usable file name is a structure problem,
    # so it is reported by the step check rather than by the reader.
    record["steps"][0]["screenshot"] = ""
    assert parse_step(record, 0, episode_dir=directory) == (None, "unparsable_metadata")
    (directory / "junk.png").write_bytes(b"not an image")
    record["steps"][0]["screenshot"] = "junk.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")
    (directory / "empty.png").write_bytes(b"")
    record["steps"][0]["screenshot"] = "empty.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")


def test_parse_step_excludes_truncated_screenshots(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(ac_step(0, {"action_type": "wait"}, "Wait", directory=directory))
    whole = (directory / "step_000_screenshot.png").read_bytes()
    # The header still reads, so only a container walk can tell the file is cut.
    truncated = directory / "truncated.png"
    truncated.write_bytes(whole[: len(whole) // 2])
    record["steps"][0]["screenshot"] = "truncated.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")
    header_only = directory / "header_only.png"
    header_only.write_bytes(whole[: 8 + 25])
    record["steps"][0]["screenshot"] = "header_only.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")


def test_parse_step_excludes_click_steps_without_a_usable_a11y(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(0, {"action_type": "click", "x": 75, "y": 75}, "Tap", directory=directory)
    )
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_a11y")
    tree = directory / "step_000_a11y.json"
    tree.write_text("{not json")
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_a11y")
    for document in ("[1, 2]", "7", '{"windows": null}', '{"windows": {}}', "{}", "null"):
        tree.write_text(document)
        assert parse_step(record, 0, episode_dir=directory) == (None, "missing_a11y"), document
    record["steps"][0]["accessibility_tree"] = "../step_000_a11y.json"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_a11y")


def test_parse_step_excludes_click_steps_whose_target_does_not_resolve(tmp_path):
    directory = tmp_path / "episode"
    click = {"action_type": "click", "x": 75, "y": 75}
    # No clickable element at all.
    record = ac_episode(ac_step(0, click, "Tap", directory=directory, nodes=[]))
    assert parse_step(record, 0, episode_dir=directory) == (None, "too_few_candidates")
    # Exactly one candidate is below the two-candidate floor.
    record = ac_episode(
        ac_step(0, click, "Tap", directory=directory, nodes=[node(text="Only one")])
    )
    assert parse_step(record, 0, episode_dir=directory) == (None, "too_few_candidates")
    # Two candidates, but the click lands outside both.
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 5, "y": 190},
            "Tap",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    assert parse_step(record, 0, episode_dir=directory) == (None, "no_target_element")


def test_parse_step_excludes_click_steps_with_too_many_candidates(tmp_path):
    directory = tmp_path / "episode"
    boxes = [
        node(
            text=f"box {index}",
            boundsInScreen={"left": 0, "top": index, "right": 10, "bottom": index + 1},
        )
        for index in range(MAX_CANDIDATES + 1)
    ]
    record = ac_episode(
        ac_step(
            0, {"action_type": "click", "x": 5, "y": 5}, "Tap", directory=directory, nodes=boxes
        )
    )
    assert parse_step(record, 0, episode_dir=directory) == (None, "too_many_candidates")


def test_parse_step_never_reads_the_a11y_tree_for_other_actions(tmp_path):
    directory = tmp_path / "episode"
    actions = [
        {"action_type": "wait"},
        {"action_type": "scroll", "direction": "up"},
        {"action_type": "input_text", "text": "hello"},
        {"action_type": "open_app", "app_name": "Maps"},
        {"action_type": "navigate_back"},
        {"action_type": "navigate_home"},
        None,
    ]
    steps = [
        ac_step(index, action, "Do it", directory=directory) for index, action in enumerate(actions)
    ]
    record = ac_episode(*steps)
    for index in range(len(actions)):
        step, reason = parse_step(record, index, episode_dir=directory)
        assert reason is None, actions[index]
        assert step.elements == []
        assert step.element_weights is None


def test_parse_step_checks_the_screenshot_before_the_action(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(ac_step(0, {"action_type": "teleport"}, "Teleport", directory=directory))
    record["steps"][0]["screenshot"] = "gone.png"
    assert parse_step(record, 0, episode_dir=directory) == (None, "missing_image")


def test_parse_step_never_raises_on_malformed_records(tmp_path):
    malformed = [
        None,
        [],
        "record",
        7,
        {},
        {"episode_id": 7, "goal": "g"},
        {"episode_id": 7, "goal": "g", "steps": None},
        {"episode_id": 7, "goal": "g", "steps": {}},
        {"episode_id": 7, "goal": "g", "steps": ["step"]},
        {"episode_id": 7, "goal": "g", "steps": [None]},
        {"episode_id": 7, "goal": "g", "steps": [{}]},
        {"episode_id": "7", "goal": "g", "steps": []},
        {"episode_id": True, "goal": "g", "steps": []},
        {"episode_id": 7, "goal": None, "steps": []},
        {"episode_id": 7, "goal": 7, "steps": []},
    ]
    for record in malformed:
        step, reason = parse_step(record, 0, episode_dir=tmp_path)
        assert step is None, record
        assert isinstance(reason, str), record
    directory = tmp_path / "episode"
    record = ac_episode(ac_step(0, {"action_type": "wait"}, "Wait", directory=directory))
    for index in (-1, 1, 99, "0", None, True, 1.5):
        step, reason = parse_step(record, index, episode_dir=directory)
        assert step is None, index
        assert isinstance(reason, str), index


def test_parse_step_never_raises_on_junk_inside_a_valid_step(tmp_path):
    directory = tmp_path / "episode"
    entry = ac_step(
        0,
        {"action_type": "click", "x": 75, "y": 75, "junk": object()},
        "Tap",
        directory=directory,
        nodes=[
            {"boundsInScreen": "box", "isClickable": True, "isVisibleToUser": True},
            node(text="ok", boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100}),
            "junk",
            node(isClickable="true"),
            node(text="also ok", boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50}),
            node(text=5, contentDescription=None, boundsInScreen={"left": "x", "right": 10}),
        ],
    )
    record = ac_episode(entry)
    step, reason = parse_step(record, 0, episode_dir=directory)
    assert reason is None
    assert step.element_weights == [1.0, 0.0]
    assert [element["text"] for element in step.elements] == ["ok", "also ok"]
    assert step.reference == {
        "thought": "",
        "action": "Tap",
        "tool_call": {
            "name": "mobile_use",
            "arguments": {"action": "click", "coordinate": [75, 75]},
        },
        "ac_action": {"action_type": "click", "x": 75, "y": 75, "junk": entry["action"]["junk"]},
        "element_positions": [0],
    }
    rows = rows_for_ac_step(step, RAW_IMAGE, (MARKED_IMAGE, "marked"))
    assert len(rows) == 3


def test_parse_step_rejects_a_step_that_dropped_the_action_key(tmp_path):
    directory = tmp_path / "episode"
    entry = ac_step(0, None, None, directory=directory)
    del entry["action"]
    # A missing key is not a null action: it must never read as terminate, so
    # the step is excluded and no row is minted for it.
    assert parse_step(ac_episode(entry), 0, episode_dir=directory) == (
        None,
        "unparsable_metadata",
    )
    entry = ac_step(0, {"action_type": "wait"}, "Wait", directory=directory)
    del entry["step_instruction"]
    assert parse_step(ac_episode(entry), 0, episode_dir=directory) == (
        None,
        "unparsable_metadata",
    )
    # The real terminal shape keeps the key and still parses, with both rows.
    terminal = ac_step(0, None, None, directory=directory)
    step = parse_ac_step(ac_episode(terminal), 0, directory=directory)
    assert step.action == "terminate"
    assert [row["id"] for row in rows_for_ac_step(step, RAW_IMAGE)] == [
        "android_control_7_step0:action",
        "android_control_7_step0:complete",
    ]


def test_parse_step_rejects_a_step_with_a_junk_instruction(tmp_path):
    directory = tmp_path / "episode"
    # The key is there but is not a string: the step does not match the schema,
    # so it is excluded instead of parsing with no instruction.
    record = ac_episode(ac_step(0, {"action_type": "wait"}, 42, directory=directory))
    assert parse_step(record, 0, episode_dir=directory) == (None, "unparsable_metadata")


def test_rows_for_ac_step_is_deterministic(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap",
            directory=directory,
            nodes=click_nodes(),
        )
    )
    marked = (MARKED_IMAGE, image_digest(write_png(directory / "step_000_marked.png")))
    first = parse_ac_step(record, 0, directory=directory)
    second = parse_ac_step(record, 0, directory=directory)
    assert first == second
    assert json.dumps(rows_for_ac_step(first, RAW_IMAGE, marked), sort_keys=True) == json.dumps(
        rows_for_ac_step(second, RAW_IMAGE, marked), sort_keys=True
    )


def test_ac_rows_pass_validate_rows(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(
        ac_step(
            0,
            {"action_type": "click", "x": 75, "y": 75},
            "Tap",
            directory=directory,
            nodes=click_nodes(),
        ),
        ac_step(1, {"action_type": "scroll", "direction": "up"}, "Scroll up", directory=directory),
        ac_step(2, {"action_type": "navigate_home"}, "Go home", directory=directory),
        ac_step(3, None, None, directory=directory),
    )
    marked = (MARKED_IMAGE, image_digest(write_png(directory / "step_000_marked.png")))
    rows = rows_for_ac_step(parse_ac_step(record, 0, directory=directory), RAW_IMAGE, marked)
    for index, action in ((1, "swipe"), (2, "system_button"), (3, "terminate")):
        step = parse_ac_step(record, index, directory=directory)
        assert step.action == action
        rows += rows_for_ac_step(step, f"episode/step_{index:03d}_screenshot.png")
    assert len(rows) == 11
    validate_rows(rows, root=tmp_path)


def test_datasets_cover_the_five_families():
    assert DATASETS == {
        "action": "gui_action",
        "complete": "gui_complete",
        "element": "screenshot_choice",
        "swipe_dir": "gui_swipe",
        "button": "gui_button",
    }
    assert ELEMENT_INSTRUCTION == "Which action should be taken next to complete the user's task?"


# --- Task 4: episode discovery, image storage, isolation, and the CLI ---------


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
ROOT = Path(__file__).parents[1]
ORIGINAL_PROCESS_EPISODE = prepare.process_episode

# Set by the parallel test below: a file every worker appends its pid to.
WORKER_PIDS: Path | None = None

# Set by the interrupt test below: the episode whose worker refuses to run.
INTERRUPT_EPISODE: str | None = None


def recording_process_episode(episode_dir, **kwargs):
    """`prepare.process_episode` with a pid record and a pause.

    Module level on purpose: the parallel pool pickles the worker by reference,
    and a function defined inside a test cannot be pickled at all.
    """
    if WORKER_PIDS is not None:
        with WORKER_PIDS.open("a") as stream:
            stream.write(f"{os.getpid()}\n")
    time.sleep(0.2)
    return ORIGINAL_PROCESS_EPISODE(episode_dir, **kwargs)


def interrupting_process_episode(episode_dir, **kwargs):
    """`prepare.process_episode` that a Ctrl-C hits on one named episode."""
    if INTERRUPT_EPISODE is not None and Path(episode_dir).name == INTERRUPT_EPISODE:
        raise KeyboardInterrupt
    return ORIGINAL_PROCESS_EPISODE(episode_dir, **kwargs)


def cli_step(directory, index, action, instruction, *, color, nodes=None):
    """One step entry whose screenshot bytes are unique (`color` sets them).

    `ac_step` writes the file the metadata entry points at; rewriting it with a
    colour no other step uses keeps two episodes from sharing a screenshot, which
    the cross-split isolation rule would otherwise have to resolve.
    """
    entry = ac_step(index, action, instruction, directory=directory, nodes=nodes)
    write_png(directory / f"step_{index:03d}_screenshot.png", color=color)
    return entry


def click_step(directory, index, color, *, nodes=None, x=75, y=75):
    nodes = click_nodes() if nodes is None else nodes
    return cli_step(
        directory,
        index,
        {"action_type": "click", "x": x, "y": y},
        "Tap JOIN A MEETING",
        color=color,
        nodes=nodes,
    )


def write_episode(source: Path, episode_id: int, *entries, goal=GOAL) -> Path:
    """Create `<source>/<episode_id>` and the metadata file naming its steps."""
    directory = source / str(episode_id)
    directory.mkdir(parents=True, exist_ok=True)
    record = ac_episode(*entries, goal=goal, episode_id=episode_id)
    (directory / f"metadata_{episode_id}.json").write_text(json.dumps(record))
    return directory


def stored_rows(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for split in prepare.SPLITS
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]


def sample_corpus(tmp_path: Path) -> Path:
    """Three episodes: every family, one blank candidate, one click per split."""
    source = tmp_path / "corpus"
    first = source / "0"
    first.mkdir(parents=True)
    write_episode(
        source,
        0,
        click_step(first, 0, (10, 20, 30)),
        cli_step(
            first,
            1,
            {"action_type": "scroll", "direction": "down"},
            "Scroll down",
            color=(10, 20, 31),
        ),
        cli_step(first, 2, {"action_type": "navigate_back"}, "Press back", color=(10, 20, 32)),
        cli_step(first, 3, None, None, color=(10, 20, 33)),
    )
    second = source / "1"
    second.mkdir(parents=True)
    write_episode(
        source,
        1,
        cli_step(
            second, 0, {"action_type": "scroll", "direction": "up"}, "Scroll up", color=(11, 20, 30)
        ),
        cli_step(second, 1, None, None, color=(11, 20, 31)),
    )
    # The second candidate carries neither text nor a description, so its payload
    # is the empty object the manifest counts.
    blank = [
        click_nodes()[0],
        node(boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100}),
    ]
    third = source / "10"
    third.mkdir(parents=True)
    write_episode(
        source,
        10,
        click_step(third, 0, (12, 20, 30), nodes=blank),
        cli_step(third, 1, None, None, color=(12, 20, 31)),
    )
    return source


def test_cli_writes_the_splits_and_a_manifest(tmp_path):
    source = sample_corpus(tmp_path)
    output = tmp_path / "out"
    returned = prepare.convert(source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    # The returned manifest and the published one differ only in the shapes JSON
    # has no tuple for.
    assert json.dumps(returned, sort_keys=True) == json.dumps(manifest, sort_keys=True)
    assert returned["split_limits"] == [("calibration", 10), ("dev", 20), ("test", 30)]
    assert manifest["schema_version"] == 1
    assert manifest["split_seed"] == SPLIT_SEED
    assert manifest["split_limits"] == [["calibration", 10], ["dev", 20], ["test", 30]]
    assert manifest["source"]["episodes"] == 3
    assert len(manifest["source"]["metadata_sha256"]) == 64
    assert manifest["source"]["metadata_files_hashed"] == 3
    assert manifest["path_convention"].startswith("repository-root relative")
    assert manifest["token_check"] == "skipped"
    assert manifest["exclusions"] == {}
    assert (output / "excluded.jsonl").read_text() == ""
    assert set(manifest["sha256"]) == set(prepare.SPLITS)
    for split in prepare.SPLITS:
        path = output / f"{split}.jsonl"
        assert path.exists()
        assert manifest["sha256"][split] == prepare.digest_file(path)
    assert (output / "dev.jsonl").read_text() == ""
    counts = {(entry["dataset"], entry["split"]): entry["n"] for entry in manifest["counts"]}
    assert counts == {
        ("gui_action", "calibration"): 2,
        ("gui_action", "test"): 2,
        ("gui_action", "train"): 4,
        ("gui_button", "train"): 1,
        ("gui_complete", "calibration"): 2,
        ("gui_complete", "test"): 2,
        ("gui_complete", "train"): 4,
        ("gui_swipe", "calibration"): 1,
        ("gui_swipe", "train"): 1,
        ("screenshot_choice", "test"): 1,
        ("screenshot_choice", "train"): 1,
    }
    assert [(entry["dataset"], entry["split"]) for entry in manifest["counts"]] == sorted(
        (entry["dataset"], entry["split"]) for entry in manifest["counts"]
    )
    assert manifest["action_classes"] == {
        "calibration": {"swipe": 1, "terminate": 1},
        "test": {"click": 1, "terminate": 1},
        "train": {"click": 1, "system_button": 1, "swipe": 1, "terminate": 1},
    }
    assert manifest["element_stats"]["train"] == {
        "basis": "post_isolation",
        "rows": 1,
        "candidates_min": 2,
        "candidates_mean": 2.0,
        "candidates_max": 2,
        "soft_targets": 0,
        "multi_hit_rate": 0.0,
        "max_hits": 1,
        "empty_target_payloads": 0,
        "empty_target_payload_rate": 0.0,
    }
    assert manifest["element_stats"]["test"] == {
        "basis": "post_isolation",
        "rows": 1,
        "candidates_min": 2,
        "candidates_mean": 2.0,
        "candidates_max": 2,
        "soft_targets": 0,
        "multi_hit_rate": 0.0,
        "max_hits": 1,
        "empty_target_payloads": 1,
        "empty_target_payload_rate": 1.0,
    }
    assert manifest["element_resolution"] == {
        "basis": "pre_isolation",
        "element_rows": 2,
        "no_target_element": 0,
        "too_few_candidates": 0,
        "too_many_candidates": 0,
        "hit_rate": 1.0,
    }
    assert "uniform" in manifest["dataset_weighting"]
    assert manifest["vocabularies"]["ac_actions"] == AC_ACTIONS
    assert manifest["vocabularies"]["buttons"] == BUTTONS
    assert manifest["vocabularies"]["swipe_directions"] == SWIPE_DIRECTIONS
    assert manifest["vocabularies"]["instructions"] == {
        **INSTRUCTIONS,
        "element": ELEMENT_INSTRUCTION,
    }
    assert manifest["vocabularies"]["complete_criteria"] == COMPLETE_CRITERIA
    assert manifest["vocabularies"]["element_rule"]
    assert manifest["vocabularies"]["marked_images"]
    assert manifest["environment"]["python"]
    assert manifest["environment"]["pillow"]
    assert len(manifest["images"]) == 10
    assert all(name.endswith(".png") for name in manifest["images"])
    digest = hashlib.sha256()
    for episode_id in (0, 1, 10):
        digest.update((source / str(episode_id) / f"metadata_{episode_id}.json").read_bytes())
    assert manifest["source"]["metadata_sha256"] == digest.hexdigest()
    rows = stored_rows(output)
    assert len(rows) == 21
    validate_rows(rows, root=ROOT)
    for row in rows:
        with Image.open(ROOT / row["image"]) as image:
            assert image.convert("RGB").size == SCREENSHOT
    # Every row of a step shares the group and the split `split_for` assigned.
    assert {row["split"] for row in rows if row["group"] == "task:android_control_0"} == {"train"}


def test_cli_pins_marked_and_raw_screenshots_to_their_names(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source,
        0,
        click_step(directory, 0, (9, 9, 9)),
        cli_step(directory, 1, None, None, color=(9, 9, 10)),
    )
    output = tmp_path / "out"
    prepare.convert(source, output)
    rows = stored_rows(output)
    element = next(row for row in rows if row["dataset"] == "screenshot_choice")
    action = next(row for row in rows if row["id"] == "android_control_0_step0:action")
    marked = ROOT / element["image"]
    raw = ROOT / action["image"]
    # The element row asks about the marked copy; the raw copy is what every
    # other row of the same step shows, and both names are their own digests.
    assert element["image"] != action["image"]
    marked_sha = hashlib.sha256(marked.read_bytes()).hexdigest()
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert marked.name == marked_sha + ".png"
    assert raw.name == raw_sha + ".png"
    assert element["aliases"] == ["image-bytes:" + raw_sha, "image-bytes:" + marked_sha]
    # The stored raw copy is byte-identical to the source screenshot, so the
    # decode `validate_rows` performs is of the very bytes the digest names.
    assert raw.read_bytes() == (directory / "step_000_screenshot.png").read_bytes()
    with Image.open(marked) as image:
        assert image.size == SCREENSHOT


def test_cli_is_deterministic(tmp_path):
    source = sample_corpus(tmp_path)
    output = tmp_path / "out"
    first = prepare.convert(source, output)
    second = prepare.convert(source, output)
    assert second == first
    assert second["sha256"] == first["sha256"]
    assert second["images"] == first["images"]


def read_output(output: Path) -> dict:
    """Every published file of one conversion, as bytes."""
    names = [f"{split}.jsonl" for split in prepare.SPLITS]
    return {
        name: (output / name).read_bytes() for name in (*names, "excluded.jsonl", "manifest.json")
    }


def parallel_corpus(tmp_path: Path) -> Path:
    """Eight episodes of three steps, each step with its own screenshot bytes.

    Big enough that a four-worker pool hands several episodes to each worker,
    and the first episode clicks into nested boxes so the soft targets of a
    multi-hit step go through the whole pipeline.
    """
    source = tmp_path / "corpus"
    for episode_id in range(8):
        directory = source / str(episode_id)
        directory.mkdir(parents=True)
        nodes = None
        if episode_id == 0:
            nodes = [
                node(
                    text="Card", boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 100}
                ),
                node(
                    text="Button",
                    boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100},
                ),
            ]
        write_episode(
            source,
            episode_id,
            click_step(directory, 0, (episode_id, 0, 0), nodes=nodes),
            cli_step(directory, 1, {"action_type": "wait"}, "Wait", color=(episode_id, 0, 1)),
            cli_step(directory, 2, None, None, color=(episode_id, 0, 2)),
        )
    return source


def test_cli_parallel_workers_write_the_serial_output(tmp_path):
    source = parallel_corpus(tmp_path)
    output = tmp_path / "out"
    serial = prepare.convert(source, output, workers=1)
    serial_files = read_output(output)
    parallel = prepare.convert(source, output, workers=4)
    # Identical bytes, not merely equivalent rows: the merge is ordered, so the
    # workers cannot reorder a split file, the manifest, or the exclusions.
    assert read_output(output) == serial_files
    assert parallel == serial
    assert parallel["sha256"] == serial["sha256"]
    assert parallel["images"] == serial["images"]
    assert "uniform" in parallel["dataset_weighting"]
    # Again, so an ordering bug that scheduling happens to hide cannot survive.
    prepare.convert(source, output, workers=4)
    assert read_output(output) == serial_files


def test_cli_parallel_workers_share_the_episodes(tmp_path, monkeypatch):
    global WORKER_PIDS
    source = parallel_corpus(tmp_path)
    WORKER_PIDS = tmp_path / "worker-pids.txt"
    monkeypatch.setattr(prepare, "process_episode", recording_process_episode)
    try:
        prepare.convert(source, tmp_path / "out", workers=4)
    finally:
        WORKER_PIDS = None
    # More than one process really did the work: the pool is not a serial loop
    # wearing a worker count.
    assert len(set((tmp_path / "worker-pids.txt").read_text().split())) > 1


def test_cli_reports_an_interrupt_instead_of_a_traceback(tmp_path, monkeypatch):
    global INTERRUPT_EPISODE
    source = parallel_corpus(tmp_path)
    INTERRUPT_EPISODE = "1"
    monkeypatch.setattr(prepare, "process_episode", interrupting_process_episode)
    try:
        # The pool path funnels through the same handler: the interrupt surfaces
        # while the parent waits for a result, and one line explains the stop.
        with pytest.raises(SystemExit, match="Interrupted after 1 of 8 episodes"):
            prepare.convert(source, tmp_path / "out", workers=1)
    finally:
        INTERRUPT_EPISODE = None


def test_cli_sweeps_temporaries_left_by_a_killed_run(tmp_path):
    source = sample_corpus(tmp_path)
    output = tmp_path / "out"
    images = output / "images"
    images.mkdir(parents=True)
    stale = images / ".deadbeef.png.1234.0.tmp"
    stale.write_bytes(b"half a png")
    manifest = prepare.convert(source, output)
    # Nothing else would ever delete it, and a leftover under images/ breaks the
    # directory-equals-manifest property for an operator re-running into it.
    assert not stale.exists()
    assert sorted(path.name for path in images.iterdir()) == sorted(manifest["images"])


def test_cli_records_exclusions_without_aborting_the_batch(tmp_path):
    source = tmp_path / "corpus"
    (source / "0").mkdir(parents=True)  # no metadata_0.json at all
    second = source / "1"
    second.mkdir(parents=True)
    write_episode(
        source,
        1,
        click_step(second, 0, (2, 0, 0), x=5, y=190),  # taps where nothing lies
        cli_step(second, 1, None, None, color=(2, 0, 1)),
    )
    third = source / "2"
    third.mkdir(parents=True)
    entry = cli_step(third, 0, {"action_type": "wait"}, "Wait", color=(3, 0, 0))
    (third / "step_000_screenshot.png").unlink()
    write_episode(source, 2, entry)
    # Episode 3 has a metadata file that parses as JSON but not as an episode,
    # and episode 4 has one that was cut off mid-write.
    fourth = source / "3"
    fourth.mkdir(parents=True)
    (fourth / "metadata_3.json").write_text(json.dumps({"episode_id": 3}))
    fifth = source / "4"
    fifth.mkdir(parents=True)
    (fifth / "metadata_4.json").write_text("{torn")
    output = tmp_path / "out"
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {
        "parse:missing_image": 1,
        "parse:no_target_element": 1,
        "parse:unparsable_metadata": 3,
    }
    entries = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    assert [(entry["id"], entry["reason"], entry["stage"]) for entry in entries] == [
        ("0", "unparsable_metadata", "parse"),
        ("android_control_1_step0", "no_target_element", "parse"),
        ("android_control_2_step0", "missing_image", "parse"),
        ("3", "unparsable_metadata", "parse"),
        ("4", "unparsable_metadata", "parse"),
    ]
    # An unreadable metadata file names the error it hit, a structurally wrong
    # document names the field that failed, and a step exclusion carries no
    # detail of its own.
    assert "metadata_0.json" in entries[0]["detail"]
    assert entries[1]["detail"] == ""
    assert entries[2]["detail"] == ""
    assert entries[3]["detail"] == "key 'goal' is not a string"
    assert entries[4]["detail"].startswith("JSONDecodeError")
    assert sum(entry["n"] for entry in manifest["counts"]) == 2
    # The failure paths store nothing they cannot reference.
    assert sorted(path.name for path in (output / "images").iterdir()) == sorted(manifest["images"])


def test_cli_writes_no_image_for_a_step_that_fails_late(tmp_path, monkeypatch):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source,
        0,
        click_step(directory, 0, (27, 0, 0)),
        cli_step(directory, 1, {"action_type": "wait"}, "Wait", color=(27, 0, 1)),
    )
    output = tmp_path / "out"

    def explode(step, image_path, marked=None):
        raise RuntimeError("boom")

    # The failure lands after the screenshots would have been stored: the step is
    # excluded and neither file may survive on disk, where no row could name it
    # and a rerun could not tell it apart from a referenced one.
    monkeypatch.setattr(prepare, "rows_for_ac_step", explode)
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:unexpected": 2}
    assert manifest["images"] == []
    assert list((output / "images").glob("*.png")) == []
    assert (output / "train.jsonl").read_text() == ""


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def corrupted_png(size=SCREENSHOT) -> bytes:
    """A PNG whose container is valid and whose pixels cannot be decoded.

    The chunk CRCs are correct and the IDAT is a valid zlib stream, so the
    container walk `Image.verify()` performs passes; every scanline starts with
    a filter type PNG does not define, so decoding the pixels fails. This is the
    shape the header-only check used to accept and then abort a whole run over.
    """
    width, height = size
    stride = 1 + width * 3
    raw = bytearray(stride * height)  # black pixels, filter byte 99 per row
    for row in range(height):
        raw[row * stride] = 99
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + png_chunk(b"IEND", b"")
    )


def test_cli_excludes_a_screenshot_whose_pixels_do_not_decode(tmp_path):
    source = tmp_path / "corpus"
    broken = source / "0"
    broken.mkdir(parents=True)
    entry = cli_step(broken, 0, {"action_type": "wait"}, "Wait", color=(31, 0, 0))
    screenshot = broken / "step_000_screenshot.png"
    screenshot.write_bytes(corrupted_png())
    with Image.open(screenshot) as handle:
        handle.verify()  # the container checks pass; only the pixels are broken
    write_episode(source, 0, entry)
    healthy = source / "1"
    healthy.mkdir(parents=True)
    write_episode(
        source,
        1,
        cli_step(healthy, 0, {"action_type": "wait"}, "Wait", color=(31, 0, 1)),
    )
    output = tmp_path / "out"
    # The step is refused at parse time, so the run finishes and publishes its
    # splits instead of aborting the whole batch at the very end.
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:missing_image": 1}
    # The refused step stores nothing; the healthy episode is untouched.
    assert len(manifest["images"]) == 1
    assert sorted(path.name for path in (output / "images").iterdir()) == manifest["images"]
    assert [row["id"] for row in stored_rows(output)] == [
        "android_control_1_step0:action",
        "android_control_1_step0:complete",
    ]
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert entry["id"] == "android_control_0_step0"
    assert entry["reason"] == "missing_image"
    for split in prepare.SPLITS:
        assert manifest["sha256"][split] == prepare.digest_file(output / f"{split}.jsonl")


def test_cli_names_the_step_and_field_that_breaks_the_metadata(tmp_path):
    source = tmp_path / "corpus"
    healthy = source / "0"
    healthy.mkdir(parents=True)
    write_episode(
        source, 0, cli_step(healthy, 0, {"action_type": "wait"}, "Wait", color=(30, 0, 0))
    )
    directory = source / "1"
    directory.mkdir(parents=True)
    kept = cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(30, 0, 1))
    broken = cli_step(directory, 1, {"action_type": "wait"}, "Wait", color=(30, 0, 2))
    del broken["action"]
    write_episode(source, 1, kept, broken)
    output = tmp_path / "out"
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:unparsable_metadata": 1}
    entry = json.loads((output / "excluded.jsonl").read_text())
    assert entry == {
        "id": "1",
        "reason": "unparsable_metadata",
        "detail": "step 1: missing key 'action'",
        "stage": "parse",
    }
    # The document is refused whole -- its other steps go with it -- while the
    # healthy episode converts as usual.
    assert [row["id"] for row in stored_rows(output)] == [
        "android_control_0_step0:action",
        "android_control_0_step0:complete",
    ]


def test_cli_manifest_counts_the_soft_element_targets(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    nested = [
        node(text="Card", boundsInScreen={"left": 0, "top": 0, "right": 100, "bottom": 100}),
        node(text="Button", boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100}),
    ]
    write_episode(
        source,
        0,
        click_step(directory, 0, (40, 0, 0), nodes=nested),  # (75, 75) is in both boxes
        click_step(directory, 1, (40, 0, 1)),  # click_nodes: only one box holds it
        cli_step(directory, 2, None, None, color=(40, 0, 2)),
    )
    output = tmp_path / "out"
    manifest = prepare.convert(source, output)
    stats = manifest["element_stats"][split_for("task:android_control_0")]
    assert stats["rows"] == 2
    assert stats["soft_targets"] == 1
    assert stats["multi_hit_rate"] == 0.5
    assert stats["max_hits"] == 2
    # The rows say the same thing the summary does.
    hits = sorted(
        sum(1 for weight in row["target"] if weight > 0)
        for row in stored_rows(output)
        if row["dataset"] == "screenshot_choice"
    )
    assert hits == [1, 2]


def test_cli_keeps_the_metadata_digest_of_an_episode_that_fails_late(tmp_path, monkeypatch):
    source = tmp_path / "corpus"
    for episode_id in (0, 1):
        directory = source / str(episode_id)
        directory.mkdir(parents=True)
        write_episode(
            source,
            episode_id,
            cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(42, 0, episode_id)),
        )
    output = tmp_path / "out"

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    # The metadata is read before the episode body runs, so a failure inside the
    # body leaves the digest of the metadata this run read exactly as the serial
    # loop counted it.
    monkeypatch.setattr(prepare, "convert_episode", explode)
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:unexpected": 2}
    assert manifest["source"]["metadata_files_hashed"] == 2
    digest = hashlib.sha256()
    for episode_id in (0, 1):
        digest.update((source / str(episode_id) / f"metadata_{episode_id}.json").read_bytes())
    assert manifest["source"]["metadata_sha256"] == digest.hexdigest()


def test_cli_excludes_metadata_nested_past_the_recursion_limit(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    (directory / "metadata_0.json").write_text("[" * 200_000)
    output = tmp_path / "out"
    manifest = prepare.convert(source, output)
    # A RecursionError escaping the reader would kill a pool worker and hang the
    # parent; the episode is excluded with the reason instead.
    assert manifest["exclusions"] == {"parse:unparsable_metadata": 1}
    entry = json.loads((output / "excluded.jsonl").read_text())
    assert entry["reason"] == "unparsable_metadata"
    assert entry["detail"].startswith("RecursionError")


def test_cli_excludes_an_episode_whose_two_ids_disagree(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "5"
    directory.mkdir(parents=True)
    entry = cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(13, 13, 13))
    # The rows would be keyed `android_control_6_*` while the metadata file is
    # named after directory 5, so the episode is excluded rather than keyed by
    # an id nobody can predict.
    (directory / "metadata_5.json").write_text(json.dumps(ac_episode(entry, episode_id=6)))
    output = tmp_path / "out"
    manifest = prepare.convert(source, output)
    assert manifest["exclusions"] == {"parse:unparsable_metadata": 1}
    assert manifest["source"] == {
        "input": str(source),
        "episodes": 1,
        "metadata_sha256": hashlib.sha256((directory / "metadata_5.json").read_bytes()).hexdigest(),
        "metadata_files_hashed": 1,
    }
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert entry == {
        "id": "5",
        "reason": "unparsable_metadata",
        "detail": "episode_id mismatch: dir 5 vs record 6",
        "stage": "parse",
    }
    assert sum(item["n"] for item in manifest["counts"]) == 0


def test_cli_refuses_to_run_outside_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT / "scripts")
    with pytest.raises(SystemExit, match="Run from the repository root"):
        prepare.convert(Path("."), tmp_path / "out")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Run from the dohnuts repository root"):
        prepare.convert(tmp_path, tmp_path / "out")


def test_cli_rejects_inputs_without_episode_directories(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    with pytest.raises(SystemExit, match="No episode directories found"):
        prepare.convert(source, tmp_path / "out")
    with pytest.raises(SystemExit, match="No episode directories found"):
        prepare.convert(tmp_path / "missing", tmp_path / "out")
    # Only all-digit directory names are episodes: files and other directories
    # are not part of the corpus.
    (source / "notes.txt").write_text("not an episode")
    (source / "12").write_text("a file, not a directory")
    (source / "episode_5").mkdir()
    with pytest.raises(SystemExit, match="No episode directories found"):
        prepare.convert(source, tmp_path / "out")


def test_cli_warns_on_empty_splits(tmp_path, capsys):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source, 0, cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(5, 5, 5))
    )
    prepare.convert(source, tmp_path / "out")
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"warning": "empty splits: dev, calibration, test"}
    assert "warning" not in captured.out


def test_cli_reports_progress_on_a_long_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(prepare, "PROGRESS_EVERY", 1)
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    broken = cli_step(directory, 2, {"action_type": "wait"}, "Wait", color=(6, 6, 8))
    (directory / "step_002_screenshot.png").unlink()
    write_episode(
        source,
        0,
        cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(6, 6, 6)),
        cli_step(directory, 1, None, None, color=(6, 6, 7)),
        broken,
    )
    prepare.convert(source, tmp_path / "out")
    captured = capsys.readouterr()
    assert captured.err.splitlines()[0] == json.dumps(
        {"progress": {"episodes": 1, "rows": 4, "exclusions": 1}}
    )


# --- Task 5: the token budget check ------------------------------------------


class StubImageProcessor:
    patch_size = 14
    merge_size = 2


class StubTokenizer:
    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text.split())))}


class StubProcessor:
    image_processor = StubImageProcessor()
    tokenizer = StubTokenizer()


def token_heavy_step(directory, index, *, color):
    """A click step whose first candidate carries a whole document as its text.

    The a11y `text` of a real Android Control node can hold an entire PDF, which
    is what pushes a step past the budget the training collator enforces.
    """
    nodes = [
        node(
            text=" ".join(["filler"] * 4000),
            boundsInScreen={"left": 0, "top": 0, "right": 50, "bottom": 50},
        ),
        node(
            text="JOIN A MEETING",
            boundsInScreen={"left": 50, "top": 50, "right": 100, "bottom": 100},
        ),
    ]
    return click_step(directory, index, color, nodes=nodes)


def test_cli_excludes_over_budget_steps_whole(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source,
        0,
        token_heavy_step(directory, 0, color=(21, 0, 0)),
        cli_step(directory, 1, {"action_type": "wait"}, "Wait", color=(21, 0, 1)),
    )
    output = tmp_path / "out"
    manifest = prepare.convert(source, output, processor=StubProcessor())
    assert manifest["token_check"] == "enabled"
    assert manifest["exclusions"] == {"parse:token_budget": 1}
    entry = json.loads((output / "excluded.jsonl").read_text().splitlines()[0])
    assert {key: entry[key] for key in entry if key != "detail"} == {
        "id": "android_control_0_step0",
        "reason": "token_budget",
        "stage": "parse",
    }
    assert int(entry["detail"]) > MAX_LENGTH
    # The excluded step mints no rows and stores no images: the budget is checked
    # before anything is written, and a step is never partially converted.
    assert [row["id"] for row in stored_rows(output)] == [
        "android_control_0_step1:action",
        "android_control_0_step1:complete",
    ]
    assert len(manifest["images"]) == 1
    # The manifest and the directory agree: an image nothing references must not
    # survive on disk either, or every rerun leaves the corpus' slack behind.
    assert sorted(path.name for path in (output / "images").iterdir()) == sorted(manifest["images"])
    assert manifest["element_stats"] == {}
    assert manifest["element_resolution"] == {
        "basis": "pre_isolation",
        "element_rows": 0,
        "no_target_element": 0,
        "too_few_candidates": 0,
        "too_many_candidates": 0,
        "hit_rate": None,
    }


def test_cli_checks_the_budget_against_the_final_rows(tmp_path):
    # The probe rows are the very rows the step would mint (the duplicate alias
    # of the dummy marked copy never leaves the probe), so the reported length is
    # the element row's own.
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    entry = token_heavy_step(directory, 0, color=(26, 0, 0))
    write_episode(source, 0, entry)
    step = parse_ac_step(ac_episode(entry, episode_id=0), 0, directory=directory)
    rows = rows_for_ac_step(step, str(step.image), (str(step.image), step.image_sha256))
    lengths = [prepare.token_length(StubProcessor(), row) for row in rows]
    assert max(lengths) > MAX_LENGTH
    output = tmp_path / "out"
    manifest = prepare.convert(source, output, processor=StubProcessor())
    assert manifest["exclusions"] == {"parse:token_budget": 1}
    entry = json.loads((output / "excluded.jsonl").read_text())
    assert entry["detail"] == str(max(lengths))


def test_cli_parallel_workers_keep_the_token_check(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source,
        0,
        token_heavy_step(directory, 0, color=(32, 0, 0)),
        cli_step(directory, 1, {"action_type": "wait"}, "Wait", color=(32, 0, 1)),
    )
    second = source / "1"
    second.mkdir(parents=True)
    write_episode(source, 1, cli_step(second, 0, {"action_type": "wait"}, "Wait", color=(32, 1, 0)))
    output = tmp_path / "out"
    serial = prepare.convert(source, output, processor=StubProcessor())
    serial_files = read_output(output)
    assert serial["exclusions"] == {"parse:token_budget": 1}
    # The workers never see the processor itself: they read the module global
    # the fork inherited, and reach the same verdict on the same step.
    parallel = prepare.convert(source, output, processor=StubProcessor(), workers=4)
    assert read_output(output) == serial_files
    assert parallel == serial
    assert parallel["token_check"] == "enabled"


def test_cli_skips_the_model_when_token_checks_are_disabled(tmp_path, monkeypatch):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source, 0, cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(22, 0, 0))
    )
    output = tmp_path / "out"

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


def test_cli_reports_an_unloadable_token_check_model(tmp_path):
    source = tmp_path / "corpus"
    directory = source / "0"
    directory.mkdir(parents=True)
    write_episode(
        source, 0, cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(23, 0, 0))
    )
    with pytest.raises(SystemExit, match="Cannot load the token-check model"):
        prepare.main(
            [
                "--input",
                str(source),
                "--output",
                str(tmp_path / "out"),
                "--model",
                "/nonexistent",
            ]
        )


def test_token_length_counts_words_and_image_patches(tmp_path):
    directory = tmp_path / "episode"
    record = ac_episode(cli_step(directory, 0, {"action_type": "wait"}, "Wait", color=(24, 0, 0)))
    step = parse_ac_step(record, 0, directory=directory)
    row = rows_for_ac_step(step, str(step.image))[0]
    prompt, _, _ = prepare.render_question(
        prepare.render(row["state"]), row["question"], has_image=True
    )
    # The 100x200 screenshot resizes to 728x364: 26x13 = 338 patches at the stub
    # factor of 14*2, minus the placeholder token already in the prompt. The real
    # processor's factor is 16*2 = 32, which resizes it to 736x384 instead.
    assert prepare.token_length(StubProcessor(), row) == len(prompt.split()) + 338 - 1


def test_token_length_matches_the_training_collator(tmp_path):
    model = ROOT / "Qwen" / "Qwen3.5-0.8B"
    if not model.is_dir():
        pytest.skip("local Qwen3.5-0.8B snapshot is not available")
    from dohnuts.training_data import DecisionCollator

    directory = tmp_path / "episode"
    record = ac_episode(click_step(directory, 0, (25, 0, 0)))
    step = parse_ac_step(record, 0, directory=directory)
    rows = rows_for_ac_step(step, str(step.image), (str(step.image), step.image_sha256))
    processor = prepare.AutoProcessor.from_pretrained(model, local_files_only=True)
    collated, *_ = DecisionCollator(model)([rows[-1]])
    assert prepare.token_length(processor, rows[-1]) == int(collated["input_ids"].shape[1])
