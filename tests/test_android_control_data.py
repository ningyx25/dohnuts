"""Android Control step parsing, action mapping, and element extraction rules."""

import json

from dohnuts.android_control_data import (
    AC_ACTIONS,
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    SCROLL_TO_SWIPE_DIRECTION,
    element_description,
    extract_elements,
    hit_test,
    map_action,
    parse_metadata,
    resolve_element_choice,
    validate_element,
)
from dohnuts.gui_data import ACTIONS

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


def test_hit_test_prefers_the_smallest_containing_element():
    elements = [
        element(0, (0, 0, 100, 100)),
        element(1, (10, 10, 40, 40)),
        element(2, (0, 0, 200, 200)),
    ]
    assert hit_test(elements, 20, 20) == 1
    assert hit_test(elements, 150, 150) == 2
    assert hit_test(elements, 95, 95) == 0


def test_hit_test_ties_go_to_the_earlier_position():
    elements = [element(0, (0, 0, 50, 50)), element(1, (0, 0, 50, 50))]
    assert hit_test(elements, 5, 5) == 0
    assert hit_test(list(reversed(elements)), 5, 5) == 0


def test_hit_test_reports_the_position_not_the_index_field():
    # Only the position is returned, so a caller can index the list it passed
    # in even if the `index` fields disagree with the positions.
    elements = [element(7, (0, 0, 50, 50)), element(9, (0, 0, 200, 200))]
    assert hit_test(elements, 5, 5) == 0
    assert hit_test(elements, 150, 150) == 1


def test_hit_test_uses_closed_bounds():
    elements = [element(0, (10, 10, 20, 20))]
    assert hit_test(elements, 10, 10) == 0
    assert hit_test(elements, 20, 20) == 0
    assert hit_test(elements, 15, 15) == 0
    assert hit_test(elements, 9.9, 15) is None
    assert hit_test(elements, 15, 20.1) is None


def test_hit_test_reports_misses_and_tolerates_malformed_input():
    assert hit_test([], 0, 0) is None
    assert hit_test([element(0, (0, 0, 10, 10))], 50, 50) is None
    assert hit_test([element(0, (0, 0, 10, 10))], float("nan"), 5) is None
    assert hit_test([element(0, (0, 0, 10, 10))], "5", 5) is None
    assert hit_test([element(0, (0, 0, 10, 10))], 5, None) is None
    unusable = [
        {"index": 0, "bounds": None},
        {"index": 1},
        {"index": 2, "bounds": [1, 2, 3]},
        {"index": 3, "bounds": [0, 0, "10", 10]},
        "junk",
        element(4, (0, 0, 100, 100)),
    ]
    assert hit_test(unusable, 5, 5) == 5  # the position of the only usable element


def test_hit_test_reports_nothing_for_a_non_list():
    assert hit_test(None, 5, 5) is None
    assert hit_test(7, 5, 5) is None
    assert hit_test("elements", 5, 5) is None
    assert hit_test({"0": element(0, (0, 0, 10, 10))}, 5, 5) is None


def test_resolve_element_choice_enforces_the_candidate_count():
    assert (MIN_CANDIDATES, MAX_CANDIDATES) == (2, 128)
    assert resolve_element_choice([], 0, 0) == (None, "too_few_candidates")
    assert resolve_element_choice([element(0, (0, 0, 10, 10))], 5, 5) == (
        None,
        "too_few_candidates",
    )
    too_many = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES + 1)]
    assert resolve_element_choice(too_many, 5, 5) == (None, "too_many_candidates")


def test_resolve_element_choice_checks_the_count_before_the_hit_test():
    too_many = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES + 1)]
    assert resolve_element_choice(too_many, 5000, 5000) == (None, "too_many_candidates")
    full = [element(index, (0, 0, 10, 10)) for index in range(MAX_CANDIDATES)]
    assert resolve_element_choice(full, 5000, 5000) == (None, "no_target_element")


def test_resolve_element_choice_resolves_a_hit():
    pair = [element(0, (0, 0, 10, 10)), element(1, (100, 100, 200, 200))]
    assert resolve_element_choice(pair, 5, 5) == (0, None)
    assert resolve_element_choice(pair, 150, 150) == (1, None)
    assert resolve_element_choice(pair, 50, 50) == (None, "no_target_element")
    # The result is a position in the list that was passed in, not a fixed id.
    assert resolve_element_choice(list(reversed(pair)), 150, 150) == (0, None)
    assert resolve_element_choice(list(reversed(pair)), 5, 5) == (1, None)


def test_resolve_element_choice_accepts_the_full_candidate_list():
    elements = [element(index, (index * 10, 0, index * 10 + 8, 8)) for index in range(128)]
    assert resolve_element_choice(elements, 1005, 4) == (100, None)


def test_resolve_element_choice_reports_a_non_list_as_too_few():
    assert resolve_element_choice(None, 5, 5) == (None, "too_few_candidates")
    assert resolve_element_choice(7, 5, 5) == (None, "too_few_candidates")
    assert resolve_element_choice({}, 5, 5) == (None, "too_few_candidates")


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


def test_extract_hit_test_and_description_agree_at_the_seam():
    # The seam Tasks 2 and 3 rely on: the description label and the marked
    # element must be the same element the hit test selected.
    bounds = {"left": 100, "top": 200, "right": 300, "bottom": 260}
    a11y = forest(
        node(text="not clickable", isClickable=False, boundsInScreen=bounds),
        node(text="TARGET", boundsInScreen=bounds),
        node(text="elsewhere", boundsInScreen={"left": 0, "top": 0, "right": 10, "bottom": 10}),
    )
    elements = extract_elements(a11y, screen_size=SCREEN)
    position, reason = resolve_element_choice(elements, 200, 230)
    assert reason is None
    assert position == 0
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
