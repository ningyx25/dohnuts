"""The training shape: one SCROLL, a direction question, and sampled apps.

The port itself is pinned in `tests/test_mobile_jev_prompt.py`; what is checked
here is the three-part overlay this module applies on top of it, and that the
parts it does not touch are left alone.
"""

import random

from dohnuts import jev_training_prompt as training
from dohnuts import mobile_jev_prompt as jev

SCREEN = (1080, 2400)
GOAL = "Open the Clock app and check the alarm."


def element(id="0", bounds=(0, 0, 100, 100), **overrides):
    return jev.Element(id=id, bounds=tuple(bounds), **overrides)


def observation(elements, package="com.example", screen=SCREEN):
    return jev.make_observation(jev.phone_from(package, elements), screen, list(elements))


def inventory(count=40, prefix="App"):
    return [f"{prefix}{index:03d}" for index in range(count)]


def scroll_screen():
    return observation([
        element(id="0", text="Save", clickable=True, bounds=(0, 0, 100, 50)),
        element(id="1", text="List", scrollable=True, bounds=(0, 60, 1080, 2200)),
    ])


def test_the_four_scroll_operations_collapse_into_one():
    request = training.build_training_request(GOAL, scroll_screen())
    criteria = request.questions["operation"]["criteria"]
    assert "SCROLL" in criteria
    assert not [name for name in criteria if name.startswith("SCROLL_")]
    assert list(criteria) == [
        "TAP", "SCROLL", "BACK", "HOME", "WAIT", "DONE", "BLOCKED",
    ]
    assert criteria["SCROLL"] == training.SCROLL_DESCRIPTION


def test_scroll_keeps_the_position_the_first_direction_had():
    # OPEN_APP, TAP and TYPE_TEXT come before the scrolls in the agent's order,
    # so the merged operation lands right where SCROLL_DOWN used to be.
    request = training.build_training_request(
        GOAL,
        observation([
            element(id="0", text="Save", clickable=True),
            element(
                id="1",
                class_name="android.widget.EditText",
                editable=True,
                focused=True,
                bounds=(0, 120, 1080, 200),
            ),
            element(id="2", text="List", scrollable=True, bounds=(0, 220, 1080, 2200)),
        ]),
        apps=("Clock", "Notes"),
    )
    assert list(request.questions["operation"]["criteria"]) == [
        "OPEN_APP", "TAP", "TYPE_TEXT", "SCROLL", "BACK", "HOME", "ENTER", "WAIT",
        "DONE", "BLOCKED",
    ]


def test_element_operations_lose_their_directions():
    request = training.build_training_request(GOAL, scroll_screen())
    entries = {entry["index"]: entry for entry in request.space.elements}
    assert entries["2"]["operations"] == ["SCROLL"]
    # The state and the candidate space are the same objects, so both agree.
    assert request.state["elements"] is request.space.elements
    assert request.state["elements"][0]["operations"] == ["TAP"]


def test_a_clickable_region_lists_both_operations_once():
    request = training.build_training_request(
        GOAL,
        observation([
            element(id="0", text="List", clickable=True, scrollable=True,
                    bounds=(0, 0, 1080, 2200)),
        ]),
    )
    entry = request.space.elements[0]
    assert entry["operations"] == ["TAP", "SCROLL"]


def test_the_direction_question_is_added_with_the_agents_wording():
    request = training.build_training_request(GOAL, scroll_screen())
    question = request.questions[training.SCROLL_DIRECT]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["DOWN", "UP", "LEFT", "RIGHT"]
    assert question["criteria"]["DOWN"] == jev.OPERATION_DESCRIPTIONS["SCROLL_DOWN"]
    assert question["instructions"] == {
        "goal": GOAL,
        "rules": training.SCROLL_DIRECT_INSTRUCTIONS,
    }


def test_the_agents_region_question_is_left_in_place():
    request = training.build_training_request(GOAL, scroll_screen())
    assert "scroll_target" in request.questions
    assert request.questions["scroll_target"]["criteria"] == {
        "2": "Scrollable region [2] List"
    }


def test_a_screen_without_a_region_gets_no_scroll_operation():
    request = training.build_training_request(
        GOAL, observation([element(id="0", text="Save", clickable=True)])
    )
    assert not [name for name in request.questions["operation"]["criteria"]
                if name.startswith("SCROLL")]
    assert training.SCROLL_DIRECT not in request.questions
    assert all("SCROLL" not in entry["operations"] for entry in request.space.elements)


def test_goal_named_apps_carry_the_question_when_the_answer_is_among_them():
    apps = ("Clock", "Notes", "Camera", "Calculator")
    request = training.build_training_request(
        "Open Clock and Notes", observation([]), apps=apps, gt_app="Clock", seed="s"
    )
    assert request.apps == ("Clock", "Notes")
    assert list(request.questions["app_target"]["criteria"].values()) == ["Clock", "Notes"]
    assert request.state["availableApps"] == [
        {"index": "1", "label": "Clock"},
        {"index": "2", "label": "Notes"},
    ]


def test_a_goal_that_names_one_app_gets_a_sample_instead():
    apps = inventory()
    request = training.build_training_request(
        "Open Clock", observation([]), apps=apps, gt_app="Clock", seed="step-0"
    )
    offered = list(request.questions["app_target"]["criteria"].values())
    assert "Clock" in offered
    assert training.APP_SAMPLE_MIN + 1 <= len(offered) <= training.APP_SAMPLE_MAX + 1
    # The sample is drawn from the inventory, so it holds nothing else.
    assert set(offered) <= set(apps) | {"Clock"}
    assert request.state["availableApps"] == [
        {"index": str(index), "label": name} for index, name in enumerate(offered, 1)
    ]


def test_a_goal_that_names_nothing_gets_a_sample_too():
    apps = inventory()
    request = training.build_training_request(
        "Do the thing", observation([]), apps=apps, gt_app="App007", seed="step-1"
    )
    offered = list(request.questions["app_target"]["criteria"].values())
    assert "App007" in offered
    assert training.APP_SAMPLE_MIN + 1 <= len(offered) <= training.APP_SAMPLE_MAX + 1


def test_a_goal_that_names_other_apps_still_offers_the_recorded_one():
    apps = ("Clock", "Notes", "Camera", "Calculator")
    request = training.build_training_request(
        "Open Clock and Notes", observation([]), apps=apps, gt_app="Camera", seed="s"
    )
    offered = list(request.questions["app_target"]["criteria"].values())
    assert "Camera" in offered
    assert len(offered) == len(apps)


def test_the_recorded_app_is_never_a_distractor():
    apps = inventory()
    for seed in ("a", "b", "c", "d"):
        request = training.build_training_request(
            "Do the thing", observation([]), apps=apps, gt_app="App003", seed=seed
        )
        offered = list(request.questions["app_target"]["criteria"].values())
        assert offered.count("App003") == 1


def test_sampling_is_reproducible_and_step_dependent():
    apps = inventory()
    first = training.app_candidates(apps, "Do it", "App001", seed="android_control_7_step0")
    again = training.app_candidates(apps, "Do it", "App001", seed="android_control_7_step0")
    other = training.app_candidates(apps, "Do it", "App001", seed="android_control_7_step1")
    assert first == again
    assert first != other
    # The same seed with a different ground truth is a different question.
    assert first != training.app_candidates(apps, "Do it", "App002", seed="android_control_7_step0")
    # A fresh Random per call, so a caller's global RNG state cannot leak in.
    random.seed(1)
    assert training.app_candidates(apps, "Do it", "App001", seed="android_control_7_step0") == first


def test_the_sample_size_stays_within_the_bounds():
    apps = inventory(200)
    sizes = {
        len(training.app_candidates(apps, "Do it", "App000", seed=f"step-{index}"))
        for index in range(40)
    }
    assert min(sizes) >= training.APP_SAMPLE_MIN + 1
    assert max(sizes) <= training.APP_SAMPLE_MAX + 1
    assert len(sizes) > 1  # the size really is drawn, not fixed


def test_a_small_inventory_is_offered_whole():
    apps = ("Clock", "Notes")
    assert training.app_candidates(apps, "Do it", "Clock", seed="s") == ["Clock", "Notes"]
    assert training.app_candidates(apps, "Do it", None, seed="s") == ["Clock", "Notes"]


def test_an_inventory_with_only_the_recorded_app_offers_one_candidate():
    # The row is dropped by the caller; the overlay does not invent apps.
    assert training.app_candidates(("Clock",), "Do it", "Clock", seed="s") == ["Clock"]


def test_an_empty_inventory_leaves_the_agents_request_alone():
    request = training.build_training_request("Do it", observation([]), apps=())
    assert "app_target" not in request.questions
    assert request.state["availableApps"] == []
    assert "OPEN_APP" not in request.questions["operation"]["criteria"]


def test_the_overlay_does_not_mutate_the_port():
    """The port's own builder still returns the agent's request, unmerged."""
    screen = scroll_screen()
    port_request = jev.build_request(GOAL, screen, (), ("Clock", "Notes"))
    assert "SCROLL_DOWN" in port_request.questions["operation"]["criteria"]
    assert training.SCROLL_DIRECT not in port_request.questions
    training.build_training_request(GOAL, screen, (), ("Clock", "Notes"))
    # A second port request is unaffected by the overlay's rewrite of its own.
    fresh = jev.build_request(GOAL, screen, (), ("Clock", "Notes"))
    assert list(fresh.questions["operation"]["criteria"]) == list(
        port_request.questions["operation"]["criteria"]
    )
