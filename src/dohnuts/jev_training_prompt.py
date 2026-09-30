"""The training shape of the prompt: the agent's request plus three deviations.

`dohnuts.mobile_jev_prompt` is a verbatim port of android_world's
`ClientMobileJev` policy, and it stays that way so it can be read next to the
original. This module wraps it for Android Control training data and changes
exactly three things, each of them deliberate and stated in the manifest:

1. **One SCROLL operation.** The agent offers `SCROLL_DOWN`, `SCROLL_UP`,
   `SCROLL_LEFT` and `SCROLL_RIGHT` as four operations; here they collapse into a
   single `SCROLL`, and a new `scroll_direct` question chooses the direction. The
   element entries lose their per-direction operations the same way.
2. **No `text_value` rows.** The question stays in the request -- the agent builds
   it, and `state.textSource` describes it -- but the conversion emits no row for
   it; `TYPE_TEXT` remains an operation.
3. **A sampled app inventory.** The agent offers up to 200 installed apps. Here
   the criteria are the apps the goal names, when there are at least two and the
   recorded app is among them; otherwise a deterministic sample of 15..30 names
   from the corpus vocabulary plus the recorded app, so the question always has
   an answer and never a single option.

Everything else -- the rules text, the state keys and their order, the tap and
scroll_target questions, the option limits -- is the agent's own, which is what
the parity test in `tests/test_mobile_jev_prompt.py` checks.
"""

import dataclasses
import random
from typing import Any

from dohnuts import mobile_jev_prompt as jev

# The sampled app question: how many distractors, and the seed the choice is
# derived from. The seed depends only on the step and the recorded app, so a
# rerun -- serial or parallel -- samples exactly the same names.
APP_SAMPLE_MIN = 15
APP_SAMPLE_MAX = 30
APP_SAMPLE_SEED = 'doh-ac-app-sample-2026'

SCROLL_OPERATION = 'SCROLL'
SCROLL_DESCRIPTION = (
    'Scroll the screen to reveal more content in one direction; another '
    'question chooses that direction.'
)

SCROLL_DIRECT = 'scroll_direct'
SCROLL_DIRECT_INSTRUCTIONS = (
    'Assuming the next operation is SCROLL, choose the direction that reveals '
    'the content the goal needs. Down reveals content further down the screen, '
    'up content above it, and left or right content to the sides. Use the '
    'visible screen and recent actions. Choose only an offered direction.'
)

# The four directions keep the agent's own wording, and the agent's order.
SCROLL_DIRECT_CRITERIA = {
    'DOWN': jev.OPERATION_DESCRIPTIONS['SCROLL_DOWN'],
    'UP': jev.OPERATION_DESCRIPTIONS['SCROLL_UP'],
    'LEFT': jev.OPERATION_DESCRIPTIONS['SCROLL_LEFT'],
    'RIGHT': jev.OPERATION_DESCRIPTIONS['SCROLL_RIGHT'],
}

SCROLL_DIRECTIONS = tuple(SCROLL_DIRECT_CRITERIA)


def app_candidates(
    inventory: Any, goal: str, gt_app: str | None, *, seed: str
) -> list[str]:
    """The apps one `app_target` question offers, in inventory order.

    The goal's own mentions win when they can carry the question: at least two of
    them, and -- when this step opened an app -- that app among them. Otherwise
    the question is a deterministic sample of `APP_SAMPLE_MIN`..`APP_SAMPLE_MAX`
    names plus the recorded app, so a goal that names one app -- or none, or
    names apps other than the one that was opened -- still gets a question with
    real alternatives instead of a single option.

    Args:
      inventory: Every app name the corpus opens, in a fixed order.
      goal: The task goal, matched with the agent's whole-word rule.
      gt_app: The app the step opened, always offered when it is known.
      seed: Distinguishes one step's sample from another's; the same seed always
        draws the same names.

    Returns:
      The offered names, in the inventory's own order.
    """
    names = list(inventory)
    named = jev.named_apps(names, goal)
    if len(named) >= 2 and (gt_app is None or gt_app in named):
        return [name for name in names if name in set(named)]
    rng = random.Random(f'{APP_SAMPLE_SEED}:{seed}:{gt_app}')
    others = [name for name in names if name != gt_app]
    count = min(rng.randint(APP_SAMPLE_MIN, APP_SAMPLE_MAX), len(others))
    chosen = set(rng.sample(others, count)) if count else set()
    if gt_app is not None:
        chosen.add(gt_app)
    offered = [name for name in names if name in chosen]
    # The recorded app is always offered, even if the inventory does not carry
    # it (a corpus whose vocabulary missed one action still gets a valid row).
    return offered + [name for name in chosen if name not in names]


def scroll_direct_question() -> dict[str, Any]:
    """The direction question, built the way the agent builds its own."""
    return {
        'type': 'choice',
        'instructions': SCROLL_DIRECT_INSTRUCTIONS,
        'criteria': dict(SCROLL_DIRECT_CRITERIA),
    }


def merge_scroll_operations(request: jev.Request) -> None:
    """Collapse the agent's four scroll operations into one, in place.

    The operation criteria keep the position the first direction had, the element
    entries list `SCROLL` once, and the direction question is added. The agent's
    `scroll_target` question stays where it was: the conversion emits no row for
    it, but the state and the history labels still use its region.
    """
    criteria = request.questions['operation']['criteria']
    if not any(name.startswith('SCROLL_') for name in criteria):
        return
    merged: dict[str, str] = {}
    for name, description in criteria.items():
        if name.startswith('SCROLL_'):
            if SCROLL_OPERATION not in merged:
                merged[SCROLL_OPERATION] = SCROLL_DESCRIPTION
            continue
        merged[name] = description
    request.questions['operation']['criteria'] = merged
    for entry in request.space.elements:
        operations = entry['operations']
        if any(name.startswith('SCROLL_') for name in operations):
            entry['operations'] = list(
                dict.fromkeys(
                    SCROLL_OPERATION if name.startswith('SCROLL_') else name
                    for name in operations
                )
            )
    request.questions[SCROLL_DIRECT] = scroll_direct_question()


def build_training_request(
    goal: str,
    observation: jev.Observation,
    history: Any = (),
    apps: Any = (),
    *,
    gt_app: str | None = None,
    seed: str = '',
) -> jev.Request:
    """The agent's request for one decision, reshaped for training.

    Args:
      goal: The task goal, verbatim.
      observation: The summarized screen.
      history: Executed decisions, most recent last.
      apps: The corpus app inventory.
      gt_app: The app this step opened, when it opened one.
      seed: The step's identity, so the app sample is reproducible.

    Returns:
      The request to write rows against: the agent's own, with the three
      deviations of this module applied.

    Raises:
      jev.PayloadTooLargeError: If the request cannot be sent untruncated.
    """
    request = jev.build_request(goal, observation, history, apps)
    merge_scroll_operations(request)
    if SCROLL_DIRECT in request.questions:
        # The overlay adds this question after the agent wrapped the others, so
        # it has to be wrapped the same way.
        question = request.questions[SCROLL_DIRECT]
        question['instructions'] = {'goal': goal, 'rules': question['instructions']}
    offered = app_candidates(apps, goal, gt_app, seed=seed)
    if not offered:
        # No inventory at all: the agent asks no app question either, so the
        # request stays exactly as it built it.
        return request
    request.questions['app_target'] = {
        'type': 'choice',
        'instructions': {
            'goal': goal,
            'rules': jev.TARGET_QUESTION_TEMPLATE.format(operation='OPEN_APP'),
        },
        'criteria': {str(index): name for index, name in enumerate(offered, 1)},
    }
    request.state['availableApps'] = [
        {'index': str(index), 'label': name} for index, name in enumerate(offered, 1)
    ]
    return dataclasses.replace(request, apps=tuple(offered))
