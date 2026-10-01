"""The recipe's budget: epochs by default, a fixed step count as an override."""

import pytest

from dohnuts.recipe import EPOCHS, TRAINING_STEPS, resolve_steps, training_recipe


def test_the_recipe_budgets_in_epochs_by_default():
    recipe = training_recipe()
    assert recipe["epochs"] == EPOCHS
    assert recipe["steps"] is TRAINING_STEPS is None
    # caps of None mean the whole split is read
    assert recipe["train_cap"] is None
    assert recipe["dev_cap"] is None


def test_serving_accepts_exactly_what_training_accepted():
    """The adapter's input limit is the training budget, not a second number to keep."""
    from dohnuts.adapters import Qwen35Adapter

    assert Qwen35Adapter.max_input_tokens == training_recipe()["max_length"]


def test_one_epoch_walks_the_split_once_whatever_the_gpu_count():
    # an update consumes batch_size * accumulation * world_size samples
    assert resolve_steps(rows=1000, epochs=1, batch_size=8, accumulation=4) == 32
    assert resolve_steps(rows=1000, epochs=1, batch_size=8, accumulation=4, world_size=5) == 7
    assert resolve_steps(rows=1000, epochs=3, batch_size=8, accumulation=4) == 94


def test_a_rounded_up_budget_never_drops_the_tail_of_the_split():
    # 1001 rows * 1 epoch / 32 = 31.28 -> 32 updates, so no row is left unseen
    assert resolve_steps(rows=1001, epochs=1, batch_size=8, accumulation=4) == 32


def test_a_fixed_step_budget_beats_the_epoch_budget():
    assert resolve_steps(rows=10_000, epochs=9, batch_size=8, accumulation=4, steps=7) == 7
    assert training_recipe(epochs=9, steps=7)["steps"] == 7


def test_a_pinned_step_budget_outranks_the_caller_and_epochs(monkeypatch):
    monkeypatch.setattr("dohnuts.recipe.TRAINING_STEPS", 500)
    assert training_recipe(epochs=9, steps=7)["steps"] == 500


def test_an_epoch_budget_needs_something_to_walk_over():
    with pytest.raises(ValueError, match="non-empty training split"):
        resolve_steps(rows=0, epochs=3, batch_size=8, accumulation=4)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"epochs": 1.5}, {"steps": 0}, {"steps": -1}])
def test_a_budget_that_cannot_be_walked_is_rejected(kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        training_recipe(**kwargs)


def test_a_cap_is_recorded_and_changes_nothing_else():
    """A caller-set cap must survive the equality check `train.main()` runs on the recipe."""
    from dohnuts.recipe import DEV_CAP, TRAIN_CAP

    assert (TRAIN_CAP, DEV_CAP) == (None, None)
    recipe = training_recipe()
    capped = training_recipe(train_cap=256, dev_cap=64)
    assert (capped["train_cap"], capped["dev_cap"]) == (256, 64)
    assert training_recipe(dev_cap=8)["train_cap"] is recipe["train_cap"]
    outside = {"train_cap", "dev_cap"}
    assert {k: v for k, v in capped.items() if k not in outside} == {
        k: v for k, v in recipe.items() if k not in outside
    }


@pytest.mark.parametrize("key", ["train_cap", "dev_cap"])
@pytest.mark.parametrize("value", [0, -3, 2.5, "256"])
def test_a_cap_that_reads_nothing_is_rejected(key, value):
    with pytest.raises(ValueError, match="must be None or a positive integer"):
        training_recipe(**{key: value})
