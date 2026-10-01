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
