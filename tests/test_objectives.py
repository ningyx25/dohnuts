"""SFT and GRPO closed forms, option validation, and the frozen reference."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dohnuts.objectives import (
    GRPOObjective,
    RLCDObjective,
    Rollout,
    SFTObjective,
    build_objective,
    group_advantages,
    grpo_options,
    make_reference,
)
from dohnuts.rlcd import RLCDConfig, rlcd_loss


class StubAdapter:
    _vision_trainable = False


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._dohnuts_image_cache = {}


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = StubAdapter()
        self.backbone = StubBackbone()
        self.head = nn.Linear(3, 2)


def distribution(*rows):
    targets = torch.tensor([list(row) for row in rows], dtype=torch.float32)
    return targets, torch.ones_like(targets, dtype=torch.bool)


def test_sft_default_is_the_distribution_cross_entropy():
    logits = torch.tensor([[0.5, -0.25, 1.5], [2.0, 0.0, -1.0]])
    targets, mask = distribution([0.2, 0.3, 0.5], [0.0, 1.0, 0.0])
    loss, metrics = SFTObjective({}).loss(logits, targets, mask, torch.tensor([False, False]))
    torch.testing.assert_close(loss, F.cross_entropy(logits, targets))
    assert metrics["rps"].item() == 0
    assert set(metrics) == {"ce", "rps"}


def test_sft_ignores_padded_candidates():
    logits = torch.tensor([[0.5, -0.25, 1.5]])
    targets = torch.tensor([[0.4, 0.6, 0.0]])
    mask = torch.tensor([[True, True, False]])
    loss, _ = SFTObjective({}).loss(logits, targets, mask, torch.tensor([False]))
    expected = F.cross_entropy(logits[:, :2], targets[:, :2])
    torch.testing.assert_close(loss, expected)


def test_rps_only_counts_rows_marked_as_score():
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    targets, mask = distribution([0.0, 0.0, 1.0], [0.5, 0.5, 0.0])
    ordinal = torch.tensor([True, False])
    loss, metrics = SFTObjective({"rps_weight": 1.0}).loss(logits, targets, mask, ordinal)
    probabilities = logits.softmax(-1)
    difference = (probabilities.cumsum(-1) - targets.cumsum(-1))[..., :-1]
    expected = difference.square().mean(-1)[0]
    torch.testing.assert_close(metrics["rps"], expected)
    torch.testing.assert_close(loss, metrics["ce"] + expected)


def test_brier_weight_adds_the_squared_error_term():
    logits = torch.tensor([[0.5, -0.25, 1.5]])
    targets, mask = distribution([0.2, 0.3, 0.5])
    loss, metrics = SFTObjective({"brier_weight": 0.5}).loss(
        logits, targets, mask, torch.tensor([False])
    )
    probabilities = logits.softmax(-1)
    expected = ((probabilities - targets) ** 2).sum(-1).mean()
    torch.testing.assert_close(metrics["brier"], expected)
    torch.testing.assert_close(loss, metrics["ce"] + 0.5 * expected)
    assert set(metrics) == {"ce", "rps", "brier"}


def test_unknown_sft_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown sft options"):
        SFTObjective({"rps_weigth": 1.0})


def test_non_distribution_targets_and_single_candidate_are_rejected():
    logits = torch.zeros(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="sum to one"):
        SFTObjective({}).loss(logits, torch.tensor([[0.3, 0.3]]), mask, torch.tensor([False]))
    with pytest.raises(ValueError, match="two candidates"):
        SFTObjective({}).loss(
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([False]),
        )


def test_grpo_defaults_and_validation():
    assert grpo_options({})["num_iterations"] == 1
    assert grpo_options({})["group_size"] == 16
    with pytest.raises(ValueError, match="Unknown grpo options"):
        grpo_options({"group_sizes": 2})
    for options in ({"group_size": 1}, {"group_size": True}, {"num_iterations": 0}):
        with pytest.raises(ValueError, match="integer >= "):
            grpo_options(options)
    with pytest.raises(ValueError, match="clip_epsilon"):
        grpo_options({"clip_epsilon": 1.0})
    with pytest.raises(ValueError, match="advantage_epsilon"):
        grpo_options({"advantage_epsilon": 0.0})
    with pytest.raises(ValueError, match="reward weight"):
        grpo_options({"correctness_weight": 0.0, "confidence_weight": 0.0})
    for options in ({"beta": -1.0}, {"beta": True}, {"brier_weight": float("inf")}):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            grpo_options(options)


def test_constant_group_advantage_is_exactly_zero():
    rewards = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert (group_advantages(rewards, 1e-6) == 0).all()


def test_group_advantages_normalize_within_each_row():
    rewards = torch.tensor([[0.0, 1.0, 2.0], [3.0, 3.0, 3.0]])
    advantages = group_advantages(rewards, 1e-6)
    row = rewards[0]
    expected = (row - row.mean()) / (row.std(unbiased=False) + 1e-6)
    torch.testing.assert_close(advantages[0], expected)
    assert (advantages[1] == 0).all()


def test_grpo_rollout_and_update_match_the_closed_form():
    torch.manual_seed(3)
    logits = torch.tensor([[0.4, -0.2, 1.1]])
    targets, mask = distribution([0.1, 0.2, 0.7])
    objective = GRPOObjective({"group_size": 8, "beta": 0.0})
    rollout = objective.prepare(logits, targets, mask)
    probabilities = logits.softmax(-1)
    assert rollout.actions.shape == (1, 8)
    assert int(rollout.actions.max()) < 3
    torch.testing.assert_close(rollout.correctness, targets.gather(1, rollout.actions))
    torch.testing.assert_close(rollout.confidence, probabilities.gather(1, rollout.actions))
    error = (
        rollout.correctness * (1 - rollout.confidence) ** 2
        + (1 - rollout.correctness) * rollout.confidence**2
    )
    torch.testing.assert_close(rollout.confidence_error, error)
    rewards = rollout.correctness - error
    torch.testing.assert_close(rollout.rewards, rewards)
    torch.testing.assert_close(rollout.advantages, group_advantages(rewards, 1e-6))
    torch.testing.assert_close(
        rollout.old_log_probs, probabilities.log().gather(1, rollout.actions)
    )
    loss, metrics = objective.loss(logits, targets, mask, torch.tensor([False]), rollout)
    # With the rollout's own logits the ratio is exactly one, so clipping is inert.
    torch.testing.assert_close(metrics["policy_loss"], -rollout.advantages.mean())
    torch.testing.assert_close(metrics["brier"], ((probabilities - targets) ** 2).sum(-1).mean())
    assert metrics["kl"].item() == 0
    assert metrics["clip_fraction"].item() == 0
    torch.testing.assert_close(loss, metrics["policy_loss"] + metrics["brier"])


def test_grpo_clips_the_ratio_and_uses_the_exact_categorical_kl():
    logits = torch.tensor([[0.4, -0.2, 1.1]])
    targets, mask = distribution([0.1, 0.2, 0.7])
    probabilities = logits.softmax(-1)
    log_probs = logits.log_softmax(-1)
    # Ratio 2.57 on the positive advantage and 0.67 on the negative one: the
    # clipped branch binds in both directions.
    rollout = Rollout(
        actions=torch.tensor([[0, 2]]),
        old_log_probs=torch.log(torch.tensor([[0.1, 0.9]])),
        advantages=torch.tensor([[1.0, -1.0]]),
        rewards=torch.zeros(1, 2),
        correctness=torch.zeros(1, 2),
        confidence=torch.zeros(1, 2),
        confidence_error=torch.zeros(1, 2),
    )
    reference = torch.tensor([[1.0, 0.0, -1.0]])
    objective = GRPOObjective(
        {"group_size": 2, "beta": 0.5, "brier_weight": 0.0}, model=StubModel()
    )
    loss, metrics = objective.loss(logits, targets, mask, torch.tensor([False]), rollout, reference)
    ratio = (log_probs.gather(1, rollout.actions) - rollout.old_log_probs).exp()
    unclipped = ratio * rollout.advantages
    clipped = ratio.clamp(1 - 0.2, 1 + 0.2) * rollout.advantages
    torch.testing.assert_close(metrics["policy_loss"], -torch.minimum(unclipped, clipped).mean())
    assert metrics["clip_fraction"].item() == 1.0
    expected_kl = (probabilities * (log_probs - reference.log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(metrics["kl"], expected_kl)
    torch.testing.assert_close(loss, metrics["policy_loss"] + 0.5 * expected_kl)


def test_padded_candidates_do_not_poison_the_kl_gradient():
    logits = torch.tensor([[0.4, -0.2, 1.1, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.3, 0.2, 0.5, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    reference = torch.zeros_like(logits)
    objective = GRPOObjective({"beta": 0.5, "brier_weight": 0.0}, model=StubModel())
    rollout = objective.prepare(logits, targets, mask)
    rollout.advantages = torch.ones_like(rollout.advantages)
    loss, metrics = objective.loss(logits, targets, mask, torch.tensor([False]), rollout, reference)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    masked = logits.detach().float().masked_fill(~mask, -torch.inf).log_softmax(-1)
    probabilities = masked.exp()
    expected = (probabilities[:, :3] * (masked[:, :3] - reference.log_softmax(-1)[:, :3])).sum()
    torch.testing.assert_close(metrics["kl"], expected)


def test_constant_group_learns_through_the_brier_term():
    logits = torch.tensor([[0.1, 0.9]], requires_grad=True)
    targets, mask = distribution([0.0, 1.0])
    objective = GRPOObjective({"beta": 0.0})
    rollout = objective.prepare(logits, targets, mask)
    rollout.advantages = torch.zeros_like(rollout.advantages)
    rollout.rewards = torch.full_like(rollout.rewards, 0.5)
    loss, metrics = objective.loss(logits, targets, mask, torch.tensor([False]), rollout)
    assert metrics["zero_advantage_group"].item() == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_correctness_is_the_soft_label_probability_mass():
    logits = torch.zeros(1, 3)
    targets = torch.tensor([[0.25, 0.25, 0.5]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    rollout = GRPOObjective({"group_size": 4, "beta": 0.0}).prepare(logits, targets, mask)
    torch.testing.assert_close(rollout.correctness[0], targets[0][rollout.actions[0]])


def test_grpo_requires_a_prepared_rollout():
    logits = torch.zeros(1, 2)
    targets, mask = distribution([0.5, 0.5])
    with pytest.raises(ValueError, match="requires a prepared rollout"):
        GRPOObjective({"beta": 0.0}).loss(logits, targets, mask, torch.tensor([False]))


def test_beta_zero_builds_no_reference():
    objective = GRPOObjective({"beta": 0.0})
    assert objective.reference is None
    assert objective.state_dict() is None
    assert objective.reference_logits(None, None, None) is None
    assert objective.num_iterations == 1


def test_reference_is_a_frozen_snapshot_for_resume():
    model = StubModel()
    with pytest.raises(ValueError, match="requires the policy model"):
        GRPOObjective({"beta": 0.02})
    objective = GRPOObjective({"beta": 0.02}, model=model)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert set(objective.reference_weights) == set(snapshot)
    assert all(not p.requires_grad for p in objective.reference.parameters())
    with torch.no_grad():
        model.head.weight.add_(1.0)
    torch.testing.assert_close(objective.reference.head.weight, snapshot["head.weight"])
    assert set(objective.state_dict()["reference_weights"]) == set(snapshot)
    with pytest.raises(ValueError, match="reference weights"):
        GRPOObjective({"beta": 0.02}, model=model, training_state={}, resuming=True)
    resumed = GRPOObjective(
        {"beta": 0.02},
        model=model,
        training_state={"reference_weights": snapshot},
        resuming=True,
    )
    torch.testing.assert_close(resumed.reference.head.weight, snapshot["head.weight"])


def test_make_reference_clears_the_stale_image_cache():
    model = StubModel()
    model.backbone._dohnuts_image_cache["stale"] = torch.zeros(1)
    reference = make_reference(model, {n: p.detach().clone() for n, p in model.named_parameters()})
    assert reference.backbone._dohnuts_image_cache == {}
    assert reference.adapter._vision_trainable is False


def test_objective_interfaces_match_the_training_loop():
    assert GRPOObjective({"beta": 0.0}).needs_rollout is True
    assert GRPOObjective({"beta": 0.0}).num_iterations == 1
    assert GRPOObjective({"num_iterations": 3, "beta": 0.0}).num_iterations == 3
    assert SFTObjective({}).needs_rollout is False
    assert SFTObjective({}).num_iterations == 1
    assert RLCDObjective(RLCDConfig()).needs_rollout is False
    assert RLCDObjective(RLCDConfig()).num_iterations == 1


def test_build_objective_dispatches_on_method():
    frozen = {"grpo": {"beta": 0.0}}
    assert isinstance(build_objective({"method": "rlcd"}), RLCDObjective)
    assert isinstance(build_objective({"method": "sft"}), SFTObjective)
    assert isinstance(build_objective({"method": "grpo", **frozen}), GRPOObjective)
    with pytest.raises(ValueError, match="Unsupported training method"):
        build_objective({"method": "dpo"})
    assert build_objective({"method": "rlcd"}).describe()["method"] == "rlcd"
    assert build_objective({"method": "rlcd"}).describe()["sigma"] == RLCDConfig().sigma


def test_describe_records_the_effective_options():
    described = build_objective({"method": "sft", "sft": {"brier_weight": 0.5}}).describe()
    assert described == {"method": "sft", "rps_weight": 0.0, "brier_weight": 0.5}
    grpo = build_objective({"method": "grpo", "grpo": {"group_size": 4, "beta": 0.0}}).describe()
    assert grpo["method"] == "grpo"
    assert grpo["group_size"] == 4
    assert grpo["num_iterations"] == 1


def test_rlcd_objective_delegates_to_the_pinned_estimator():
    logits = torch.randn(2, 3)
    targets, mask = distribution([0.2, 0.3, 0.5], [0.0, 1.0, 0.0])
    ordinal = torch.tensor([True, False])
    config = RLCDConfig()
    torch.manual_seed(0)
    expected, expected_metrics = rlcd_loss(
        logits, targets, mask=mask, ordinal=ordinal, config=config
    )
    torch.manual_seed(0)
    objective = RLCDObjective(config)
    loss, metrics = objective.loss(logits, targets, mask, ordinal)
    torch.testing.assert_close(loss, expected)
    assert set(metrics) == set(expected_metrics) == set(objective.metric_names)
    assert objective.state_dict() is None
    with pytest.raises(ValueError, match="RLCDConfig"):
        RLCDObjective({})
