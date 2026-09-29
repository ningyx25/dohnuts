"""The two-projection decision head: shapes, closed form, and gradients."""

import math

import torch

from dohnuts.model import DecisionHead


def make_head(hidden_size=8, projection_dim=4):
    torch.manual_seed(0)
    return DecisionHead(hidden_size, projection_dim, dtype=torch.float32)


def test_scores_are_the_scaled_projection_dot_product():
    head = make_head()
    hidden = torch.randn(3, 7, 8)
    positions = torch.tensor([[0, 3], [1, 2], [4, 6]])
    decision_positions = torch.tensor([1, 5, 2])
    logits = head(hidden, positions, decision_positions)
    rows = torch.arange(3)
    decision = head.decision(hidden[rows, decision_positions].float())
    candidates = head.candidate(hidden[rows[:, None], positions].float())
    expected = (candidates * decision[:, None, :]).sum(-1) / math.sqrt(4)
    assert logits.shape == (3, 2)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected)


def test_scores_stay_fp32_under_a_bf16_backbone():
    head = make_head()
    hidden = torch.randn(2, 5, 8).to(torch.bfloat16)
    logits = head(
        hidden,
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([0, 1]),
    )
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_decision_position_changes_every_candidate_score():
    head = make_head()
    hidden = torch.randn(1, 6, 8)
    positions = torch.tensor([[1, 2, 3]])
    first = head(hidden, positions, torch.tensor([0]))
    second = head(hidden, positions, torch.tensor([4]))
    assert not torch.allclose(first, second)


def test_both_projections_receive_gradient():
    head = make_head()
    hidden = torch.randn(2, 5, 8)
    head(hidden, torch.tensor([[0, 1], [2, 3]]), torch.tensor([2, 4])).sum().backward()
    assert head.decision.weight.grad is not None
    assert head.candidate.weight.grad is not None
    assert head.decision.weight.grad.abs().sum() > 0
    assert head.candidate.weight.grad.abs().sum() > 0


def test_scale_is_not_serialized_and_parameters_are_two_projections():
    head = make_head()
    assert set(head.state_dict()) == {"decision.weight", "candidate.weight"}
    assert sum(p.numel() for p in head.parameters()) == 2 * 8 * 4
    assert head.scale == 2.0


def test_batching_does_not_mix_rows():
    head = make_head()
    hidden = torch.randn(3, 6, 8)
    positions = torch.tensor([[1, 2], [3, 4], [0, 5]])
    decision_positions = torch.tensor([2, 0, 3])
    batched = head(hidden, positions, decision_positions)
    for row in range(3):
        single = head(
            hidden[row : row + 1],
            positions[row : row + 1],
            decision_positions[row : row + 1],
        )
        torch.testing.assert_close(batched[row], single[0])
