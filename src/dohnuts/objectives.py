"""SFT and GRPO training targets, batched over questions on one shared forward.

RLCD remains in ``dohnuts.rlcd`` and is only routed through this interface so the
training loop has one entry point. See docs/rlcd.md for the objective comparison.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from dohnuts.rlcd import RLCDConfig, rlcd_loss

SFT_DEFAULTS = {"rps_weight": 0.0, "brier_weight": 0.0}

GRPO_DEFAULTS = {
    "group_size": 16,
    "num_iterations": 1,
    "correctness_weight": 1.0,
    "confidence_weight": 1.0,
    "brier_weight": 1.0,
    "beta": 0.02,
    "clip_epsilon": 0.2,
    "advantage_epsilon": 1e-6,
}


def _options(defaults, options, label):
    """Validate a closed option set: unknown keys and out-of-range values fail loudly."""
    unknown = sorted(set(options) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown {label} options: {unknown}")
    result = dict(defaults, **options)
    for name, minimum in (("group_size", 2), ("num_iterations", 1)):
        if name not in result:
            continue
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{label}.{name} must be an integer >= {minimum}")
    for name, value in result.items():
        if name in {"group_size", "num_iterations"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{name} must be finite and nonnegative")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label}.{name} must be finite and nonnegative")
    return result


def sft_options(options):
    return _options(SFT_DEFAULTS, options, "sft")


def grpo_options(options):
    result = _options(GRPO_DEFAULTS, options, "grpo")
    if not 0 < result["clip_epsilon"] < 1 or result["advantage_epsilon"] <= 0:
        raise ValueError("GRPO requires 0 < clip_epsilon < 1 and advantage_epsilon > 0")
    if result["correctness_weight"] == result["confidence_weight"] == 0:
        raise ValueError("At least one GRPO reward weight must be positive")
    return result


def _check_batch(logits, targets, mask):
    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Expected logits, targets, and mask of shape [B,K]")
    if logits.shape[1] < 2:
        raise ValueError("Each question requires at least two candidates")
    rows = targets.sum(-1)
    if not torch.isclose(rows, torch.ones_like(rows), atol=1e-5).all():
        raise ValueError("Targets must be label distributions that sum to one")


class SFTObjective:
    """Distribution cross-entropy with optional RPS and Brier terms.

    RPS is an ordinal penalty and only counts rows the collator marked as score
    questions; it is skipped entirely when its weight is zero.
    """

    num_iterations = 1
    needs_rollout = False

    def __init__(self, config):
        options = sft_options(config)
        self.rps_weight = options["rps_weight"]
        self.brier_weight = options["brier_weight"]
        self.metric_names = ("ce", "rps") + (("brier",) if self.brier_weight else ())

    def prepare(self, logits, targets, mask):
        return None

    def reference_logits(self, inputs, positions, decision_positions):
        return None

    def loss(self, logits, targets, mask, ordinal, rollout=None, reference_logits=None):
        _check_batch(logits, targets, mask)
        log_probs = logits.float().masked_fill(~mask, -torch.inf).log_softmax(-1)
        probabilities = log_probs.exp()
        ce = -(targets * log_probs).masked_fill(~mask, 0).sum(-1).mean()
        difference = (probabilities.cumsum(-1) - targets.cumsum(-1))[..., :-1]
        rows = difference.square().mean(-1)
        rps = (rows * ordinal).sum() / ordinal.sum().clamp_min(1)
        brier = ((probabilities - targets).square() * mask).sum(-1).mean()
        total = ce + self.rps_weight * rps + self.brier_weight * brier
        metrics = {"ce": ce.detach(), "rps": rps.detach()}
        if self.brier_weight:
            metrics["brier"] = brier.detach()
        return total, metrics

    def describe(self):
        return {
            "method": "sft",
            "rps_weight": self.rps_weight,
            "brier_weight": self.brier_weight,
        }

    def state_dict(self):
        return None


class RLCDObjective:
    """The pinned Laya estimate, re-exposed on the common batched interface."""

    num_iterations = 1
    needs_rollout = False
    metric_names = ("reward_mean", "reward_std", "advantage_rms", "policy_loss", "ce_loss")

    def __init__(self, config):
        if not isinstance(config, RLCDConfig):
            raise ValueError("RLCDObjective requires an RLCDConfig")
        self.config = config

    def prepare(self, logits, targets, mask):
        return None

    def reference_logits(self, inputs, positions, decision_positions):
        return None

    def loss(self, logits, targets, mask, ordinal, rollout=None, reference_logits=None):
        return rlcd_loss(logits, targets, mask=mask, ordinal=ordinal, config=self.config)

    def describe(self):
        return {"method": "rlcd", **asdict(self.config)}

    def state_dict(self):
        return None


@dataclass
class Rollout:
    """One frozen group per question, reused by every policy update in the batch."""

    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    rewards: Tensor
    correctness: Tensor
    confidence: Tensor
    confidence_error: Tensor


def group_advantages(rewards, epsilon):
    """Normalize within each question's group; a constant group is exactly zero."""
    rewards = rewards.detach()
    centered = rewards - rewards.mean(-1, keepdim=True)
    std = rewards.std(-1, unbiased=False, keepdim=True)
    zero = (rewards == rewards[:, :1]).all(dim=-1, keepdim=True)
    return torch.where(zero, torch.zeros_like(centered), centered / (std + epsilon))


def make_reference(model, weights):
    """A frozen copy of the policy whose trainable parameters hold the snapshot."""
    reference = copy.deepcopy(model)
    reference.requires_grad_(False).eval()
    # The copy is frozen, so it may cache image features again; the policy moves on.
    reference.adapter._vision_trainable = False
    cache = getattr(reference.backbone, "_dohnuts_image_cache", None)
    if cache is not None:
        cache.clear()
    parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    return reference


class GRPOObjective:
    """Categorical GRPO with a fixed reference: multinomial sampling, clipped ratio,
    exact categorical KL, and a Brier term that keeps learning signal when a whole
    group samples the same candidate (advantages are exactly zero there)."""

    needs_rollout = True

    def __init__(self, options, *, model=None, training_state=None, resuming=False):
        self.options = grpo_options(options)
        self.num_iterations = self.options["num_iterations"]
        self.metric_names = (
            "policy_loss",
            "kl",
            "brier",
            "reward_mean",
            "reward_std",
            "clip_fraction",
            "zero_advantage_group",
            "sample_correctness",
            "sample_confidence",
            "confidence_error",
        )
        self.reference = None
        self.reference_weights = None
        if not self.options["beta"]:
            return
        if model is None:
            raise ValueError("GRPO with beta > 0 requires the policy model")
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        if resuming:
            weights = (training_state or {}).get("reference_weights") or {}
            if set(weights) != names:
                raise ValueError("GRPO resume requires the original frozen reference weights")
            self.reference_weights = weights
        else:
            self.reference_weights = {
                n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
            }
        self.reference = make_reference(model, self.reference_weights)

    @torch.no_grad()
    def prepare(self, logits, targets, mask):
        _check_batch(logits, targets, mask)
        options = self.options
        log_probs = logits.float().masked_fill(~mask, -torch.inf).log_softmax(-1)
        probabilities = log_probs.exp()
        actions = torch.multinomial(probabilities, options["group_size"], replacement=True)
        correctness = targets.gather(1, actions)
        confidence = probabilities.gather(1, actions)
        error = correctness * (1 - confidence).square() + (1 - correctness) * confidence.square()
        rewards = options["correctness_weight"] * correctness - options["confidence_weight"] * error
        return Rollout(
            actions=actions,
            old_log_probs=log_probs.gather(1, actions),
            advantages=group_advantages(rewards, options["advantage_epsilon"]),
            rewards=rewards,
            correctness=correctness,
            confidence=confidence,
            confidence_error=error,
        )

    @torch.no_grad()
    def reference_logits(self, inputs, positions, decision_positions):
        if self.reference is None:
            return None
        return self.reference(inputs, positions, decision_positions)

    def loss(self, logits, targets, mask, ordinal, rollout=None, reference_logits=None):
        if rollout is None:
            raise ValueError("GRPO requires a prepared rollout")
        options = self.options
        log_probs = logits.float().masked_fill(~mask, -torch.inf).log_softmax(-1)
        probabilities = log_probs.exp()
        ratio = (log_probs.gather(1, rollout.actions) - rollout.old_log_probs).exp()
        advantages = rollout.advantages
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - options["clip_epsilon"], 1 + options["clip_epsilon"]) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        brier = ((probabilities - targets).square() * mask).sum(-1).mean()
        if reference_logits is None:
            kl = logits.new_zeros(())
        else:
            reference = reference_logits.float().log_softmax(-1)
            # Padded candidates carry log-probability -inf, and 0 * -inf is NaN.
            # Mask before the product so the backward pass stays finite: the
            # probability is already exactly zero there, so nothing is lost.
            terms = probabilities * (
                log_probs.masked_fill(~mask, 0.0) - reference.masked_fill(~mask, 0.0)
            )
            kl = terms.sum(-1).mean()
        total = policy_loss + options["beta"] * kl + options["brier_weight"] * brier
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl": kl.detach(),
            "brier": brier.detach(),
            "reward_mean": rollout.rewards.mean(),
            "reward_std": rollout.rewards.std(unbiased=False),
            "clip_fraction": (clipped < unclipped).float().mean(),
            "zero_advantage_group": (rollout.advantages == 0).all().float(),
            "sample_correctness": rollout.correctness.mean(),
            "sample_confidence": rollout.confidence.mean(),
            "confidence_error": rollout.confidence_error.mean(),
        }
        return total, {key: value.detach() for key, value in metrics.items()}

    def describe(self):
        return {"method": "grpo", **self.options}

    def state_dict(self):
        return {"reference_weights": self.reference_weights} if self.reference is not None else None


def build_objective(config, *, policy=None, model=None, training_state=None, resuming=False):
    """Route the recipe's method to its objective, without touching rlcd.py."""
    method = config["method"]
    if method == "rlcd":
        return RLCDObjective(policy or RLCDConfig(**config.get("rlcd", {})))
    if method == "sft":
        return SFTObjective(config.get("sft", {}))
    if method == "grpo":
        return GRPOObjective(
            config.get("grpo", {}),
            model=model,
            training_state=training_state,
            resuming=resuming,
        )
    raise ValueError(f"Unsupported training method: {method}")
