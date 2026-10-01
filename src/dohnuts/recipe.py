"""The Dohnuts training and evaluation recipe.

Values are recorded in each run for reproducibility. They are product defaults,
not a configuration surface. Model-specific behavior belongs in an adapter.

The budget is expressed in epochs by default: ``steps`` stays ``None`` and
``train()`` turns ``epochs`` into optimizer updates once it knows how many
training rows the recipe reads and how many GPUs split the work. Setting
``TRAINING_STEPS`` pins a fixed number of updates instead and overrides the
epoch budget, which is what an apples-to-apples comparison against a previous
run needs.
"""

import math
from pathlib import Path

from dohnuts.objectives import grpo_options, sft_options
from dohnuts.rlcd import RLCDConfig

IMAGE_PIXELS = 1024**2
# Measured, not chosen: 1024^2 screenshots cost 1,369 visual tokens, and the longest
# measured row of ac-jev-v1 -- 20,000 random rows through the collator's own formula --
# reaches 8,942 tokens. 12,288 leaves room above that tail while staying far inside the
# base model's 262,144-token context.
MAX_LENGTH = 12_288
TRAINING_STEPS = None
EPOCHS = 3
# Reading limits per dataset group.  ``None`` reads the whole split, which is what a
# full-corpus run wants; a number caps each group, which is what keeps a repeated
# evaluation pass over a large split affordable.  These are plain defaults, not pins:
# a caller passing ``train_cap``/``dev_cap`` overrides them, and whatever the run used
# is recorded in its recipe.
TRAIN_CAP = None
DEV_CAP = None
LR_DECAY_STEPS = 2400
BASE_MODEL = Path(".cache/models/Qwen3.5-0.8B")
DATA = Path("data/processed/v1")
METHODS = ("rlcd", "sft", "grpo")
STAGES = ("warmup", "text", "joint", "vision_top")


def resolve_steps(*, rows, epochs, batch_size, accumulation, world_size=1, steps=None):
    """Turn the recipe's budget into the number of optimizer updates.

    A fixed ``steps`` budget wins, because it is the override. Otherwise one
    update consumes ``batch_size * accumulation * world_size`` samples, so the
    same ``epochs`` walks the training split exactly once per epoch whatever the
    GPU count -- adding GPUs shortens the run instead of quietly training longer.
    """
    if steps is not None:
        return steps
    if rows < 1:
        raise ValueError("An epoch budget needs a non-empty training split")
    per_update = batch_size * accumulation * world_size
    return math.ceil(rows * epochs / per_update)


def training_recipe(
    *,
    model=BASE_MODEL,
    data=DATA,
    seed=42,
    rlcd=None,
    epochs=EPOCHS,
    steps=TRAINING_STEPS,
    train_cap=TRAIN_CAP,
    dev_cap=DEV_CAP,
    method="rlcd",
    stage="text",
    projection_dim=256,
    lora_rank=8,
    lora_alpha=16,
    sft=None,
    grpo=None,
):
    if TRAINING_STEPS is not None:
        # A step budget pinned in this module outranks both the caller and epochs.
        steps = TRAINING_STEPS
    if steps is not None and (not isinstance(steps, int) or steps < 1):
        raise ValueError("Training steps must be a positive integer")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Training epochs must be a positive integer")
    for name, cap in (("train_cap", train_cap), ("dev_cap", dev_cap)):
        if cap is not None and (not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{name} must be None or a positive integer")
    if method not in METHODS:
        raise ValueError(f"Unsupported training method: {method}")
    if stage not in STAGES:
        raise ValueError(f"Unsupported training stage: {stage}")
    if not isinstance(projection_dim, int) or projection_dim < 1:
        raise ValueError("The projection dimension must be a positive integer")
    if not isinstance(lora_rank, int) or lora_rank < 1 or not isinstance(lora_alpha, int):
        raise ValueError("LoRA rank and alpha must be positive integers")
    policy = rlcd or RLCDConfig()
    recipe = {
        "model": str(model),
        "data": str(data),
        "seed": seed,
        "method": method,
        "stage": stage,
        "projection_dim": projection_dim,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "batch_size": 8,
        "accumulation": 4,
        "epochs": epochs,
        "steps": steps,
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "merger_lr": 1e-5,
        "vision_lr": 2e-6,
        "train_cap": train_cap,
        "dev_cap": dev_cap,
        "image_pixels": IMAGE_PIXELS,
        "max_length": MAX_LENGTH,
        "cpu_threads": 8,
        "workers": 2,
        "eval_batch_size": 16,
        "log_every": 10,
        "eval_every": 400,
        "save_every": 100,
        "backend": {
            "linear_patch": True,
            "triton_convolution": False,
            "fused_norm_and_swiglu": True,
            "shared_prefix": True,
            "frozen_vision_cache_MiB": 128,
        },
    }
    if method == "rlcd":
        if policy != RLCDConfig():
            recipe["rlcd"] = {"sigma": policy.sigma, "ce_weight": policy.ce_weight}
    elif method == "sft":
        recipe["sft"] = sft_options(sft or {})
    else:
        recipe["grpo"] = grpo_options(grpo or {})
    return recipe
