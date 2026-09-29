"""The Dohnuts training and evaluation recipe.

Values are recorded in each run for reproducibility. They are product defaults,
not a configuration surface. Model-specific behavior belongs in an adapter.
"""

from pathlib import Path

from dohnuts.objectives import grpo_options, sft_options
from dohnuts.rlcd import RLCDConfig

IMAGE_PIXELS = 512**2
MAX_LENGTH = 2048
TRAINING_STEPS = 3600
LR_DECAY_STEPS = 2400
BASE_MODEL = Path(".cache/models/Qwen3.5-0.8B")
DATA = Path("data/processed/v1")
METHODS = ("rlcd", "sft", "grpo")
STAGES = ("warmup", "text", "joint", "vision_top")


def training_recipe(
    *,
    model=BASE_MODEL,
    data=DATA,
    seed=42,
    rlcd=None,
    steps=TRAINING_STEPS,
    method="rlcd",
    stage="text",
    projection_dim=256,
    lora_rank=8,
    lora_alpha=16,
    sft=None,
    grpo=None,
):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Training steps must be a positive integer")
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
        "steps": steps,
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "merger_lr": 1e-5,
        "vision_lr": 2e-6,
        "train_cap": 6000,
        "dev_cap": 256,
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
