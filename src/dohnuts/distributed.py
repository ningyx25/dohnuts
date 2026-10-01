"""Single-machine multi-GPU data-parallel training utilities.

When launched via ``torchrun --nproc_per_node=N``, every helper uses the
``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` environment variables to
coordinate.  When the environment is absent (plain ``python`` launch),
everything degrades to single-GPU identity operations so the rest of the
codebase can call these unconditionally.
"""

import os

import torch
import torch.distributed as dist
from torch import Tensor

_rank: int = 0
_local_rank: int = 0
_world_size: int = 1
_initialized: bool = False


def setup() -> tuple[int, int, int]:
    """Initialize the NCCL process group when ``torchrun`` is the launcher.

    Returns ``(rank, local_rank, world_size)``.  With a single process the
    return is ``(0, 0, 1)`` and no process group is created.
    """
    global _rank, _local_rank, _world_size, _initialized

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        # Stay a pure no-op: single-GPU startup must not touch CUDA earlier
        # than it did before, and CPU-only tests can exercise this path.
        _rank, _local_rank, _world_size = 0, 0, 1
        return _rank, _local_rank, _world_size

    _rank = int(os.environ["RANK"])
    _local_rank = int(os.environ["LOCAL_RANK"])
    _world_size = world

    target = torch.device("cuda", _local_rank)
    torch.cuda.set_device(target)
    # Bind the group to this rank's device so barrier and friends never have to
    # infer it from the ambient CUDA context.
    dist.init_process_group(backend="nccl", device_id=target)
    _initialized = True
    return _rank, _local_rank, _world_size


def cleanup():
    """Destroy the process group if it was initialized."""
    global _initialized
    if _initialized:
        dist.destroy_process_group()
        _initialized = False


def rank() -> int:
    """Global rank of this process (0 when not distributed)."""
    return _rank


def local_rank() -> int:
    """GPU-local rank of this process (0 when not distributed)."""
    return _local_rank


def world_size() -> int:
    """Total number of processes (1 when not distributed)."""
    return _world_size


def is_main_process() -> bool:
    """True on rank 0 or when not distributed."""
    return _rank == 0


def is_distributed() -> bool:
    """True only when multiple processes are cooperating."""
    return _world_size > 1


def barrier():
    """Block until all ranks arrive; no-op when not distributed."""
    if _initialized:
        dist.barrier()


def all_reduce_mean(tensor: Tensor) -> Tensor:
    """In-place average across ranks; identity when not distributed."""
    if _initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(_world_size)
    return tensor


def all_reduce_sum(tensor: Tensor) -> Tensor:
    """In-place sum across ranks; identity when not distributed."""
    if _initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_object(obj):
    """Collect one picklable object from every rank; a one-element list when alone."""
    if not _initialized:
        return [obj]
    gathered = [None] * _world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def device() -> torch.device:
    """The CUDA device this process owns."""
    return torch.device("cuda", _local_rank)
