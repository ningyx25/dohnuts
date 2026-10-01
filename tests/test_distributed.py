"""Single-process identities and data-parallel partitioning of the launcher.

A plain ``python`` launch must behave exactly like the original single-GPU
code, and each rank must read a disjoint slice of a step's global mixture.
"""

import random

import pytest
import torch
import torch.distributed as dist

from dohnuts import distributed
from dohnuts.train import reduce_consumed
from dohnuts.training_data import TrainingBatches, microbatch_index


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    """Every test starts from the un-launched environment and restores it after."""
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        monkeypatch.delenv(name, raising=False)
    yield
    monkeypatch.undo()


def test_setup_without_a_launcher_is_single_process():
    assert distributed.setup() == (0, 0, 1)
    assert not distributed.is_distributed()
    assert distributed.is_main_process()
    assert distributed.world_size() == 1
    assert distributed.rank() == 0
    assert distributed.local_rank() == 0


def test_world_size_one_env_is_still_single_process(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert distributed.setup() == (0, 0, 1)
    assert not distributed.is_distributed()


def test_device_follows_the_local_rank():
    assert distributed.device() == torch.device("cuda", 0)


def test_collectives_are_identity_when_not_distributed():
    mean = torch.tensor([1.0, 3.0])
    assert distributed.all_reduce_mean(mean) is mean
    torch.testing.assert_close(mean, torch.tensor([1.0, 3.0]))
    total = torch.tensor([1.0, 3.0])
    assert distributed.all_reduce_sum(total) is total
    torch.testing.assert_close(total, torch.tensor([1.0, 3.0]))
    assert distributed.all_gather_object({"rank": 0}) == [{"rank": 0}]


def test_barrier_and_cleanup_without_a_process_group():
    distributed.barrier()
    distributed.cleanup()
    assert not distributed.is_distributed()


def test_collectives_against_a_real_gloo_group(tmp_path, monkeypatch):
    # Exercise the initialized path (the branch a real run takes). A one-rank
    # gloo group is enough to prove the reductions and gather actually run.
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path}/rendezvous",
        rank=0,
        world_size=1,
    )
    monkeypatch.setattr(distributed, "_initialized", True)
    monkeypatch.setattr(distributed, "_world_size", 1)
    try:
        mean = torch.tensor([1.0, 3.0])
        assert distributed.all_reduce_mean(mean) is mean
        torch.testing.assert_close(mean, torch.tensor([1.0, 3.0]))
        total = torch.tensor([1.0, 3.0])
        assert distributed.all_reduce_sum(total) is total
        torch.testing.assert_close(total, torch.tensor([1.0, 3.0]))
        assert distributed.all_gather_object("rank0") == ["rank0"]
        distributed.barrier()
    finally:
        dist.destroy_process_group()


def test_reduce_consumed_sums_disjoint_rank_counts(monkeypatch):
    monkeypatch.setattr(distributed, "device", lambda: torch.device("cpu"))
    consumed = {"gui": 3, "ac": 5}
    assert reduce_consumed(consumed) == {"ac": 5, "gui": 3}


def test_microbatch_index_with_one_rank_is_the_original_offset():
    for start_step in (0, 7):
        for index in range(5):
            expected = start_step * 3 + index
            assert (
                microbatch_index(index, start_step=start_step, accumulation=3, rank=0, world_size=1)
                == expected
            )


def test_microbatch_index_partitions_each_step_across_ranks():
    accumulation, world_size = 3, 4
    for step in range(3):
        seen = []
        for rank in range(world_size):
            for within in range(accumulation):
                index = step * accumulation + within
                seen.append(
                    microbatch_index(
                        index,
                        start_step=0,
                        accumulation=accumulation,
                        rank=rank,
                        world_size=world_size,
                    )
                )
        # One step consumes a contiguous global block, with no rank overlap.
        assert sorted(seen) == list(
            range(step * accumulation * world_size, (step + 1) * accumulation * world_size)
        )
        assert len(set(seen)) == accumulation * world_size


def test_microbatch_index_resumes_at_the_right_global_step():
    # start_step must advance the global mixture by whole world-sized steps.
    assert (
        microbatch_index(0, start_step=5, accumulation=2, rank=2, world_size=4) == 5 * 2 * 4 + 2 * 2
    )


class RecordingCollator:
    """Stand-in collator: the batch is its own IDs plus the permutation seed."""

    def __call__(self, rows, *, permutation_seed=None):
        return [row["id"] for row in rows], permutation_seed


def dataset(*, rank=0, world_size=1, start_step=0):
    groups = {
        "gui": [{"id": f"gui-{i}"} for i in range(4)],
        "ac": [{"id": f"ac-{i}"} for i in range(4)],
    }
    return TrainingBatches(
        groups,
        RecordingCollator(),
        seed=42,
        batch_size=2,
        steps=4,
        accumulation=2,
        start_step=start_step,
        rank=rank,
        world_size=world_size,
    )


def test_single_rank_dataset_reproduces_the_original_indexing():
    batches = dataset()
    assert len(batches) == 8
    for index in range(len(batches)):
        rng = random.Random(42 * 1000000007 + index)
        key = sorted(["gui", "ac"])[rng.randrange(2)]
        rows = [rng.choice(batches.groups[key]) for _ in range(2)]
        expected_ids = [row["id"] for row in rows]
        expected_seed = rng.randrange(2**63)
        ids, permutation_seed = batches[index][0]
        assert ids == expected_ids
        assert permutation_seed == expected_seed


def test_ranks_never_share_a_batch_within_a_step():
    # Distinct global indices give distinct permutations and (almost surely)
    # distinct example draws; the guarantee is the disjoint index set itself.
    accumulation, world_size = 2, 3
    step = 1
    per_rank = []
    for rank in range(world_size):
        rank_batches = dataset(rank=rank, world_size=world_size)
        per_rank.append(
            [rank_batches[step * accumulation + within][0][1] for within in range(accumulation)]
        )
    flat = [seed for seeds in per_rank for seed in seeds]
    assert len(set(flat)) == accumulation * world_size


def test_resume_keeps_each_ranks_slice_aligned():
    # Resuming at step S must be indistinguishable from training the same
    # world through step 0: local index i maps to the same global microbatch as
    # local index S*accumulation + i of a run that started at zero.
    world_size, accumulation, start_step = 2, 2, 3
    for rank in range(world_size):
        resumed = dataset(rank=rank, world_size=world_size, start_step=start_step)
        fresh = dataset(rank=rank, world_size=world_size, start_step=0)
        assert len(resumed) == (4 - start_step) * accumulation
        for index in range(len(resumed)):
            assert resumed[index] == fresh[start_step * accumulation + index]


def test_the_collective_timeout_outlives_torchs_ten_minute_default():
    """A rank that evaluates a large split alone holds the others at a barrier that long.

    The ac-jev-v1 baseline pass on a 9B model was killed by torch's 600 s default, so the
    group timeout is the run's to choose and it is recorded in the environment metrics.
    """
    assert distributed.NCCL_TIMEOUT_S > 600
    assert distributed.timeout_seconds() is None  # no process group, no collective to time out


@pytest.mark.parametrize("shards", [1, 2, 3, 4, 5, 7])
def test_evaluation_shards_rebuild_the_single_process_batch_sequence(shards):
    """Each rank owns one contiguous block, so rank-order concatenation restores the order.

    That is the whole reason a sharded pass writes the same file, in the same order, with
    the same per-row padding as one process: the batch sequence is sliced, never rebuilt.
    """
    from dohnuts.training_data import EvaluationBatches

    groups = {
        "a": [{"id": f"a{i}", "dataset": "a", "state": {}, "question": {}} for i in range(9)],
        "b": [{"id": f"b{i}", "dataset": "b", "state": {}, "question": {}} for i in range(5)],
    }
    whole = EvaluationBatches(groups, None, 4)
    # 9 rows of a and 5 of b in batches of four: three batches plus two.
    assert len(whole) == 5
    assert whole.rows == 14
    assert whole.total_rows == 14

    parts = [EvaluationBatches(groups, None, 4, shard=r, shards=shards) for r in range(shards)]
    assert [batch for part in parts for batch in part.batches] == whole.batches
    assert sum(part.rows for part in parts) == whole.rows
    assert {key: sum(part.rows_by_dataset[key] for part in parts) for key in ("a", "b")} == {
        "a": 9,
        "b": 5,
    }
