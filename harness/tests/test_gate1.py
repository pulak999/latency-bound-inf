import pytest

from harness.engines.dummy import SleepDistribution
from harness.gate1 import simulate_mailbox


def test_fast_model_serves_every_frame_immediately():
    frames = simulate_mailbox([0.5] * 4, period_s=1.0, n_frames=4)
    assert [f.t_start for f in frames] == [0.0, 1.0, 2.0, 3.0]
    assert all(f.latency == 0.5 for f in frames)


def test_model_slower_than_cadence_waits_then_overwrites():
    # Service 1.5 s at T = 1 s. Frame 0: 0 -> 1.5. Frame 1 waits, 1.5 -> 3.0.
    # Frame 2 waits, 3.0 -> 4.5. Frame 3 arrives at 3.0 and waits; frame 4 arrives
    # at 4.0 while the model is still busy, so frame 3 is overwritten. Frame 4: 4.5 -> 6.0.
    frames = simulate_mailbox([1.5] * 5, period_s=1.0, n_frames=5)
    assert [f.t_start for f in frames] == [0.0, 1.5, 3.0, None, 4.5]
    assert [f.latency for f in frames] == pytest.approx([1.5, 2.0, 2.5, None, 2.0])


def test_overwrite_happens_when_two_frames_arrive_during_one_service():
    frames = simulate_mailbox([2.5, 2.5, 2.5], period_s=1.0, n_frames=4)
    # Frame 0: 0 -> 2.5. Frames 1 and 2 arrive during it; 1 is overwritten by 2.
    # Frame 2 served 2.5 -> 5.0. Frame 3 arrives at 3.0, waits, served 5.0 -> 7.5.
    assert frames[1].t_end is None
    assert [f.t_start for f in frames if f.t_end is not None] == [0.0, 2.5, 5.0]


def test_sleep_draws_are_reproducible_and_match_the_engine_order():
    dist = SleepDistribution("uniform", 0.2, 2.0, seed=7)
    assert dist.draws(5) == dist.draws(5)
    assert dist.draws(5)[:3] == dist.draws(3)
    assert all(0.2 <= d <= 2.0 for d in dist.draws(100))


def test_not_enough_draws_is_an_error():
    with pytest.raises(ValueError):
        simulate_mailbox([0.1], period_s=1.0, n_frames=2)
