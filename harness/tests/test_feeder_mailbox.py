import asyncio
import time

from harness.feeder import run_mailbox
from harness.mailbox import Frame
from harness.records import RequestRecord

STATIC = {"engine": "test", "model": "m", "quant": "q", "power_mode": "p"}


def record_for(frame: Frame, t_send: float, t_end: float) -> RequestRecord:
    return RequestRecord(
        req_id=frame.frame_id,
        t_arrival=frame.t_obs,
        t_send=t_send,
        t_first_token=t_end,
        t_last_token=t_end,
        deadline=frame.deadline,
        n_prompt_tokens=None,
        n_output_tokens=None,
        output_text="keep",
        engine_prompt_ms=None,
        engine_decode_ms=None,
        tile_id=frame.tile_id,
        **STATIC,
    )


def test_slow_model_leaves_overwritten_frames_and_keeps_every_frame_id():
    period = 0.05

    async def slow(frame: Frame) -> RequestRecord:
        t_send = time.monotonic()
        await asyncio.sleep(period * 2.5)  # each frame takes 2.5 periods
        return record_for(frame, t_send, time.monotonic())

    run = asyncio.run(
        run_mailbox(
            slow,
            period,
            n_frames=10,
            tile_ids=["a", "b"],
            overwritten_fields=STATIC,
            start_delay_s=0.0,
        )
    )

    assert [r.req_id for r in run.records] == list(range(10))
    assert run.frames_emitted == 10
    outcomes = [r.outcome for r in run.records]
    assert outcomes[0] == "ok"
    assert "overwritten" in outcomes
    assert all(r.t_send is None and r.missed for r in run.records if r.outcome == "overwritten")
    assert [r.tile_id for r in run.records] == ["a", "b"] * 5
    # Frames never wait more than one period in a depth-1 mailbox.
    assert all(r.waiting_s <= period * 1.1 for r in run.records if r.outcome == "ok")


def test_fast_model_serves_every_frame_on_time():
    period = 0.03

    async def fast(frame: Frame) -> RequestRecord:
        t_send = time.monotonic()
        await asyncio.sleep(0.005)
        return record_for(frame, t_send, time.monotonic())

    run = asyncio.run(
        run_mailbox(
            fast, period, n_frames=8, tile_ids=["a"], overwritten_fields=STATIC, start_delay_s=0.0
        )
    )
    assert all(r.outcome == "ok" and not r.missed for r in run.records)
    assert run.emitter_faults == []


def test_stop_at_cuts_emission_short_and_fills_the_run_in_place():
    from harness.feeder import MailboxRun

    async def fast(frame: Frame) -> RequestRecord:
        return record_for(frame, time.monotonic(), time.monotonic())

    run = MailboxRun()
    period = 0.02
    result = asyncio.run(
        run_mailbox(
            fast,
            period,
            n_frames=100,
            tile_ids=["a"],
            overwritten_fields=STATIC,
            start_delay_s=0.0,
            stop_at=time.monotonic() + period * 5.5,
            run=run,
        )
    )
    assert result is run
    assert 3 <= run.frames_emitted <= 7
