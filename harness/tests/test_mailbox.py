import asyncio

from harness.mailbox import Frame, Mailbox


def frame(i: int) -> Frame:
    return Frame(frame_id=i, tile_id=f"t{i}", t_obs=float(i), deadline=float(i) + 1)


def test_put_replaces_an_untaken_frame_and_reports_it():
    box = Mailbox()
    assert box.put(frame(0)) is None
    assert box.put(frame(1)) == frame(0)
    assert box.put(frame(2)) == frame(1)
    assert asyncio.run(box.take()) == frame(2)


def test_take_waits_for_a_frame_and_returns_none_after_close():
    async def scenario():
        box = Mailbox()
        taker = asyncio.create_task(box.take())
        await asyncio.sleep(0.01)
        assert not taker.done()
        box.put(frame(0))
        assert await taker == frame(0)
        box.close()
        assert await box.take() is None

    asyncio.run(scenario())


def test_close_delivers_the_last_frame_before_none():
    async def scenario():
        box = Mailbox()
        box.put(frame(0))
        box.close()
        assert await box.take() == frame(0)
        assert await box.take() is None

    asyncio.run(scenario())
