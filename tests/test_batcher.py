import asyncio

from moderation.batcher import MicroBatcher


def test_concurrent_requests_are_batched():
    sizes = []

    def fn(items):
        sizes.append(len(items))
        return [x * 2 for x in items]

    async def run():
        b = MicroBatcher(fn, max_size=16, max_wait_ms=20)
        await b.start()
        results = await asyncio.gather(*[b.submit(i) for i in range(40)])
        await b.stop()
        return results

    assert asyncio.run(run()) == [i * 2 for i in range(40)]
    assert len(sizes) < 40 and max(sizes) <= 16  # запросы объединились в батчи


def test_errors_propagate_and_loop_survives():
    def fn(items):
        if "bad" in items:
            raise ValueError("boom")
        return items

    async def run():
        b = MicroBatcher(fn, max_size=4, max_wait_ms=1)
        await b.start()
        try:
            await b.submit("bad")
        except ValueError:
            pass
        ok = await b.submit("ok")
        await b.stop()
        return ok

    assert asyncio.run(run()) == "ok"
