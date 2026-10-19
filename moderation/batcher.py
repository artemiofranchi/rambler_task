"""Микробатчинг: одиночные запросы копятся до batch_max_size или batch_max_wait_ms и идут в модель одним батчем.

Трансформер на батче из 64 текстов почти так же быстр, как на одном, — под конкурентной нагрузкой
это кратно поднимает throughput, а задержка вырастает максимум на batch_max_wait_ms.
"""
import asyncio


class MicroBatcher:
    def __init__(self, fn, max_size: int = 64, max_wait_ms: float = 5.0):
        self.fn = fn
        self.max_size = max_size
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            items = [item for item, _ in batch]
            try:
                # модель работает в отдельном потоке, event loop продолжает принимать запросы
                results = await asyncio.to_thread(self.fn, items)
            except Exception as exc:  # ошибка батча не должна ронять цикл
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
