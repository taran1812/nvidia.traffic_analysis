import asyncio
import numpy as np


class AsyncBatcher:
    def __init__(self, triton, max_batch: int = 16, drain_interval: float = 0.05, max_queue: int = 500):
        self._triton = triton
        self._max_batch = max_batch
        self._drain_interval = drain_interval
        self._max_queue = max_queue
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._drain_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, frame: np.ndarray) -> tuple[list, float]:
        if self._queue.qsize() >= self._max_queue:
            raise RuntimeError("queue full")
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((frame, future))
        return await future

    async def _drain_loop(self):
        loop = asyncio.get_event_loop()
        while self._running:
            await asyncio.sleep(self._drain_interval)
            if self._queue.empty():
                continue

            items = []
            while not self._queue.empty() and len(items) < self._max_batch:
                items.append(self._queue.get_nowait())

            if not items:
                continue

            frames = np.stack([item[0] for item in items])
            futures = [item[1] for item in items]

            try:
                dets_list, elapsed_ms = await loop.run_in_executor(
                    None, self._triton.infer_batch, frames
                )
                per_item_ms = elapsed_ms / len(items)
                for future, dets in zip(futures, dets_list):
                    if not future.done():
                        future.set_result((dets, per_item_ms))
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
