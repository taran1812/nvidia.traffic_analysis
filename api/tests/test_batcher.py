import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock
from api.batcher import AsyncBatcher


def make_frame():
    return np.zeros((3, 640, 640), dtype=np.float32)


@pytest.mark.asyncio
async def test_single_submit_resolves():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[]], 5.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=0.01)
    await batcher.start()
    dets, ms = await batcher.submit(make_frame())
    assert dets == []
    assert isinstance(ms, float)
    await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_submits_all_resolve():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[], [], []], 10.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=0.05)
    await batcher.start()
    # Enqueue all before drain fires
    tasks = [asyncio.create_task(batcher.submit(make_frame())) for _ in range(3)]
    await asyncio.sleep(0)  # yield to let tasks enqueue
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    for dets, ms in results:
        assert isinstance(dets, list)
    assert mock_triton.infer_batch.call_count == 1
    assert mock_triton.infer_batch.call_args[0][0].shape == (3, 3, 640, 640)
    await batcher.stop()


@pytest.mark.asyncio
async def test_queue_full_raises():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[]], 5.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=999, max_queue=2)
    await batcher.start()
    loop = asyncio.get_running_loop()
    await batcher._queue.put((make_frame(), loop.create_future()))
    await batcher._queue.put((make_frame(), loop.create_future()))
    with pytest.raises(RuntimeError, match="queue full"):
        await batcher.submit(make_frame())
    await batcher.stop()
