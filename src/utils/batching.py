# src/utils/batching.py

from typing import Iterable, List, Generator


def batch_iter(iterable: Iterable, batch_size: int) -> Generator[List, None, None]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
