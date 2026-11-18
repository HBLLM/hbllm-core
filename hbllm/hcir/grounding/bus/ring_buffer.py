from __future__ import annotations

import collections
import threading
from typing import Generic, TypeVar

from hbllm.hcir.grounding.contracts.enums import OverflowPolicy

T = TypeVar("T")


class BufferOverflowError(RuntimeError):
    """Raised when a buffer is full and policy is REJECT_WITH_ERROR."""


class BoundedRingBuffer(Generic[T]):
    """Thread-safe bounded ring buffer supporting configurable overflow policies.

    Supports DROP_OLDEST (latest-wins decimation), DROP_NEWEST, BLOCK_PRODUCER,
    and REJECT_WITH_ERROR with explicit telemetry of dropped packets.
    """

    def __init__(
        self,
        capacity: int,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._overflow_policy = overflow_policy
        self._buffer: collections.deque[T] = collections.deque()
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self._total_enqueued = 0
        self._total_dequeued = 0
        self._dropped_count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overflow_policy(self) -> OverflowPolicy:
        return self._overflow_policy

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    def push(self, item: T, timeout: float | None = None) -> bool:
        """Push an item into the ring buffer.

        Returns:
            True if item was successfully stored, False if discarded by DROP_NEWEST
            or timeout in BLOCK_PRODUCER.
        """
        with self._lock:
            # Check if buffer is full
            while len(self._buffer) >= self._capacity:
                if self._overflow_policy == OverflowPolicy.DROP_OLDEST:
                    self._buffer.popleft()
                    self._dropped_count += 1
                    break
                elif self._overflow_policy == OverflowPolicy.DROP_NEWEST:
                    self._dropped_count += 1
                    return False
                elif self._overflow_policy == OverflowPolicy.REJECT_WITH_ERROR:
                    self._dropped_count += 1
                    raise BufferOverflowError(f"BoundedRingBuffer full (capacity={self._capacity})")
                elif self._overflow_policy == OverflowPolicy.BLOCK_PRODUCER:
                    got_space = self._not_full.wait(timeout=timeout)
                    if not got_space:
                        self._dropped_count += 1
                        return False
                else:
                    self._buffer.popleft()
                    self._dropped_count += 1
                    break

            self._buffer.append(item)
            self._total_enqueued += 1
            self._not_empty.notify()
            return True

    def pop(self, timeout: float | None = None) -> T | None:
        """Pop the oldest item from the buffer, blocking up to timeout seconds."""
        with self._lock:
            while len(self._buffer) == 0:
                if timeout is not None and timeout <= 0:
                    return None
                got_item = self._not_empty.wait(timeout=timeout)
                if not got_item and len(self._buffer) == 0:
                    return None

            item = self._buffer.popleft()
            self._total_dequeued += 1
            self._not_full.notify()
            return item

    def pop_all(self) -> list[T]:
        """Pop all available items in the buffer without blocking."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            self._total_dequeued += len(items)
            self._not_full.notify_all()
            return items

    def peek(self) -> T | None:
        """Peek at the oldest item without removing it."""
        with self._lock:
            return self._buffer[0] if self._buffer else None

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._not_full.notify_all()
