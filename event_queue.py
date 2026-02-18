import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(order=True)
class AgentEvent:
    """An event from any external source that the agent should process."""
    priority: int = field(compare=True, default=5)
    source: str = field(compare=False, default="")
    event_type: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)
    timestamp: str = field(compare=False, default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventQueue:
    """Thread-safe priority queue for external events."""

    def __init__(self):
        self._queue: queue.PriorityQueue[AgentEvent] = queue.PriorityQueue()
        self._listeners: list = []
        self._lock = threading.Lock()

    def put(self, event: AgentEvent):
        """Add an event to the queue."""
        self._queue.put(event)

    def get_all_pending(self) -> list[AgentEvent]:
        """Drain all pending events from the queue and return them sorted by priority."""
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def has_pending(self) -> bool:
        """Check if there are pending events without consuming them."""
        return not self._queue.empty()

    def register_listener(self, listener):
        """Register a listener to be started as a daemon thread."""
        with self._lock:
            self._listeners.append(listener)

    def start_listeners(self):
        """Start all registered listeners as daemon threads."""
        for listener in self._listeners:
            thread = threading.Thread(
                target=listener.run,
                args=(self,),
                daemon=True,
                name=f"listener-{listener.name}",
            )
            thread.start()

    def stop_listeners(self):
        """Signal all listeners to stop."""
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception:
                pass
