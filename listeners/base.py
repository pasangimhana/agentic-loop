from abc import ABC, abstractmethod


class BaseListener(ABC):
    """Base class for all event source listeners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this listener."""
        ...

    @abstractmethod
    def run(self, event_queue) -> None:
        """Main loop — runs in a daemon thread. Push AgentEvents to event_queue."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Signal the listener to shut down gracefully."""
        ...
