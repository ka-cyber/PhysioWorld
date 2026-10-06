"""Typing protocols for extension points."""

from typing import Any, Protocol, Sequence


class Transform(Protocol):
    def apply(self, stream: Any) -> Any: ...


class Representation(Protocol):
    name: str

    def transform(self, history: Sequence[float]) -> Sequence[float]: ...


class Policy(Protocol):
    name: str

    def action(self, state: Any, available_actions: Sequence[Any]) -> Any: ...
