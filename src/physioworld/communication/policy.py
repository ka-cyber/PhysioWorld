"""Adaptive communication protection driven by declared downstream signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ._validation import finite, integer, probability
from .models import Packet
from .protection import SemanticPriorityEncoder


@dataclass(frozen=True)
class ProtectionContext:
    piv: float
    urgency: float
    uncertainty: float
    resource_fraction: float
    channel_loss_probability: float

    def __post_init__(self) -> None:
        for name in ("piv", "urgency", "uncertainty"):
            finite(getattr(self, name), name, 0)
        probability(self.resource_fraction, "resource_fraction")
        probability(self.channel_loss_probability, "channel_loss_probability")


@dataclass(frozen=True)
class RedundancyPolicy:
    minimum: int = 1
    maximum: int = 8

    def __post_init__(self) -> None:
        integer(self.minimum, "minimum", 1)
        integer(self.maximum, "maximum", self.minimum)

    def choose(self, context: ProtectionContext) -> int:
        if not isinstance(context, ProtectionContext):
            raise ValueError("protection context is required")
        if context.resource_fraction == 0 or context.piv == 0:
            return self.minimum
        score = (
            context.piv
            * (1 + context.urgency)
            * (1 + context.uncertainty)
            * (1 + context.channel_loss_probability)
        )
        score *= max(0.0, min(1.0, context.resource_fraction))
        # Saturate before integer conversion, including finite-input overflow.
        if score >= 1:
            return self.maximum
        return self.minimum + round(score * (self.maximum - self.minimum))

    def protect(
        self,
        packets: Sequence[Packet],
        contexts: Mapping[int, ProtectionContext],
        packet_budget: int,
        *,
        byte_budget: int | None = None,
        generation_id: str = "default",
    ) -> tuple[Packet, ...]:
        """Turn adaptive copy decisions into transmitted payloads under hard budgets.

        ``minimum`` is a requested copy floor, not an exemption from budgets.
        Insufficient resources may leave sources unavailable, as exposed by the
        receiver. Each call is an independent protected generation.
        """
        ids = {packet.sequence for packet in packets}
        if set(contexts) != ids:
            raise ValueError("contexts must match source IDs")
        limits = {sequence: self.choose(context) for sequence, context in contexts.items()}
        encoder = SemanticPriorityEncoder(tuple(packets), limits, generation_id, limits)
        return encoder.encode(packet_budget, byte_budget=byte_budget)
