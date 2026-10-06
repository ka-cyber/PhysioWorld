"""Seeded, stateful discrete-event transport, with explicit unavailable data.

Packet timestamps are acquisition seconds. ``transmit`` runs the submitted
batch to completion; later calls start no earlier than the channel's clock.
Queue service, random streams and burst state persist until ``reset()``.
ARQ uses a fixed timeout and instantaneous, reliable acknowledgement on valid
arrival. Delayed ACKs can therefore cause legitimate repeated wire arrivals.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from physioworld.perturbations.network import LatencyJitter, PacketLoss

from ._validation import finite, integer, probability
from .models import ChannelResult, Packet, PacketHeader, TransmissionEvent


def packetize(
    data: bytes,
    payload_size: int,
    *,
    timestamp: float = 0.0,
    sequences: Sequence[int] | None = None,
    timestamps: Sequence[float] | None = None,
    source_id: str = "default",
    modality: str | None = None,
    representation: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Packet, ...]:
    integer(payload_size, "payload_size", 1)
    finite(timestamp, "timestamp", 0)
    if not isinstance(data, bytes):
        raise ValueError("data must be bytes")
    chunks = tuple(
        data[offset : offset + payload_size] for offset in range(0, len(data), payload_size)
    )
    ids = tuple(range(len(chunks))) if sequences is None else tuple(sequences)
    times = (timestamp,) * len(chunks) if timestamps is None else tuple(timestamps)
    if len(ids) != len(chunks) or len(times) != len(chunks):
        raise ValueError("packet IDs and timestamps must match packet count")
    _unique_ids(ids)
    info = dict(metadata or {})
    if "packetization" in info:
        raise ValueError("packetization metadata is reserved")
    info["packetization"] = {"sequences": list(ids), "data_length": len(data)}
    return tuple(
        Packet(
            seq,
            chunk,
            time,
            info,
            PacketHeader(1, source_id, seq, time, modality, representation, len(chunk)),
        )
        for seq, chunk, time in zip(ids, chunks, times)
    )


def _unique_ids(ids: Sequence[int]) -> None:
    for sequence in ids:
        integer(sequence, "sequence")
    if len(set(ids)) != len(ids):
        raise ValueError("packet sequence IDs must be unique")


@dataclass(frozen=True)
class GilbertElliottLoss:
    """Two-state Markov loss model, emitting before transitioning each attempt.

    Transition probabilities are per transmission, not per elapsed second.
    Independent good/bad emission probabilities distinguish this model from
    a fixed run-length burst mask.
    """

    good_to_bad: float
    bad_to_good: float
    good_loss_probability: float = 0.0
    bad_loss_probability: float = 1.0
    initial_state: str = "good"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("good_to_bad", "bad_to_good", "good_loss_probability", "bad_loss_probability"):
            probability(getattr(self, name), name)
        if self.initial_state not in {"good", "bad"}:
            raise ValueError("initial_state must be good or bad")
        integer(self.seed, "seed")


@dataclass
class ErasureChannel:
    loss: PacketLoss | GilbertElliottLoss
    latency: LatencyJitter = field(default_factory=lambda: LatencyJitter(0.0))
    erasure_probability: float = 0.0
    duplicate_probability: float = 0.0
    duplicate_delay: float = 0.0
    reorder_probability: float = 0.0
    reorder_delay: float = 0.01
    corruption_probability: float = 0.0
    service_time: float = 0.0
    bytes_per_second: float | None = None
    queue_capacity: int | None = None
    deadline: float | None = None
    max_retries: int = 0
    retry_timeout: float = 1.0
    seed: int | None = None
    wire_transform: Callable[[bytes, Packet, int], bytes] | None = None
    _clock: float = field(init=False, default=0.0, repr=False)
    _busy_until: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self.reset()

    def _validate(self) -> None:
        if not isinstance(self.loss, (PacketLoss, GilbertElliottLoss)):
            raise ValueError("loss must be PacketLoss or GilbertElliottLoss")
        if isinstance(self.loss, PacketLoss):
            probability(self.loss.probability, "loss probability")
            integer(self.loss.burst_length, "burst_length", 1)
        else:
            self.loss.__post_init__()
        integer(self.loss.seed, "loss seed")
        if not isinstance(self.latency, LatencyJitter):
            raise ValueError("latency must be LatencyJitter")
        finite(self.latency.latency, "latency", 0)
        finite(self.latency.jitter, "jitter", 0)
        integer(self.latency.seed, "latency seed")
        for name in (
            "erasure_probability",
            "duplicate_probability",
            "reorder_probability",
            "corruption_probability",
        ):
            probability(getattr(self, name), name)
        for name in ("duplicate_delay", "reorder_delay", "service_time"):
            finite(getattr(self, name), name, 0)
        if self.bytes_per_second is not None:
            if finite(self.bytes_per_second, "bytes_per_second", 0) == 0:
                raise ValueError("bytes_per_second must be positive")
        if self.queue_capacity is not None:
            integer(self.queue_capacity, "queue_capacity")
        if self.deadline is not None:
            finite(self.deadline, "deadline", 0)
        integer(self.max_retries, "max_retries")
        if finite(self.retry_timeout, "retry_timeout", 0) == 0:
            raise ValueError("retry_timeout must be positive")
        if self.seed is not None:
            integer(self.seed, "seed")
        if self.wire_transform is not None and not callable(self.wire_transform):
            raise ValueError("wire_transform must be callable")

    @property
    def time(self) -> float:
        return self._clock

    @property
    def channel_state(self) -> str:
        return self._state

    def reset(self, *, seed: int | None = None) -> None:
        """Replay from a clean queue/clock/burst state; optional seed overrides configs."""
        if seed is not None:
            integer(seed, "seed")
            self.seed = seed
        self._validate()
        base = self.seed
        self._loss_rng = random.Random(self.loss.seed if base is None else base)
        self._latency_rng = random.Random(self.latency.seed if base is None else base + 1)
        self._effects_rng = random.Random(0 if base is None else base + 2)
        self._state = (
            self.loss.initial_state if isinstance(self.loss, GilbertElliottLoss) else "iid"
        )
        self._burst_remaining = 0
        self._clock = self._busy_until = 0.0
        self._service_finishes: list[float] = []

    def _loss_draw(self) -> tuple[bool, str]:
        state = self._state
        if isinstance(self.loss, GilbertElliottLoss):
            p = (
                self.loss.good_loss_probability
                if state == "good"
                else self.loss.bad_loss_probability
            )
            lost = self._loss_rng.random() < p
            transition = self._loss_rng.random()
            if state == "good" and transition < self.loss.good_to_bad:
                self._state = "bad"
            elif state == "bad" and transition < self.loss.bad_to_good:
                self._state = "good"
            return lost, state
        triggered = self._loss_rng.random() < self.loss.probability
        if triggered:
            self._burst_remaining = max(self._burst_remaining, self.loss.burst_length)
        lost = self._burst_remaining > 0
        self._burst_remaining = max(0, self._burst_remaining - 1)
        return lost, state

    def transmit(
        self,
        packets: Sequence[Packet],
        *,
        deadline: float | None = None,
        deadlines: Mapping[int, float] | None = None,
        ready_times: Mapping[int, float] | None = None,
    ) -> ChannelResult:
        """Transmit unique input IDs; wire duplicates/retries are explicitly allowed.

        ``self.deadline`` is a relative acquisition-to-arrival budget. The call's
        ``deadline`` and ``deadlines`` are absolute simulation times and further
        constrain that budget. Arrival exactly on the deadline is accepted.
        Queue capacity counts packets in service plus waiting for service.
        ``ready_times`` declares computation/resource availability in absolute
        seconds without changing the packet's original acquisition timestamp.
        ``received`` is arrival-ordered and includes identical wire duplicates;
        use PacketReceiver for idempotent acceptance before reconstruction.
        """
        self._validate()
        sent = tuple(packets)
        if any(not isinstance(packet, Packet) for packet in sent):
            raise ValueError("channel inputs must be packets")
        _unique_ids(tuple(packet.sequence for packet in sent))
        if deadline is not None:
            finite(deadline, "absolute deadline", 0)
        limits = dict(deadlines or {})
        if set(limits) - {packet.sequence for packet in sent}:
            raise ValueError("deadline map contains unknown sequence IDs")
        for value in limits.values():
            finite(value, "absolute deadline", 0)
        ready = dict(ready_times or {})
        if set(ready) - {packet.sequence for packet in sent}:
            raise ValueError("ready-time map contains unknown sequence IDs")
        for packet in sent:
            if packet.sequence in ready:
                finite(ready[packet.sequence], "ready time", packet.timestamp)
        end_times: dict[int, float | None] = {}
        frames = {packet.sequence: packet.serialize() for packet in sent}
        for packet in sent:
            values = [
                v
                for v in (
                    deadline,
                    limits.get(packet.sequence),
                    packet.timestamp + self.deadline if self.deadline is not None else None,
                )
                if v is not None
            ]
            end_times[packet.sequence] = min(values) if values else None
            limit = end_times[packet.sequence]
            if limit is not None:
                finite(limit, "computed deadline", 0)

        # Arrivals precede departures, then queue admissions at equal time.
        pending: list[tuple[float, int, int, str, Any]] = []
        counter = 0

        def schedule(time: float, kind: str, data: Any) -> None:
            nonlocal counter
            finite(time, "event time", 0)
            counter += 1
            priority = {"arrival": 0, "send": 1, "ready": 2}[kind]
            heapq.heappush(pending, (time, priority, counter, kind, data))

        for packet in sent:
            schedule(
                max(self._clock, packet.timestamp, ready.get(packet.sequence, packet.timestamp)),
                "ready",
                (packet, 0),
            )
        acknowledged: set[int] = set()
        received: list[Packet] = []
        events: list[TransmissionEvent] = []
        delays: dict[int, float] = {}
        arrivals: dict[int, float] = {}
        wire_bytes = 0
        while pending:
            time, _, _, kind, data = heapq.heappop(pending)
            if kind == "ready" and data[0].sequence in acknowledged:
                # Cancelled ARQ timers do not artificially advance the clock.
                continue
            self._clock = max(self._clock, time)
            if kind == "arrival":
                packet, attempt, departure, frame, state, duplicate = data
                limit = end_times[packet.sequence]
                recovered = None
                try:
                    recovered = Packet.deserialize(frame)
                    if recovered.serialize() != frames[packet.sequence]:
                        raise ValueError("wire frame changed source identity or payload")
                    status = "deadline" if limit is not None and time > limit else "delivered"
                except ValueError:
                    status = "corrupt"
                if status == "delivered":
                    if recovered is None:
                        raise RuntimeError("delivered event did not contain a recovered packet")
                    acknowledged.add(packet.sequence)
                    received.append(recovered)
                    if packet.sequence not in delays:
                        # Object-associated acquisition time, independent of numeric ID.
                        delays[packet.sequence] = time - packet.timestamp
                        arrivals[packet.sequence] = time
                else:
                    recovered = None
                events.append(
                    TransmissionEvent(
                        packet.sequence,
                        attempt,
                        departure,
                        time,
                        status,
                        recovered,
                        state,
                        duplicate,
                    )
                )
                continue

            packet, attempt = data
            if kind == "ready":
                if packet.sequence in acknowledged:
                    continue
                limit = end_times[packet.sequence]
                if limit is not None and time > limit:
                    events.append(
                        TransmissionEvent(packet.sequence, attempt, time, None, "expired")
                    )
                    continue
                while self._service_finishes and self._service_finishes[0] <= time:
                    heapq.heappop(self._service_finishes)
                if (
                    self.queue_capacity is not None
                    and len(self._service_finishes) >= self.queue_capacity
                ):
                    events.append(
                        TransmissionEvent(packet.sequence, attempt, time, None, "queue_full")
                    )
                    if attempt < self.max_retries:
                        schedule(time + self.retry_timeout, "ready", (packet, attempt + 1))
                    continue
                duration = self.service_time
                if self.bytes_per_second is not None:
                    duration += len(frames[packet.sequence]) / self.bytes_per_second
                departure = max(time, self._busy_until) + duration
                finite(departure, "departure", 0)
                if limit is not None and departure > limit:
                    events.append(
                        TransmissionEvent(packet.sequence, attempt, time, None, "expired")
                    )
                    continue
                self._busy_until = departure
                heapq.heappush(self._service_finishes, departure)
                # Once admitted, a queued attempt is committed. Emission and
                # Markov transitions occur at actual departure, not admission.
                schedule(departure, "send", (packet, attempt))
                continue

            departure = time
            lost, state = self._loss_draw()
            wire_bytes += len(frames[packet.sequence])
            # Avoid uniform(-jitter, jitter)'s overflowing interval subtraction.
            delay = max(
                0.0,
                self.latency.latency + (2 * self._latency_rng.random() - 1) * self.latency.jitter,
            )
            if self._effects_rng.random() < self.reorder_probability:
                delay += self.reorder_delay
            finite(delay, "sampled delay", 0)
            arrival = finite(departure + delay, "arrival", 0)
            if lost:
                events.append(
                    TransmissionEvent(
                        packet.sequence, attempt, departure, None, "lost", channel_state=state
                    )
                )
            elif self._effects_rng.random() < self.erasure_probability:
                events.append(
                    TransmissionEvent(
                        packet.sequence, attempt, departure, arrival, "erased", channel_state=state
                    )
                )
            else:
                frame = frames[packet.sequence]
                if self._effects_rng.random() < self.corruption_probability:
                    corrupted = bytearray(frame)
                    corrupted[self._effects_rng.randrange(len(corrupted))] ^= 1
                    frame = bytes(corrupted)
                if self.wire_transform is not None:
                    frame = self.wire_transform(frame, packet, attempt)
                    if not isinstance(frame, bytes):
                        raise ValueError("wire_transform must return bytes")
                schedule(arrival, "arrival", (packet, attempt, departure, frame, state, False))
                if self._effects_rng.random() < self.duplicate_probability:
                    wire_bytes += len(frame)
                    schedule(
                        arrival + self.duplicate_delay,
                        "arrival",
                        (packet, attempt, departure, frame, state, True),
                    )
            if attempt < self.max_retries:
                schedule(departure + self.retry_timeout, "ready", (packet, attempt + 1))

        # Loss/erasure events also advance the blocking simulation clock.
        self._clock = max([self._clock] + [e.arrival or e.departure for e in events])
        events.sort(
            key=lambda event: event.arrival if event.arrival is not None else event.departure
        )
        return ChannelResult(
            sent,
            tuple(received),
            tuple(p.sequence for p in sent if p.sequence not in acknowledged),
            delays,
            tuple(events),
            arrivals,
            wire_bytes,
        )


def reconstruct(
    packets: Iterable[Packet],
    *,
    expected_count: int | None = None,
    expected_sequences: Sequence[int] | None = None,
) -> bytes:
    """Reassemble exactly the declared source order; missing data never become zeros.

    Packetization manifests detect trailing erasures. Legacy packets without a
    manifest default to contiguous IDs; non-contiguous streams need explicit
    ``expected_sequences``. Repeated inputs are invalid, even if identical.
    """
    source = tuple(packets)
    _unique_ids(tuple(packet.sequence for packet in source))
    if expected_count is not None:
        integer(expected_count, "expected_count")
        if len(source) != expected_count:
            raise ValueError("cannot reconstruct erased data without a decoder")
    manifests = [p.metadata.get("packetization") for p in source]
    manifest = next((value for value in manifests if value is not None), None)
    if manifest is not None and any(value != manifest for value in manifests):
        raise ValueError("inconsistent packetization manifests")
    if manifest is not None:
        if not isinstance(manifest, dict) or set(manifest) != {"sequences", "data_length"}:
            raise ValueError("invalid packetization manifest")
        if not isinstance(manifest["sequences"], list):
            raise ValueError("manifest sequences must be a list")
        integer(manifest["data_length"], "manifest data_length")
        _unique_ids(manifest["sequences"])
    if expected_sequences is not None:
        expected = tuple(expected_sequences)
    elif manifest is not None:
        try:
            expected = tuple(manifest["sequences"])
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid packetization manifest") from exc
    elif expected_count is not None:
        expected = tuple(range(expected_count))
    else:
        # Never allocate proportional to an arbitrary (possibly enormous) ID.
        expected = tuple(range(len(source)))
    _unique_ids(expected)
    if expected_count is not None and len(expected) != expected_count:
        raise ValueError("expected count disagrees with source manifest")
    actual = {packet.sequence: packet for packet in source}
    if set(expected) != set(actual):
        raise ValueError("cannot reconstruct erased data without a decoder")
    result = b"".join(actual[sequence].payload for sequence in expected)
    if manifest is not None:
        if tuple(manifest["sequences"]) != expected or manifest.get("data_length") != len(result):
            raise ValueError("source length/order disagrees with packetization manifest")
    return result
