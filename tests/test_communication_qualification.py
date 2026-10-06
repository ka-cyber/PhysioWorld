"""Dependency-free communication qualification, including independent GF references.

These are software/numerical verification fixtures and synthetic simulations,
not empirical physiological validation. Randomized sweeps use frozen seeds.
"""

from __future__ import annotations

import json
import math
import random
import runpy
import struct
import unittest
import zlib
from dataclasses import replace
from pathlib import Path

from physioworld.communication import (
    GF256,
    Codeword,
    CommunicationMetrics,
    ErasureChannel,
    EvidenceClass,
    GilbertElliottLoss,
    LatencyJitter,
    Packet,
    PacketBuffer,
    PacketHeader,
    PacketLoss,
    PacketReceiver,
    ParityDecoder,
    ParityEncoder,
    ProtectionContext,
    RedundancyPolicy,
    RLNCDecoder,
    RLNCEncoder,
    RLNCPacketDecoder,
    RLNCPacketEncoder,
    ScientificContext,
    SemanticPriorityDecoder,
    SemanticPriorityEncoder,
    UniformQuantizer,
    evaluate_semantics,
    mean_squared_distortion,
    packetize,
    permute_payloads,
    reconstruct,
)


def polynomial_product(a: int, b: int) -> int:
    """Independent carryless polynomial convolution followed by long division."""
    product = 0
    for bit in range(8):
        if b & (1 << bit):
            product ^= a << bit
    for degree in range(14, 7, -1):
        if product & (1 << degree):
            product ^= 0x11D << (degree - 8)
    return product


REFERENCE_MUL = tuple(tuple(polynomial_product(a, b) for b in range(256)) for a in range(256))
REFERENCE_INV = (0,) + tuple(
    next(b for b in range(1, 256) if REFERENCE_MUL[a][b] == 1) for a in range(1, 256)
)


def reference_combine(coefficients: bytes, source: tuple[bytes, ...]) -> bytes:
    output = bytearray(len(source[0]))
    for coefficient, symbol in zip(coefficients, source):
        for index, value in enumerate(symbol):
            output[index] ^= REFERENCE_MUL[coefficient][value]
    return bytes(output)


def reference_solve(words: list[Codeword], dimension: int, width: int) -> tuple[int, tuple | None]:
    """Batch Gauss-Jordan reference with row swaps, independent of online decoder."""
    matrix = [list(word.coefficients + word.payload) for word in words]
    rank = 0
    for column in range(dimension):
        pivot = next((row for row in range(rank, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = REFERENCE_INV[matrix[rank][column]]
        matrix[rank] = [REFERENCE_MUL[scale][value] for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [
                    a ^ REFERENCE_MUL[factor][b] for a, b in zip(matrix[row], matrix[rank])
                ]
        rank += 1
    if rank != dimension:
        return rank, None
    return rank, tuple(bytes(matrix[i][dimension : dimension + width]) for i in range(dimension))


def scientific_context(tasks: tuple[str, ...] = ("flow", "pressure")) -> ScientificContext:
    return ScientificContext(
        "sup_task mean |conditional expectation gap|",
        "synthetic qualification fixture",
        "fixed-seed software verification",
        (),
        "fixed declared task evaluator",
        "original task-role assignment",
        1.0,
        tasks,
        "declared packet budget",
        "deterministic fixtures; no empirical uncertainty",
        EvidenceClass.NUMERICAL_VERIFICATION,
        ("equal sample weights",),
    )


class PacketQualification(unittest.TestCase):
    def test_serialized_roundtrip_header_semantics_and_length(self):
        packet = Packet(
            999999,
            b"\0\xff\x01payload",
            2.5,
            {"units": "mmHg", "nested": [1, "µ"]},
            PacketHeader(1, "sensor-A", 999999, 2.5, "BP", "raw", 10),
        )
        recovered = Packet.deserialize(packet.serialize())
        self.assertEqual(recovered, packet)
        self.assertEqual(recovered.serialize(), packet.serialize())
        self.assertEqual(packet.payload_length, 10)
        self.assertEqual(packet.wire_length, len(packet.serialize()))
        self.assertEqual(packet.checksum, zlib.crc32(packet.serialize()[:-4]))

    def test_framing_corruption_truncation_and_malformed_headers(self):
        frame = Packet(10, b"abcdef", 0.25).serialize()
        for index in range(len(frame)):
            corrupted = bytearray(frame)
            corrupted[index] ^= 1
            with self.assertRaises(ValueError):
                Packet.deserialize(bytes(corrupted))
        for size in (0, 1, 12, len(frame) - 1):
            with self.assertRaises(ValueError):
                Packet.deserialize(frame[:size])
        with self.assertRaises(ValueError):
            Packet.deserialize(frame + b"extra")
        for document in (
            b"[]",
            b"null",
            b'{"header":{},"metadata":{}}',
            b'{"header":{},"header":{},"metadata":{}}',
        ):
            body = struct.pack("!4sBII", b"PWPK", 1, len(document), 1) + document + b"x"
            with self.assertRaises(ValueError):
                Packet.deserialize(body + struct.pack("!I", zlib.crc32(body)))

    def test_invalid_identifiers_timestamps_lengths_metadata(self):
        for sequence in (-1, 1.2, True, "10"):
            with self.assertRaises(ValueError):
                Packet(sequence, b"x")
        for timestamp in (float("nan"), float("inf"), -0.1, "now"):
            with self.assertRaises(ValueError):
                Packet(0, b"x", timestamp)
        with self.assertRaises(ValueError):
            Packet(10, b"x", 2, header=PacketHeader(1, "s", 10, 1, payload_length=1))
        with self.assertRaises(ValueError):
            Packet(10, b"x", header=PacketHeader(1, "s", 10, 0, payload_length=2))
        for metadata in ({"x": float("nan")}, {"x": float("inf")}, {1: "bad"}, {"x": object()}):
            with self.assertRaises(ValueError):
                Packet(0, b"x", metadata=metadata)
        for version in (0, 2):
            with self.assertRaises(ValueError):
                PacketHeader(version, "s", 1, 0)

    def test_packetization_roundtrip_arbitrary_source_order_and_trailing_erasure(self):
        packets = packetize(
            b"abcdefgh",
            3,
            sequences=(30, 10, 20),
            timestamps=(0, 0.5, 2),
            source_id="device",
            modality="ECG",
            representation="quantized",
        )
        self.assertEqual(reconstruct(reversed(packets)), b"abcdefgh")
        self.assertEqual(
            reconstruct(reversed(packets), expected_sequences=(30, 10, 20)), b"abcdefgh"
        )
        with self.assertRaises(ValueError):
            reconstruct(packets[:-1])
        with self.assertRaises(ValueError):
            reconstruct(packets, expected_count=4)
        with self.assertRaises(ValueError):
            packetize(b"abc", 1, sequences=(10, 10, 20))
        self.assertEqual(packetize(b"", 4), ())
        self.assertEqual(reconstruct(()), b"")

    def test_explicit_repeat_rejection_and_idempotent_wire_receiver(self):
        packet = Packet(10, b"x")
        buffer = PacketBuffer()
        buffer.add(packet)
        with self.assertRaises(ValueError):
            buffer.add(packet)
        receiver = PacketReceiver()
        self.assertTrue(receiver.add(packet))
        self.assertFalse(receiver.add(Packet.deserialize(packet.serialize())))
        self.assertEqual(receiver.duplicate_count, 1)
        self.assertEqual(receiver.available((10, 20)), (packet, None))
        with self.assertRaises(ValueError):
            receiver.add(Packet(10, b"different"))
        with self.assertRaises(ValueError):
            reconstruct((packet, packet), expected_sequences=(10,))
        with self.assertRaises(ValueError):
            ErasureChannel(PacketLoss(0)).transmit((packet, packet))

    def test_seeded_packet_id_and_serialization_fuzz(self):
        rng = random.Random(712)
        for _ in range(120):
            count = rng.randrange(1, 20)
            ids = rng.sample(range(10, 1_000_000), count)
            packets = tuple(
                Packet(
                    seq,
                    bytes(rng.randrange(256) for _ in range(rng.randrange(25))),
                    rng.random() * 100,
                    {"role": f"task-{i}"},
                )
                for i, seq in enumerate(ids)
            )
            for packet in packets:
                self.assertEqual(Packet.deserialize(packet.serialize()), packet)
            shuffled = list(packets)
            rng.shuffle(shuffled)
            self.assertEqual(
                reconstruct(shuffled, expected_sequences=ids), b"".join(p.payload for p in packets)
            )

    def test_enormous_ids_do_not_allocate_sequence_sized_storage(self):
        packet = Packet(10**100, b"x")
        with self.assertRaises(ValueError):
            reconstruct((packet,))
        with self.assertRaises(ValueError):
            reconstruct((packet,), expected_count=10**100)
        self.assertEqual(reconstruct((packet,), expected_sequences=(10**100,)), b"x")
        self.assertEqual(ErasureChannel(PacketLoss(0)).transmit((packet,)).delays, {10**100: 0})


class ChannelQualification(unittest.TestCase):
    def test_latency_never_indexes_sequence_ids(self):
        for ids in ((0, 1, 2), (10, 20, 30), (30, 10, 20), (7, 99999, 15)):
            with self.subTest(ids=ids):
                packets = tuple(
                    Packet(seq, bytes([i]), time)
                    for i, (seq, time) in enumerate(zip(ids, (2.0, 0.0, 0.5)))
                )
                result = ErasureChannel(PacketLoss(0), LatencyJitter(0.25)).transmit(packets)
                self.assertEqual(result.delays, dict.fromkeys(ids, 0.25))
                self.assertEqual(
                    tuple(p.sequence for p in result.received), (ids[1], ids[2], ids[0])
                )
                for packet in packets:
                    self.assertEqual(result.arrival_times[packet.sequence], packet.timestamp + 0.25)
                self.assertEqual(result.transport.packet_delivery, 1)

    def test_seeded_jitter_reordering_and_id_relabeling_invariance(self):
        def run(ids):
            packets = tuple(Packet(seq, bytes([i])) for i, seq in enumerate(ids))
            return ErasureChannel(
                PacketLoss(0),
                LatencyJitter(1, 0.75),
                reorder_probability=0.5,
                reorder_delay=3,
                seed=15,
            ).transmit(packets)

        first = run(tuple(range(30)))
        second = run(tuple(10 + 1000 * i for i in range(30)))
        self.assertEqual(
            tuple(p.payload for p in first.received), tuple(p.payload for p in second.received)
        )
        self.assertNotEqual(tuple(p.sequence for p in first.received), tuple(range(30)))
        self.assertEqual(tuple(first.delays.values()), tuple(second.delays.values()))
        self.assertGreater(first.transport.latency_jitter, 0)
        for event in first.events:
            self.assertGreaterEqual(event.arrival, event.departure)
            self.assertIsNotNone(event.packet)

    def test_loss_and_erasure_have_no_recovered_zero_payload(self):
        packets = packetize(b"abcdef", 2)
        for channel, status in (
            (ErasureChannel(PacketLoss(1)), "lost"),
            (ErasureChannel(PacketLoss(0), erasure_probability=1), "erased"),
        ):
            result = channel.transmit(packets)
            self.assertEqual(result.received, ())
            self.assertEqual(result.lost_sequences, (0, 1, 2))
            self.assertTrue(all(e.status == status and e.packet is None for e in result.events))
            with self.assertRaises(ValueError):
                reconstruct(result.received, expected_count=3)

    def test_gilbert_state_persists_across_batches_and_reset_replays(self):
        channel = ErasureChannel(GilbertElliottLoss(1, 1))
        first = channel.transmit((Packet(10, b"a"),))
        self.assertEqual(first.transport.packet_delivery, 1)
        self.assertEqual(channel.channel_state, "bad")
        second = channel.transmit((Packet(20, b"b"),))
        self.assertEqual(second.lost_sequences, (20,))
        self.assertEqual(channel.channel_state, "good")
        channel.reset()
        self.assertEqual(channel.transmit((Packet(10, b"a"),)), first)
        absorbing = ErasureChannel(GilbertElliottLoss(1, 0))
        result = absorbing.transmit(tuple(Packet(i, b"x") for i in range(8)))
        self.assertEqual(result.lost_sequences, tuple(range(1, 8)))
        self.assertEqual(tuple(e.channel_state for e in result.events), ("good",) + ("bad",) * 7)

    def test_iid_rng_and_fixed_burst_memory_do_not_restart_per_call(self):
        packets = tuple(Packet(i, b"x") for i in range(40))
        config = PacketLoss(0.35, burst_length=3, seed=42)
        whole = ErasureChannel(config).transmit(packets)
        split = ErasureChannel(config)
        losses = tuple(seq for p in packets for seq in split.transmit((p,)).lost_sequences)
        self.assertEqual(losses, whole.lost_sequences)
        self.assertNotEqual(losses, ())
        self.assertNotEqual(losses, tuple(range(40)))

    def test_deterministic_seed_replay_and_new_seed_changes_trace(self):
        packets = tuple(Packet(i * 100 + 7, b"x") for i in range(50))
        channel = ErasureChannel(
            GilbertElliottLoss(0.2, 0.4, 0.1, 0.9),
            LatencyJitter(0.1, 0.08),
            duplicate_probability=0.3,
            reorder_probability=0.2,
            seed=800,
        )
        first = channel.transmit(packets)
        channel.reset()
        self.assertEqual(channel.transmit(packets), first)
        channel.reset(seed=801)
        self.assertNotEqual(channel.transmit(packets), first)

    def test_queue_service_capacity_and_byte_rate(self):
        packets = tuple(Packet(i, b"x") for i in range(3))
        result = ErasureChannel(PacketLoss(0), service_time=0.2).transmit(packets)
        for i, packet in enumerate(packets):
            self.assertAlmostEqual(result.delays[packet.sequence], (i + 1) * 0.2)
        self.assertAlmostEqual(result.transport.mean_latency, 0.4)
        limited = ErasureChannel(PacketLoss(0), service_time=0.2, queue_capacity=1).transmit(
            packets
        )
        self.assertEqual(limited.lost_sequences, (1, 2))
        self.assertEqual(limited.transport.queue_drops, 2)
        result = ErasureChannel(PacketLoss(0), bytes_per_second=100).transmit(packets[:1])
        self.assertAlmostEqual(result.delays[0], packets[0].wire_length / 100)
        self.assertEqual(result.transport.wire_bytes, packets[0].wire_length)
        self.assertEqual(
            ErasureChannel(PacketLoss(0), queue_capacity=0).transmit(packets).received, ()
        )

    def test_deadline_budgets_absolute_overrides_and_boundary(self):
        packets = tuple(Packet(i, b"x", 2) for i in range(3))
        result = ErasureChannel(PacketLoss(0), LatencyJitter(0.5), deadline=0.5).transmit(
            packets, deadline=3, deadlines={1: 2.4}
        )
        self.assertEqual(tuple(p.sequence for p in result.received), (0, 2))
        self.assertEqual(result.lost_sequences, (1,))
        self.assertEqual(result.transport.deadline_events, 1)
        queued = ErasureChannel(PacketLoss(0), service_time=1, deadline=1.5).transmit(packets)
        self.assertEqual(tuple(p.sequence for p in queued.received), (0,))
        self.assertTrue(all(e.packet is None for e in queued.events if e.status != "delivered"))
        with self.assertRaises(ValueError):
            ErasureChannel(PacketLoss(0)).transmit(packets, deadlines={99: 2})

    def test_retransmission_repairs_burst_loss_and_respects_deadline(self):
        packet = Packet(100, b"not zeros")
        loss = GilbertElliottLoss(0, 1, initial_state="bad")
        success = ErasureChannel(
            loss, LatencyJitter(0.05), max_retries=2, retry_timeout=0.2
        ).transmit((packet,))
        self.assertEqual(success.received, (packet,))
        self.assertAlmostEqual(success.delays[100], 0.25)
        self.assertEqual(success.transport.transmission_attempts, 2)
        expired = ErasureChannel(
            loss, LatencyJitter(0.05), deadline=0.24, max_retries=2, retry_timeout=0.2
        ).transmit((packet,))
        self.assertEqual(expired.received, ())
        self.assertEqual(expired.lost_sequences, (100,))

    def test_corrupt_checksum_is_rejected_then_retransmitted(self):
        def corrupt_first(frame, packet, attempt):
            return frame[:-1] + bytes([frame[-1] ^ 1]) if attempt == 0 else frame

        packet = Packet(70, b"bytes survive retry")
        result = ErasureChannel(
            PacketLoss(0), max_retries=1, retry_timeout=0.1, wire_transform=corrupt_first
        ).transmit((packet,))
        self.assertEqual(result.received, (packet,))
        self.assertEqual(result.transport.corrupt_events, 1)
        self.assertEqual(result.transport.transmission_attempts, 2)
        self.assertIsNone(result.events[0].packet)
        result = ErasureChannel(PacketLoss(0), corruption_probability=1).transmit((packet,))
        self.assertEqual(result.received, ())
        self.assertEqual(result.transport.corrupt_events, 1)

    def test_explicit_duplicates_and_timeout_induced_duplicates(self):
        packet = Packet(8, b"x")
        duplicated = ErasureChannel(
            PacketLoss(0), duplicate_probability=1, duplicate_delay=0.1
        ).transmit((packet,))
        self.assertEqual(duplicated.received, (packet, packet))
        self.assertEqual(duplicated.transport.packet_delivery, 1)
        self.assertEqual(duplicated.transport.duplicate_arrivals, 1)
        self.assertEqual(duplicated.transport.transmission_attempts, 1)
        self.assertEqual(duplicated.transport.wire_bytes, packet.wire_length * 2)
        self.assertEqual(duplicated.delays, {8: 0})
        delayed = ErasureChannel(
            PacketLoss(0), LatencyJitter(2), max_retries=2, retry_timeout=0.5
        ).transmit((packet,))
        self.assertEqual(delayed.received, (packet,) * 3)
        self.assertEqual(delayed.transport.transmission_attempts, 3)
        self.assertEqual(delayed.transport.duplicate_arrivals, 2)
        receiver = PacketReceiver()
        for copy in delayed.received:
            receiver.add(copy)
        self.assertEqual(reconstruct(receiver.ordered(), expected_sequences=(8,)), b"x")

    def test_cancelled_timeouts_do_not_delay_next_batch(self):
        channel = ErasureChannel(PacketLoss(0), LatencyJitter(0.1), max_retries=2, retry_timeout=20)
        channel.transmit((Packet(0, b"x"),))
        self.assertAlmostEqual(channel.time, 0.1)
        second = channel.transmit((Packet(1, b"x", 0.1),))
        self.assertAlmostEqual(second.delays[1], 0.1)

    def test_finite_parameter_validation_and_no_nan_defaults(self):
        for invalid in (float("nan"), float("inf"), -0.1):
            for name in (
                "erasure_probability",
                "duplicate_probability",
                "reorder_probability",
                "corruption_probability",
                "duplicate_delay",
                "reorder_delay",
                "service_time",
                "deadline",
                "retry_timeout",
                "bytes_per_second",
            ):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    ErasureChannel(PacketLoss(0), **{name: invalid})
            with self.assertRaises(ValueError):
                ErasureChannel(PacketLoss(invalid))
            with self.assertRaises(ValueError):
                ErasureChannel(PacketLoss(0), LatencyJitter(invalid))
            with self.assertRaises(ValueError):
                ErasureChannel(PacketLoss(0), LatencyJitter(0, invalid))
            with self.assertRaises(ValueError):
                GilbertElliottLoss(invalid, 0.2)
        for name in (
            "erasure_probability",
            "duplicate_probability",
            "reorder_probability",
            "corruption_probability",
        ):
            with self.assertRaises(ValueError):
                ErasureChannel(PacketLoss(0), **{name: 1.01})
        for name in ("max_retries", "queue_capacity"):
            with self.assertRaises(ValueError):
                ErasureChannel(PacketLoss(0), **{name: 1.5})
        empty = ErasureChannel(PacketLoss(0)).transmit(())
        self.assertEqual(empty.transport.packet_delivery, 1)
        self.assertIsNone(empty.transport.mean_latency)
        self.assertIsNone(empty.transport.latency_jitter)

    def test_exported_configuration_helpers_validate_eagerly_and_accept_legacy_types(self):
        from physioworld.perturbations.network import LatencyJitter as LegacyLatency
        from physioworld.perturbations.network import PacketLoss as LegacyLoss

        for value in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                PacketLoss(value)
            with self.assertRaises(ValueError):
                LatencyJitter(value)
            with self.assertRaises(ValueError):
                LatencyJitter(0, value)
        for helper in (PacketLoss(0).mask, LatencyJitter(0).delays):
            for count in (-1, True, 2.5):
                with self.assertRaises(ValueError):
                    helper(count)
        packet = Packet(10, b"x")
        self.assertEqual(
            ErasureChannel(LegacyLoss(0), LegacyLatency(0)).transmit((packet,)).received, (packet,)
        )
        with self.assertRaises(ValueError):
            ErasureChannel(LegacyLoss(0), LegacyLatency(float("nan")))

    def test_large_finite_delays_produce_finite_statistics(self):
        result = ErasureChannel(PacketLoss(0), LatencyJitter(1e308)).transmit(
            (Packet(10, b"x"), Packet(30, b"y"))
        )
        self.assertTrue(math.isfinite(result.transport.mean_latency))
        self.assertEqual(result.transport.latency_jitter, 0)
        with self.assertRaises(ValueError):
            ErasureChannel(PacketLoss(0), LatencyJitter(1e308), erasure_probability=1).transmit(
                (Packet(10, b"x", 1e308),)
            )

    def test_randomized_channel_configuration_and_chronology_invariants(self):
        rng = random.Random(4409)
        for case in range(100):
            ids = rng.sample(range(10, 100000), rng.randrange(1, 9))
            packets = tuple(
                Packet(seq, bytes([i]), rng.random() * 0.1) for i, seq in enumerate(ids)
            )
            channel = ErasureChannel(
                PacketLoss(rng.choice((0, 0.1, 0.5, 1))),
                LatencyJitter(rng.choice((0, 0.01, 0.1)), rng.choice((0, 0.05))),
                erasure_probability=rng.choice((0, 0.3, 1)),
                duplicate_probability=rng.choice((0, 0.5, 1)),
                corruption_probability=rng.choice((0, 0.2)),
                reorder_probability=rng.choice((0, 0.5)),
                reorder_delay=0.1,
                service_time=rng.choice((0, 0.01, 0.03)),
                queue_capacity=rng.choice((None, 0, 1, 3)),
                deadline=rng.choice((None, 0, 0.05, 0.3)),
                max_retries=rng.choice((0, 2)),
                retry_timeout=0.03,
                seed=case,
            )
            result = channel.transmit(packets)
            by_id = {p.sequence: p for p in packets}
            for event in result.events:
                self.assertGreaterEqual(event.departure, by_id[event.sequence].timestamp)
                if event.arrival is not None:
                    self.assertGreaterEqual(event.arrival, event.departure)
                if event.packet is not None:
                    self.assertEqual(event.packet, by_id[event.sequence])
                    if channel.deadline is not None:
                        self.assertLessEqual(
                            event.arrival - event.packet.timestamp, channel.deadline + 1e-12
                        )
            delivered = {p.sequence for p in result.received}
            self.assertEqual(set(result.lost_sequences), set(ids) - delivered)
            self.assertLessEqual(result.transport.packet_delivery, 1)
            for sequence, delay in result.delays.items():
                arrivals = [
                    e.arrival
                    for e in result.events
                    if e.sequence == sequence and e.status == "delivered"
                ]
                self.assertAlmostEqual(delay, min(arrivals) - by_id[sequence].timestamp)


class CodingQualification(unittest.TestCase):
    def test_all_gf_products_and_inverses_against_polynomial_reference(self):
        for a in range(256):
            for b in range(256):
                self.assertEqual(GF256.mul(a, b), REFERENCE_MUL[a][b])
                self.assertEqual(GF256.add(a, b), a ^ b)
        for a in range(1, 256):
            self.assertEqual(GF256.inv(a), REFERENCE_INV[a])
        self.assertEqual(GF256.mul(0x80, 2), 0x1D)
        with self.assertRaises(ZeroDivisionError):
            GF256.inv(0)
        for a in (-1, 256, True, 1.5):
            with self.assertRaises(ValueError):
                GF256.mul(a, 1)
        with self.assertRaises(ValueError):
            GF256.pow(1, -1)

    def test_dimensions_1_through_19_multiseed_erasure_reorder_reference(self):
        for dimension in range(1, 20):
            for seed in (0, 3, 11, 101, 901):
                with self.subTest(dimension=dimension, seed=seed):
                    rng = random.Random(1000 * dimension + seed)
                    width = rng.randrange(1, 13)
                    source = tuple(
                        bytes(rng.randrange(256) for _ in range(width)) for _ in range(dimension)
                    )
                    words = RLNCEncoder(source, seed).encode(2 * dimension + 10)
                    for word in words:
                        self.assertEqual(word.payload, reference_combine(word.coefficients, source))
                    # Erase every systematic word; decode from actual random repairs.
                    survivors = [
                        replace(word, sequence=10 + word.sequence * 19)
                        for word in words[dimension:]
                        if rng.random() > 0.15
                    ]
                    rng.shuffle(survivors)
                    rank, solution = reference_solve(survivors, dimension, width)
                    decoder = RLNCDecoder(dimension, width)
                    for word in survivors:
                        decoder.add(Codeword.deserialize(word.serialize()))
                    self.assertEqual(decoder.rank, rank)
                    self.assertEqual(decoder.complete, rank == dimension)
                    if rank == dimension:
                        self.assertEqual(decoder.decode(), source)
                        self.assertEqual(decoder.decode(), solution)
                    else:
                        with self.assertRaises(ValueError):
                            decoder.decode()

    def test_late_earlier_pivot_and_reference_incremental_ranks(self):
        source = (b"A", b"B", b"C")
        coefficients = (b"\0\x01\x01", b"\x01\x01\0", b"\x01\0\x01", b"\0\0\x02")
        decoder = RLNCDecoder(3, 1)
        words = []
        for i, coefficients_row in enumerate(coefficients):
            word = Codeword(coefficients_row, reference_combine(coefficients_row, source), i * 11)
            words.append(word)
            decoder.add(word)
            self.assertEqual(decoder.rank, reference_solve(words, 3, 1)[0])
        self.assertEqual(decoder.decode(), source)

    def test_missing_symbols_remain_unavailable_not_zero(self):
        source = (b"\0\0", b"ab", b"cd")
        words = RLNCEncoder(source).encode()
        decoder = RLNCDecoder(3, 2)
        decoder.add(words[1])
        self.assertEqual(decoder.available(), (None, b"ab", None))
        with self.assertRaises(ValueError):
            decoder.decode()
        decoder.add(Codeword(b"\x01\0\x01", reference_combine(b"\x01\0\x01", source), 9))
        self.assertEqual(decoder.available(), (None, b"ab", None))
        decoder.add(words[2])
        self.assertEqual(decoder.decode(), source)

    def test_codeword_duplicates_dimensions_and_inconsistent_equations(self):
        decoder = RLNCDecoder(2, 1)
        word = Codeword(b"\x01\0", b"a", 10)
        self.assertTrue(decoder.add(word))
        self.assertFalse(decoder.add(word))
        with self.assertRaises(ValueError):
            decoder.add(Codeword(b"\x01\0", b"b", 10))
        with self.assertRaises(ValueError):
            decoder.add(Codeword(b"\x01\0", b"b", 20))
        with self.assertRaises(ValueError):
            decoder.add(Codeword(b"\0\0", b"x", 30))
        with self.assertRaises(ValueError):
            decoder.add(Codeword(b"\x01", b"x", 40))
        self.assertEqual(decoder.rank, 1)
        frame = word.serialize()
        with self.assertRaises(ValueError):
            Codeword.deserialize(frame[:-1] + bytes([frame[-1] ^ 1]))

    def test_encoder_repair_batches_and_zero_count_have_real_semantics(self):
        encoder = RLNCEncoder((b"a", b"b", b"c"), seed=4)
        self.assertEqual(encoder.encode(0), ())
        first = encoder.encode(2)
        second = encoder.encode(1)
        repair = encoder.encode(4)
        self.assertEqual(tuple(word.sequence for word in first + second + repair), tuple(range(7)))
        self.assertEqual(second[0].coefficients, b"\0\0\x01")
        self.assertEqual(encoder.encode(0), ())
        self.assertEqual(first + second + repair, RLNCEncoder((b"a", b"b", b"c"), 4).encode(7))

    def test_actual_packet_rlnc_recovers_frames_after_erasure_reorder_and_timing(self):
        for dimension in range(1, 20):
            for seed in (3, 19):
                sources = tuple(
                    Packet(
                        10 + i * 13,
                        bytes([i]) * (i + 1),
                        i / 10,
                        {"units": "mmHg", "role": f"task-{i}"},
                    )
                    for i in range(dimension)
                )
                encoder = RLNCPacketEncoder(sources, seed=seed, generation_id="g")
                wire = encoder.encode(dimension * 2 + 10)
                # Non-contiguous outer transport IDs are distinct from coding IDs.
                wire = tuple(
                    replace(
                        p, sequence=100 + 17 * i, header=replace(p.header, sequence=100 + 17 * i)
                    )
                    for i, p in enumerate(wire[dimension:])
                )
                result = ErasureChannel(
                    PacketLoss(0.05), LatencyJitter(0.1, 0.05), duplicate_probability=0.2, seed=seed
                ).transmit(wire)
                decoder = RLNCPacketDecoder(dimension, encoder.source_width, "g")
                for packet in result.received:
                    decoder.add(packet)
                self.assertEqual(
                    tuple(p.serialize() for p in decoder.decode()),
                    tuple(p.serialize() for p in sources),
                )
                self.assertTrue(
                    all(e.departure >= max(p.timestamp for p in sources) for e in result.events)
                )

    def test_packet_coding_rejects_future_leakage_and_generation_mismatch(self):
        encoder = RLNCPacketEncoder((Packet(10, b"x", 2), Packet(20, b"abc", 3)))
        with self.assertRaises(ValueError):
            encoder.encode(timestamp=2.9)
        word = encoder.encode(1)[0]
        decoder = RLNCPacketDecoder(2, encoder.source_width, "wrong")
        with self.assertRaises(ValueError):
            decoder.add(word)
        decoder = RLNCPacketDecoder(2, encoder.source_width)
        decoder.add(word)
        self.assertEqual(decoder.available()[0], encoder.packets[0])
        self.assertIsNone(decoder.available()[1])

    def test_packet_decoder_rejects_conflicting_wire_ids_and_bool_dimensions(self):
        encoder = RLNCPacketEncoder((Packet(10, b"x"), Packet(20, b"y")))
        words = encoder.encode(2)
        decoder = RLNCPacketDecoder(2, encoder.source_width)
        decoder.add(words[0])
        changed = replace(
            words[1],
            sequence=words[0].sequence,
            header=replace(words[1].header, sequence=words[0].sequence),
        )
        with self.assertRaises(ValueError):
            decoder.add(changed)
        malformed = replace(words[1], metadata={**words[1].metadata, "dimension": True})
        with self.assertRaises(ValueError):
            decoder.add(malformed)
        self.assertEqual(decoder.rank, 1)

    def test_real_xor_parity_recovers_every_single_erasure_not_two(self):
        sources = tuple(
            Packet(10 + 20 * i, bytes(range(i + 1)), i * 0.1, {"role": f"modality-{i}"})
            for i in range(5)
        )
        encoder = ParityEncoder(sources, generation_id="parity")
        words = encoder.encode()
        self.assertEqual(len(words), 6)
        self.assertEqual(Codeword.deserialize(words[-1].payload).coefficients, b"\x01" * 5)
        for erased in range(6):
            decoder = ParityDecoder(5, encoder.source_width, "parity")
            for i, packet in reversed(tuple(enumerate(words))):
                if i != erased:
                    decoder.add(packet)
            self.assertEqual(decoder.decode(), sources)
        decoder = ParityDecoder(5, encoder.source_width, "parity")
        for packet in words[2:]:
            decoder.add(packet)
        self.assertFalse(decoder.complete)
        self.assertEqual(decoder.available()[:2], (None, None))
        with self.assertRaises(ValueError):
            decoder.decode()

    def test_parity_packet_and_full_wire_byte_budgets_are_enforced(self):
        source = (Packet(10, b"long payload"), Packet(20, b"short"))
        encoder = ParityEncoder(source)
        with self.assertRaises(ValueError):
            encoder.encode(packet_budget=2)
        with self.assertRaises(ValueError):
            encoder.encode(byte_budget=1)
        block = encoder.encode(packet_budget=3)
        self.assertEqual(tuple(packet.sequence for packet in block), (0, 1, 2))
        byte_count = sum(packet.wire_length for packet in block)
        fresh = ParityEncoder(source)
        self.assertEqual(fresh.encode(packet_budget=3, byte_budget=byte_count), block)
        with self.assertRaises(ValueError):
            ParityEncoder(source).encode(byte_budget=byte_count - 1)

    def test_semantic_priority_has_real_payloads_and_hard_budgets(self):
        sources = (Packet(10, b"critical"), Packet(20, b"ordinary"), Packet(30, b"minor"))
        encoder = SemanticPriorityEncoder(sources, {10: 10, 20: 2, 30: 0.1}, max_copies=5)
        wire = encoder.encode(7)
        self.assertEqual(len(wire), 7)
        self.assertGreater(encoder.last_allocation[10], encoder.last_allocation[20])
        self.assertGreater(encoder.last_allocation[10], encoder.last_allocation[30])
        self.assertEqual(len({p.sequence for p in wire}), 7)
        decoder = SemanticPriorityDecoder((10, 20, 30))
        # Erase all first copies: only actually allocated repeats survive.
        for packet in reversed(wire):
            if packet.metadata["copy"] > 0:
                decoder.add(packet)
        self.assertEqual(decoder.available()[0], sources[0])
        self.assertIsNone(decoder.available()[2])
        with self.assertRaises(ValueError):
            decoder.decode()
        for packet in wire:
            decoder.add(packet)
        self.assertEqual(decoder.decode(), sources)
        baseline_bytes = wire[0].wire_length
        constrained = SemanticPriorityEncoder(sources, {10: 10, 20: 2, 30: 0.1})
        protected = constrained.encode(20, byte_budget=baseline_bytes)
        self.assertEqual(len(protected), 1)
        self.assertLessEqual(sum(p.wire_length for p in protected), baseline_bytes)
        self.assertEqual(Packet.deserialize(protected[0].payload), sources[0])
        self.assertEqual(constrained.encode(0), ())

    def test_adaptive_policy_emits_budgeted_protection(self):
        sources = (Packet(1, b"high"), Packet(2, b"low"))
        contexts = {1: ProtectionContext(1, 1, 1, 1, 0.5), 2: ProtectionContext(0, 0, 0, 0.1, 0.5)}
        policy = RedundancyPolicy(1, 5)
        self.assertEqual(policy.choose(contexts[1]), 5)
        self.assertEqual(policy.choose(contexts[2]), 1)
        wire = policy.protect(sources, contexts, 6, generation_id="adaptive")
        self.assertEqual(len(wire), 6)
        self.assertEqual(sum(p.metadata["source_sequence"] == 1 for p in wire), 5)
        decoder = SemanticPriorityDecoder((1, 2), "adaptive")
        for packet in wire:
            decoder.add(packet)
        self.assertEqual(decoder.decode(), sources)
        with self.assertRaises(ValueError):
            ProtectionContext(float("nan"), 1, 1, 1, 0)
        self.assertEqual(policy.choose(ProtectionContext(1e308, 1e308, 1, 0, 0)), 1)


class SemanticQualification(unittest.TestCase):
    def test_task_law_supremum_is_separate_from_mean_distortion(self):
        context = scientific_context()
        result = evaluate_semantics(
            (0, 1),
            (0, 1),
            context=context,
            reference_task_laws=((0, 0), (1, 1)),
            observed_task_laws=((0.2, 0.2), (0.4, 0.4)),
            reference_task_value=3,
            task_value=1,
        )
        self.assertEqual(result.reconstruction_distortion, 0)
        self.assertAlmostEqual(result.task_law_discrepancy, 0.6)
        self.assertEqual(result.task_value, 1)
        self.assertIsNone(result.capability_preservation)
        unavailable = evaluate_semantics((0, 1), (0, None), context=context)
        self.assertEqual(unavailable.available_fraction, 0.5)
        self.assertIsNone(unavailable.reconstruction_distortion)
        self.assertIsNone(unavailable.task_law_discrepancy)

    def test_permanent_permutation_keeps_transport_exact_but_breaks_task_roles(self):
        source = (
            Packet(10, b"\0", metadata={"role": "flow"}),
            Packet(20, b"\x01", metadata={"role": "pressure"}),
        )
        control = permute_payloads(source, (1, 0))
        for original, permuted in zip(source, control):
            self.assertEqual(original.metadata, permuted.metadata)
            self.assertEqual(Packet.deserialize(permuted.serialize()), permuted)
        results = []
        for packets in (source, control):
            encoder = RLNCPacketEncoder(packets)
            channel = ErasureChannel(PacketLoss(0))
            result = channel.transmit(encoder.encode(4))
            decoder = RLNCPacketDecoder(2, encoder.source_width)
            for packet in reversed(result.received):
                decoder.add(packet)
            recovered = decoder.decode()
            self.assertEqual(recovered, packets)
            semantic = result.evaluate_semantics(
                (0, 1),
                tuple(float(p.payload[0]) for p in recovered),
                context=scientific_context(),
                reference_task_laws=((0,), (1,)),
                observed_task_laws=tuple((float(p.payload[0]),) for p in recovered),
            )
            results.append((result.transport, semantic))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][1].task_law_discrepancy, 0)
        self.assertEqual(results[1][1].task_law_discrepancy, 1)

    def test_executable_negative_control_reports_context_and_paired_evidence(self):
        module = runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "examples"
                / "communication_negative_control.py"
            )
        )
        report = module["run_negative_control"](seed=17, trials=4)
        self.assertEqual(report["valid"]["transport"], report["permuted"]["transport"])
        self.assertEqual(report["valid"]["transport"]["decode_rate"], 1)
        self.assertEqual(report["valid"]["semantic"]["task_law_discrepancy"], 0)
        self.assertEqual(report["permuted"]["semantic"]["task_law_discrepancy"], 1)
        self.assertEqual(report["scientific_context"]["evidence_class"], EvidenceClass.SIMULATION)
        json.dumps(report, allow_nan=False)

    def test_metric_counts_and_semantic_input_validation(self):
        for counts in ((1, 2, 1, 1, 1, 1), (1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 2)):
            with self.assertRaises(ValueError):
                CommunicationMetrics.from_counts(*counts)
        metrics = CommunicationMetrics.from_counts(0, 0, 0, 0, 0, 0)
        self.assertEqual(metrics.packet_delivery, 1)
        self.assertIsNone(metrics.task_law_metric)
        with self.assertRaises(ValueError):
            evaluate_semantics(
                (0,), (0,), context=scientific_context(), reference_task_laws=((0,),)
            )
        with self.assertRaises(ValueError):
            mean_squared_distortion((0,), (float("nan"),))
        with self.assertRaises(ValueError):
            UniformQuantizer(float("nan"), 1, 256)
        with self.assertRaises(ValueError):
            UniformQuantizer(0, 1, 4).decode(b"\xff")
        with self.assertRaises(ValueError):
            permute_payloads((Packet(0, b"x"), Packet(1, b"y")), (1, 1))


if __name__ == "__main__":
    unittest.main()
