"""Numerical and seeded-simulation VVUQ for the communication kernels.

The field solver in this module is deliberately independent of the production
GF(256) arithmetic. Transport simulations use only the packaged packet-loss
fixture and synthetic role-indexed task vectors. They test transport and
semantic integrity separately; neither is empirical or clinical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from importlib.resources import files
from typing import Sequence

from physioworld.communication import (
    ErasureChannel,
    EvidenceClass,
    LatencyJitter,
    Packet,
    PacketHeader,
    PacketLoss,
    RLNCDecoder,
    RLNCEncoder,
    RLNCPacketDecoder,
    RLNCPacketEncoder,
    ScientificContext,
    evaluate_semantics,
    permute_payloads,
)
from physioworld.communication.models import SemanticStatistics, TransportStatistics


def _polynomial_product(a: int, b: int) -> int:
    product = 0
    for bit in range(8):
        if b & (1 << bit):
            product ^= a << bit
    for degree in range(14, 7, -1):
        if product & (1 << degree):
            product ^= 0x11D << (degree - 8)
    return product


REFERENCE_MUL = tuple(tuple(_polynomial_product(a, b) for b in range(256)) for a in range(256))
REFERENCE_INV = (0,) + tuple(
    next(b for b in range(1, 256) if REFERENCE_MUL[a][b] == 1) for a in range(1, 256)
)


def _reference_combine(coefficients: bytes, source: tuple[bytes, ...]) -> bytes:
    output = bytearray(len(source[0]))
    for coefficient, symbol in zip(coefficients, source):
        for index, value in enumerate(symbol):
            output[index] ^= REFERENCE_MUL[coefficient][value]
    return bytes(output)


def _reference_solve(words, dimension: int, width: int):
    """Batch Gauss-Jordan reference with row swaps and independent GF tables."""
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
                    left ^ REFERENCE_MUL[factor][right]
                    for left, right in zip(matrix[row], matrix[rank])
                ]
        rank += 1
    if rank != dimension:
        return rank, None
    return rank, tuple(bytes(matrix[i][dimension : dimension + width]) for i in range(dimension))


def _comparison(observed: float, reference: float, tolerance: float) -> dict:
    residual = observed - reference
    return {
        "observed": observed,
        "reference": reference,
        "residual": residual,
        "absolute_tolerance": tolerance,
        "passed": math.isfinite(residual) and abs(residual) <= tolerance,
    }


def _symbols_hex(symbols):
    return None if symbols is None else [symbol.hex() for symbol in symbols]


def _source_packets() -> tuple[Packet, ...]:
    return tuple(
        Packet(
            sequence,
            values,
            0.0,
            {"task": role},
            PacketHeader(1, "synthetic", sequence, 0.0, role, "binary-task-law", len(values)),
        )
        for sequence, role, values in (
            (10, "flow-low", b"\x00\x00\x01\x01"),
            (30, "pressure-low", b"\x01\x01\x00\x00"),
        )
    )


def _context() -> ScientificContext:
    return ScientificContext(
        estimand="sup_task mean_sample |E[g|H] - E[g|Z_received]|",
        data_domain="synthetic two-role binary task expectations",
        evaluation_regime="paired seeded finite-field transport simulation",
        conditioning=("complete decoded blocks",),
        policy_class="fixed role-indexed binary decisions",
        reference_state="unpermuted source role assignments",
        time_horizon=0.0,
        task_family=("flow-low", "pressure-low"),
        resource_state="four coded packets per two-source block",
        uncertainty_treatment="seeded finite-trial frequency; no statistical interval",
        evidence_class=EvidenceClass.SIMULATION,
        assumptions=(
            "task expectations are deterministic binary functions in this synthetic world",
            "receiver retains declared role assignments",
            "paired transport seeds are shared between arms",
        ),
    )


def _semantic_trial(packets: tuple[Packet, ...], seed: int):
    encoder = RLNCPacketEncoder(packets, seed=seed, generation_id="vvuq")
    channel = ErasureChannel(
        PacketLoss(0.2, seed=seed),
        LatencyJitter(0.003, 0.001, seed=seed),
        duplicate_probability=0.1,
        reorder_probability=0.2,
        reorder_delay=0.004,
        seed=seed,
    )
    result = channel.transmit(encoder.encode(10))
    decoder = RLNCPacketDecoder(encoder.dimension, encoder.source_width, "vvuq")
    for packet in result.received:
        decoder.add(packet)
    decoded = decoder.decode() if decoder.complete else None
    return result, decoded


def _semantic_arm(packets: tuple[Packet, ...], seed: int, trials: int, context):
    transport: list[TransportStatistics] = []
    semantic: list[SemanticStatistics] = []
    for offset in range(trials):
        result, decoded = _semantic_trial(packets, seed + offset)
        transport.append(result.transport)
        if decoded is None:
            continue
        laws = tuple(tuple(float(value) for value in packet.payload) for packet in decoded)
        reference_laws = tuple(
            tuple(float(value) for value in packet.payload) for packet in _source_packets()
        )
        reference = tuple(value for law in reference_laws for value in law)
        observed = tuple(value for law in laws for value in law)
        semantic.append(
            evaluate_semantics(
                reference,
                observed,
                context=context,
                reference_task_laws=reference_laws,
                observed_task_laws=laws,
                reference_task_value=1.0,
                task_value=sum(a == b for a, b in zip(reference, observed)) / len(reference),
            )
        )
    decode_rate = len(semantic) / trials
    return {
        "transport": {
            "trials": trials,
            "mean_packet_delivery": sum(item.packet_delivery for item in transport) / trials,
            "mean_latency_s": sum(item.mean_latency or 0.0 for item in transport) / trials,
            "decode_rate": decode_rate,
            "transport_reproducibility_key": [
                (item.sent_packets, item.delivered_packets, item.transmission_attempts)
                for item in transport[:8]
            ],
        },
        "semantic": {
            "evaluated_blocks": len(semantic),
            "mean_task_law_discrepancy": _mean_optional(
                item.task_law_discrepancy for item in semantic
            ),
            "mean_reconstruction_distortion": _mean_optional(
                item.reconstruction_distortion for item in semantic
            ),
            "mean_task_value": _mean_optional(item.task_value for item in semantic),
        },
    }


def _mean_optional(values):
    observed = tuple(value for value in values if value is not None)
    return sum(observed) / len(observed) if observed else None


def _stochastic_convergence(seed: int, counts: Sequence[int], loss_probability: float):
    rows = []
    source = (b"\x00", b"\x01", b"\x02")
    for count in counts:
        rng = random.Random(seed)
        delivered = 0
        total = 0
        full_rank = 0
        for _ in range(count):
            words = RLNCEncoder(source, seed=rng.randrange(2**31)).encode(4)
            received = [word for word in words if rng.random() >= loss_probability]
            delivered += len(received)
            total += len(words)
            rank, decoded = _reference_solve(received, 3, 1)
            full_rank += int(rank == 3 and decoded == source)
        frequency = delivered / total
        rows.append(
            {
                "trials": count,
                "delivery_fraction": frequency,
                "expected_delivery_fraction": 1.0 - loss_probability,
                "delivery_absolute_error": abs(frequency - (1.0 - loss_probability)),
                "decode_fraction": full_rank / count,
            }
        )
    return rows


def run_communication_vvuq(*, seed: int = 17, trials: int = 64) -> dict:
    """Run independent GF checks, seeded frequency convergence, and role controls."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 8:
        raise ValueError("trials must be an integer >= 8")
    payload = (
        files("physioworld")
        .joinpath("corpus")
        .joinpath("golden")
        .joinpath("packet_loss.json")
        .read_bytes()
    )
    fixture = json.loads(payload)
    source_symbols = (b"\x00", b"\x01", b"\x02")
    encoder = RLNCEncoder(source_symbols, seed=13)
    words = encoder.encode(4)
    dropped = tuple(word for word in words if word.sequence not in set(fixture["lost_sequences"]))
    reference_rank, reference_decoded = _reference_solve(dropped, 3, 1)
    production_decoder = RLNCDecoder(3, 1)
    for word in dropped:
        production_decoder.add(word)
    reference_checks = {
        "field_table_inverse": all(
            REFERENCE_MUL[value][REFERENCE_INV[value]] == 1 for value in range(1, 256)
        ),
        "word_combine": all(
            _reference_combine(word.coefficients, source_symbols) == word.payload for word in words
        ),
        "fixture_rank": {
            "production": production_decoder.rank,
            "independent": reference_rank,
            "passed": production_decoder.rank == reference_rank == 3,
        },
        "fixture_decode": {
            "production": _symbols_hex(
                production_decoder.decode() if production_decoder.complete else None
            ),
            "independent": _symbols_hex(reference_decoded),
            "passed": production_decoder.complete and reference_decoded == source_symbols,
        },
    }
    frequency = _stochastic_convergence(seed, (max(8, trials // 8), trials, trials * 4), 0.2)
    frequency_checks = {
        "normalization": all(0.0 <= row["delivery_fraction"] <= 1.0 for row in frequency),
        "large_sample_error": frequency[-1]["delivery_absolute_error"] <= 0.05,
        "reproducible": frequency
        == _stochastic_convergence(seed, tuple(row["trials"] for row in frequency), 0.2),
    }
    context = _context()
    valid = _semantic_arm(_source_packets(), seed, trials, context)
    permuted = _semantic_arm(permute_payloads(_source_packets(), (1, 0)), seed, trials, context)
    transport_difference = max(
        abs(valid["transport"][name] - permuted["transport"][name])
        for name in ("mean_packet_delivery", "mean_latency_s", "decode_rate")
    )
    semantic_negative_control = {
        "paired_transport_difference": transport_difference,
        "valid_task_law_discrepancy": valid["semantic"]["mean_task_law_discrepancy"],
        "permuted_task_law_discrepancy": permuted["semantic"]["mean_task_law_discrepancy"],
        "passed": (
            transport_difference == 0.0
            and valid["semantic"]["mean_task_law_discrepancy"] == 0.0
            and permuted["semantic"]["mean_task_law_discrepancy"] > 0.0
        ),
    }
    passed = (
        all(
            value is True or (isinstance(value, dict) and value.get("passed", False) is True)
            for value in reference_checks.values()
        )
        and all(frequency_checks.values())
        and semantic_negative_control["passed"]
    )
    return {
        "experiment_id": "communication_vvuq",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "NUMERICAL_VERIFICATION_AND_SIMULATION",
        "data_origin": "GOLDEN_PACKET_FIXTURE_AND_SYNTHETIC_ROLE_TASKS",
        "data_fidelity": "MATHEMATICAL_AND_SYNTHETIC",
        "clinical_claim": False,
        "fixture": {
            "name": "packet_loss.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": fixture,
        },
        "independent_reference": {
            "method": "carryless_polynomial_GF256_and_batch_Gauss_Jordan",
            "checks": reference_checks,
        },
        "stochastic_convergence": {
            "loss_probability": 0.2,
            "seed": seed,
            "rows": frequency,
            "checks": frequency_checks,
            "interpretation": "seeded finite-frequency convergence, not an uncertainty interval",
        },
        "semantic_negative_control": {
            "scientific_context": asdict(context),
            "valid": valid,
            "permuted": permuted,
            "check": semantic_negative_control,
        },
        "limitations": [
            "Finite-field and transport simulation only; no empirical or clinical evidence.",
            "The prompt's historical semantic discrepancy values require unavailable original corpus inputs and are not treated as reproduced.",
            "The negative control tests declared role semantics, not all possible application semantics.",
        ],
    }
