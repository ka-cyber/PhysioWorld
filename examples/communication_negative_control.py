"""Seeded synthetic control: equal decoding, incorrect physiological role semantics.

Run from the project with:
    PYTHONPATH=src python examples/communication_negative_control.py

The two binary payloads are synthetic conditional task expectations, not clinical
measurements. The control swaps their role assignments before coding and retains
valid checksums. It illustrates the section-10 principle without reproducing or
substituting for the historical audited cohort's numerical estimates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from physioworld.communication import (
    ErasureChannel,
    EvidenceClass,
    LatencyJitter,
    Packet,
    PacketHeader,
    PacketLoss,
    RLNCPacketDecoder,
    RLNCPacketEncoder,
    ScientificContext,
    evaluate_semantics,
    permute_payloads,
)


def run_negative_control(*, seed: int = 17, trials: int = 16) -> dict:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    source = tuple(
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
    reference_laws = tuple(tuple(float(value) for value in p.payload) for p in source)
    reference = tuple(value for law in reference_laws for value in law)
    context = ScientificContext(
        estimand="sup_task mean_sample |E[g_task|H] - E[g_task|Z_received]|",
        data_domain="synthetic two-role binary task expectations; four equiprobable histories",
        evaluation_regime="paired seeded finite-field transport simulation",
        conditioning=("complete decoded blocks for semantic metrics",),
        policy_class="fixed role-indexed binary decisions",
        reference_state="unpermuted source role assignments",
        time_horizon=0.0,
        task_family=("flow-low", "pressure-low"),
        resource_state="10 coded packets per two-source block; no retransmission",
        uncertainty_treatment="finite deterministic seeded trial summary; no confidence interval",
        evidence_class=EvidenceClass.SIMULATION,
        assumptions=(
            "task expectations are deterministic binary functions in this synthetic world",
            "role-indexed receiver does not know the source permutation",
            "transport and coding seeds are paired between arms",
        ),
    )
    arms = {}
    for name, packets in (("valid", source), ("permuted", permute_payloads(source, (1, 0)))):
        successful = 0
        semantic_results = []
        transport_results = []
        for trial in range(trials):
            trial_seed = seed + trial
            encoder = RLNCPacketEncoder(packets, seed=trial_seed, generation_id="control")
            channel = ErasureChannel(
                PacketLoss(0.2, seed=trial_seed),
                LatencyJitter(0.003, 0.001, seed=trial_seed),
                duplicate_probability=0.1,
                reorder_probability=0.2,
                reorder_delay=0.004,
                seed=trial_seed,
            )
            result = channel.transmit(encoder.encode(10))
            transport_results.append(result.transport)
            decoder = RLNCPacketDecoder(encoder.dimension, encoder.source_width, "control")
            for packet in result.received:
                decoder.add(packet)
            if not decoder.complete:
                continue
            successful += 1
            decoded = decoder.decode()
            # Transport reconstruction is exact relative to each arm's actual
            # encoded bytes, including the permuted arm's incorrect assignments.
            if tuple(p.serialize() for p in decoded) != tuple(p.serialize() for p in packets):
                raise AssertionError("transport reconstruction failed")
            laws = tuple(tuple(float(value) for value in p.payload) for p in decoded)
            observed = tuple(value for law in laws for value in law)
            value = sum(a == b for a, b in zip(reference, observed)) / len(reference)
            semantic_results.append(
                evaluate_semantics(
                    reference,
                    observed,
                    context=context,
                    reference_task_laws=reference_laws,
                    observed_task_laws=laws,
                    reference_task_value=1.0,
                    task_value=value,
                )
            )
        arms[name] = {
            "transport": {
                "packet_delivery": sum(r.packet_delivery for r in transport_results) / trials,
                "decode_rate": successful / trials,
                "exact_decoded_blocks": successful,
                "trials": trials,
            },
            "semantic": {
                "evaluated_blocks": len(semantic_results),
                "task_law_discrepancy": (
                    sum(r.task_law_discrepancy for r in semantic_results) / successful
                    if successful
                    else None
                ),
                "reconstruction_distortion_against_correct_roles": (
                    sum(r.reconstruction_distortion for r in semantic_results) / successful
                    if successful
                    else None
                ),
                "task_value": (
                    sum(r.task_value for r in semantic_results) / successful if successful else None
                ),
            },
        }
    if arms["valid"]["transport"] != arms["permuted"]["transport"]:
        raise AssertionError("paired control did not preserve transport performance")
    return {"seed": seed, "scientific_context": asdict(context), **arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trials", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_negative_control(seed=args.seed, trials=args.trials), indent=2))


if __name__ == "__main__":
    main()
