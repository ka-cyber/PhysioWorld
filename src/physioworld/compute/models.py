"""Serial FIFO compute/placement simulation with explicit network and thermal cost.

This predicts execution from supplied operation/power profiles; it does not
claim measured hardware performance or inference accuracy. Quote/choose are
side-effect free; submit reserves the queue and performs seeded failure draws.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Mapping

from physioworld.core import EnergyStatus, EnergyValue


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    location: str
    cpu_capacity: float
    memory_bytes: int
    energy_per_second_j: float
    queue: int = 0
    network_latency_s: float = 0.0
    communication_overhead_bytes: int = 0
    queue_work_operations: float = 0.0
    network_bandwidth_bytes_s: float = 1e12
    communication_energy_per_byte_j: float = 0.0
    idle_power_w: float = 0.0
    requester_wait_power_w: float = 0.0
    accelerator_capacity: float | None = None
    accelerator_power_w: float = 0.0
    failure_probability: float = 0.0
    ambient_temperature_c: float = 25.0
    initial_temperature_c: float = 25.0
    thermal_resistance_k_w: float = 1.0
    thermal_capacitance_j_k: float = 100.0
    max_temperature_c: float = 100.0

    def __post_init__(self):
        values = (
            self.cpu_capacity,
            self.energy_per_second_j,
            self.network_latency_s,
            self.queue_work_operations,
            self.network_bandwidth_bytes_s,
            self.communication_energy_per_byte_j,
            self.idle_power_w,
            self.requester_wait_power_w,
            self.accelerator_power_w,
            self.failure_probability,
            self.ambient_temperature_c,
            self.initial_temperature_c,
            self.thermal_resistance_k_w,
            self.thermal_capacitance_j_k,
            self.max_temperature_c,
        )
        if (
            not self.node_id
            or self.location not in ("local", "edge", "cloud")
            or not all(math.isfinite(v) for v in values)
            or min(
                self.cpu_capacity,
                self.network_bandwidth_bytes_s,
                self.thermal_resistance_k_w,
                self.thermal_capacitance_j_k,
            )
            <= 0
            or min(
                self.memory_bytes,
                self.queue,
                self.energy_per_second_j,
                self.network_latency_s,
                self.communication_overhead_bytes,
                self.queue_work_operations,
                self.communication_energy_per_byte_j,
                self.idle_power_w,
                self.accelerator_power_w,
            )
            < 0
            or self.requester_wait_power_w < 0
            or min(self.ambient_temperature_c, self.initial_temperature_c, self.max_temperature_c)
            < -273.15
            or not 0 <= self.failure_probability <= 1
            or self.initial_temperature_c > self.max_temperature_c
            or self.queue > 0
            and self.queue_work_operations == 0
            or self.accelerator_capacity is not None
            and (not math.isfinite(self.accelerator_capacity) or self.accelerator_capacity <= 0)
        ):
            raise ValueError(
                "invalid compute capacity/cost; queue counts require explicit backlog operations"
            )
        if any(
            not isinstance(v, int) or isinstance(v, bool)
            for v in (self.memory_bytes, self.queue, self.communication_overhead_bytes)
        ):
            raise ValueError("memory, job counts and byte counts must be integers")


@dataclass(frozen=True)
class ApproximateVariant:
    operation_fraction: float
    task_loss_bound: float
    bound_source: str

    def __post_init__(self):
        if (
            not math.isfinite(self.operation_fraction)
            or not 0 < self.operation_fraction <= 1
            or not math.isfinite(self.task_loss_bound)
            or self.task_loss_bound < 0
            or not self.bound_source
        ):
            raise ValueError(
                "approximation requires a declared operation fraction and task-loss bound source"
            )


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    memory_bytes: int
    operations: float
    accuracy_proxy: float = 0.0
    input_bytes: int = 0
    accelerator_fraction: float = 0.0
    variants: Mapping[str, ApproximateVariant] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not self.model_id
            or min(self.memory_bytes, self.input_bytes) < 0
            or not math.isfinite(self.operations)
            or self.operations < 0
            or not math.isfinite(self.accuracy_proxy)
            or not math.isfinite(self.accelerator_fraction)
            or not 0 <= self.accelerator_fraction <= 1
            or "full" in self.variants
        ):
            raise ValueError("invalid model profile")
        if any(
            not isinstance(v, int) or isinstance(v, bool)
            for v in (self.memory_bytes, self.input_bytes)
        ):
            raise ValueError("model memory/input byte counts must be integers")


@dataclass(frozen=True)
class ExecutionResult:
    node_id: str
    model_id: str
    execution_time_s: float
    energy: EnergyValue
    feasible: bool
    reason: str | None = None
    communication_overhead_bytes: int = 0
    network_latency_s: float = 0.0
    queue_wait_s: float = 0.0
    service_time_s: float = 0.0
    start_time_s: float = 0.0
    completion_time_s: float = 0.0
    final_temperature_c: float = 25.0
    compute_energy_j: float = 0.0
    communication_energy_j: float = 0.0
    waiting_energy_j: float = 0.0
    failure_probability: float = 0.0
    succeeded: bool | None = None
    variant: str = "full"
    task_loss_bound: float = 0.0
    operations: float = 0.0

    @property
    def energy_j(self):
        return self.energy.value_joules


@dataclass
class ComputeEngine:
    nodes: Mapping[str, ComputeNode]
    models: Mapping[str, ModelProfile]
    seed: int = 0

    def __post_init__(self):
        if any(k != n.node_id for k, n in self.nodes.items()) or any(
            k != m.model_id for k, m in self.models.items()
        ):
            raise ValueError("node/model keys must match identifiers")
        self._rng = random.Random(self.seed)
        self._available = {
            k: n.queue_work_operations / n.cpu_capacity for k, n in self.nodes.items()
        }
        self._thermal_time = dict(self._available)
        self._temperature = {
            k: self._thermal(n, n.initial_temperature_c, n.energy_per_second_j, self._available[k])
            for k, n in self.nodes.items()
        }
        self._last_arrival = {k: 0.0 for k in self.nodes}

    @staticmethod
    def _thermal(node, temperature, power, duration):
        equilibrium = node.ambient_temperature_c + power * node.thermal_resistance_k_w
        return equilibrium + (temperature - equilibrium) * math.exp(
            -duration / (node.thermal_resistance_k_w * node.thermal_capacitance_j_k)
        )

    def execute(
        self,
        node_id,
        model_id,
        *,
        input_operations=None,
        arrival_time_s=0.0,
        deadline_s=None,
        variant="full",
        commit=False,
    ):
        if node_id not in self.nodes or model_id not in self.models:
            raise KeyError(node_id if node_id not in self.nodes else model_id)
        node, model = self.nodes[node_id], self.models[model_id]
        operations = model.operations if input_operations is None else input_operations
        if (
            not math.isfinite(operations)
            or operations < 0
            or not math.isfinite(arrival_time_s)
            or arrival_time_s < 0
            or deadline_s is not None
            and (not math.isfinite(deadline_s) or deadline_s < arrival_time_s)
        ):
            raise ValueError("invalid operation count, arrival or absolute deadline")
        if commit and arrival_time_s < self._last_arrival[node_id]:
            raise ValueError("FIFO submissions must be in nondecreasing arrival order")
        loss = 0.0
        if variant != "full":
            if variant not in model.variants:
                raise ValueError("unknown approximation/early-exit profile")
            profile = model.variants[variant]
            operations *= profile.operation_fraction
            loss = profile.task_loss_bound
        reason = None
        if model.memory_bytes > node.memory_bytes:
            reason = "model exceeds memory budget"
        fraction = model.accelerator_fraction
        if fraction > 0 and node.accelerator_capacity is None:
            reason = reason or "required accelerator unavailable"
        cpu_time = operations * (1 - fraction) / node.cpu_capacity
        accelerator_time = operations * fraction / (node.accelerator_capacity or node.cpu_capacity)
        service = cpu_time + accelerator_time  # Sequential CPU/accelerator phases.
        remote = node.location != "local"
        transferred = model.input_bytes + node.communication_overhead_bytes if remote else 0
        network = (
            node.network_latency_s + transferred / node.network_bandwidth_bytes_s if remote else 0.0
        )
        ready = arrival_time_s + network
        start = max(ready, self._available[node_id])
        wait, completion = start - ready, start + service
        start_temp = self._thermal(
            node,
            self._temperature[node_id],
            node.idle_power_w,
            max(0.0, start - self._thermal_time[node_id]),
        )
        cpu_temp = self._thermal(node, start_temp, node.energy_per_second_j, cpu_time)
        temperature = self._thermal(node, cpu_temp, node.accelerator_power_w, accelerator_time)
        if max(start_temp, cpu_temp, temperature) > node.max_temperature_c:
            reason = reason or "thermal limit"
        if deadline_s is not None and completion > deadline_s:
            reason = reason or "deadline exceeded"
        compute_energy = (
            cpu_time * node.energy_per_second_j + accelerator_time * node.accelerator_power_w
        )
        communication_energy = transferred * node.communication_energy_per_byte_j
        waiting_energy = wait * node.requester_wait_power_w
        succeeded = None
        if commit and reason is None:
            self._available[node_id], self._thermal_time[node_id] = completion, completion
            self._temperature[node_id], self._last_arrival[node_id] = temperature, arrival_time_s
            succeeded = self._rng.random() >= node.failure_probability
        return ExecutionResult(
            node_id,
            model_id,
            completion - arrival_time_s,
            EnergyValue(
                compute_energy + communication_energy + waiting_energy,
                EnergyStatus.SIMULATED,
                f"compute:{node_id}",
            ),
            reason is None,
            reason,
            transferred,
            network,
            wait,
            service,
            start,
            completion,
            temperature,
            compute_energy,
            communication_energy,
            waiting_energy,
            node.failure_probability,
            succeeded,
            variant,
            loss,
            operations,
        )

    def submit(self, node_id, model_id, **kwargs):
        return self.execute(node_id, model_id, commit=True, **kwargs)

    def choose(self, candidates, *, latency_limit=None, objective=None, **kwargs):
        if "commit" in kwargs:
            raise ValueError("placement quotes cannot commit jobs")
        if latency_limit is not None and (not math.isfinite(latency_limit) or latency_limit < 0):
            raise ValueError("invalid latency limit")
        results = tuple(self.execute(n, m, **kwargs) for n, m in candidates)
        feasible = tuple(
            r
            for r in results
            if r.feasible and (latency_limit is None or r.execution_time_s <= latency_limit)
        )
        if not feasible:
            raise RuntimeError("no feasible compute placement")
        objective = objective or (lambda result: -result.execution_time_s)
        scores = tuple((objective(result), result) for result in feasible)
        if not all(math.isfinite(v) for v, _ in scores):
            raise ValueError("placement objective must be finite")
        return max(scores, key=lambda pair: pair[0])[1]
