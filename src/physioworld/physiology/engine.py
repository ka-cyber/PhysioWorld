"""Embeddable hybrid physiological simulation, deterministic branching and restart.

Checkpoint access is privileged scientific/debug state access; policies receive
only immutable ``Observation`` objects constructed from their declared requests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .information import (
    ASSESSMENT_VARIABLES,
    Assessment,
    DataRequest,
    Observation,
    RequestAccessError,
)
from .models import (
    MODEL_VERSION,
    VARIABLE_UNITS,
    CoupledCardiorespiratoryModel,
    EnvironmentConditions,
    EquipmentState,
    PatientParameters,
    PhysiologyState,
)

CHECKPOINT_SCHEMA = "physioworld.physiology.checkpoint/1"
CONTROL_RANGES = {
    "exercise_mets": (1.0, 8.0),
    "hemorrhage_ml_min": (0.0, 500.0),
    "infusion_ml_min": (0.0, 1000.0),
    "maintenance_ml_min": (0.0, 10.0),
    "orthostasis": (0.0, 1.0),
    "lbnp_mmhg": (0.0, 60.0),
    "arrhythmia_rate_bpm": (0.0, 220.0),
    "rhythm_efficiency": (0.1, 1.0),
    "contractility": (0.1, 2.0),
}
DEFAULT_CONTROLS = {name: 0.0 for name in CONTROL_RANGES}
DEFAULT_CONTROLS.update(
    exercise_mets=1.0, maintenance_ml_min=0.8, rhythm_efficiency=1.0, contractility=1.0
)
COMPARTMENT_UNITS = {
    f"compartment.{name}": unit
    for name, unit in {
        "arterial_o2_ml": "mL STPD",
        "venous_o2_ml": "mL STPD",
        "arterial_co2_ml": "mL STPD",
        "venous_co2_ml": "mL STPD",
        "alveolar_o2_ml": "mL STPD",
        "alveolar_co2_ml": "mL STPD",
        "plasma_volume_l": "L",
        "red_cell_volume_l": "L",
        "interstitial_volume_l": "L",
        "sodium_mmol": "mmol",
    }.items()
}


class OperatingEnvelopeError(ValueError):
    pass


class OperatingEnvelopeWarning(RuntimeWarning):
    """A recorded trajectory crossed a declared model envelope."""


@dataclass(frozen=True)
class PhysiologicalAction:
    """Scheduled action with a cause and an explicit simulation evidence basis.

    Kinds: control, exercise, hemorrhage, infusion, orthostasis, lbnp, arrhythmia,
    oxygen, ventilation, environment, equipment, bolus, resource, sensor, policy.
    Unit-bearing parameter names are required; duration scheduling is expressed
    as a second action with the desired post-event value (no implicit reset).
    """

    time_s: float
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cause: str = "declared scenario intervention"
    evidence_basis: str = "MODEL_ASSUMPTION"
    safety_status: str = "SIMULATION_ONLY"
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("action time must be finite nonnegative seconds")
        if not self.cause or not self.evidence_basis or not self.safety_status:
            raise ValueError("action cause, evidence basis and safety status required")
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


class PhysiologyEngine:
    """Versioned reduced-order cardiovascular/respiratory/metabolic/renal world.

    ``advance`` takes a *duration* in seconds; ``advance_to`` an absolute time.
    RK4 step-doubling controls local error (atol + rtol*state magnitude), rejects
    nonpositive physical stores and lands exactly on action/forcing boundaries.
    Forcing uses an independent fixed 1-s clock, observations keyed noise, so
    counterfactual branches share exogenous random variates after divergence.
    """

    def __init__(
        self,
        patient: PatientParameters | None = None,
        *,
        seed: int = 0,
        fidelity: str = "coupled",
        environment: EnvironmentConditions | None = None,
        equipment: EquipmentState | None = None,
        requests: DataRequest | None = None,
        resources: Mapping[str, Any] | None = None,
        resource_units: Mapping[str, str] | None = None,
        max_step_s: float = 0.25,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        process_noise_std: float = 0.0,
        observation_noise: Mapping[str, float] | None = None,
        envelope_policy: str = "record",
        available_actions: Sequence[str] | None = None,
    ):
        if not all(math.isfinite(v) and v > 0 for v in (max_step_s, rtol, atol)):
            raise ValueError("solver tolerances and step must be finite positive")
        if max_step_s > 1:
            raise ValueError("maximum step exceeds supported 1-s envelope")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(process_noise_std) or process_noise_std < 0:
            raise ValueError("process noise must be finite nonnegative")
        if envelope_policy not in {"record", "fail"}:
            raise ValueError("envelope_policy must be record or fail")
        self.patient = patient or PatientParameters()
        self.model = CoupledCardiorespiratoryModel(self.patient, fidelity)
        self.environment = environment or EnvironmentConditions()
        self.equipment = equipment or EquipmentState()
        self.requests = requests or DataRequest()
        self.resources = _json_copy(dict(resources or {}))
        self.resource_units = dict(resource_units or {})
        self.controls = dict(DEFAULT_CONTROLS)
        self.time_s = 0.0
        self._state = self.model.initial_state(self.environment)
        self.seed = seed
        self.rng_identity = f"{self.patient.subject_id}:{seed}"
        self._rng = random.Random(self.rng_identity)
        self.process_noise_std = process_noise_std
        self.observation_noise = dict(observation_noise or {})
        for name, sigma in self.observation_noise.items():
            if name not in VARIABLE_UNITS or not math.isfinite(sigma) or sigma < 0:
                raise ValueError(
                    "observation noise requires known variables and finite nonnegative sigma"
                )
        self._forcing = self._rng.gauss(0, process_noise_std)
        self._noise_tick = 1
        self.max_step_s, self.rtol, self.atol = max_step_s, rtol, atol
        self.envelope_policy = envelope_policy
        self.available_actions = (
            tuple(available_actions)
            if available_actions is not None
            else (
                "control",
                "exercise",
                "hemorrhage",
                "infusion",
                "orthostasis",
                "lbnp",
                "arrhythmia",
                "oxygen",
                "ventilation",
                "environment",
                "equipment",
                "bolus",
                "resource",
                "sensor",
                "policy",
            )
        )
        self.sensor_available: dict[str, bool] = {}
        self.policy_state: dict[str, Any] = {}
        self._pending: list[tuple[int, PhysiologicalAction]] = []
        self._next_action_id = 0
        self.events: list[dict[str, Any]] = []
        self.chronology: list[dict[str, Any]] = []
        self.envelope_violations: list[dict[str, Any]] = []
        self._violated_names: set[str] = set()
        self.solver_statistics = {"accepted_steps": 0, "rejected_steps": 0}
        self.branch_id = "root"
        self.lineage: list[dict[str, Any]] = []
        self._initial_balances = self._balances()
        self._validate_request(self.requests)
        self._check_envelope()

    def _outputs(self) -> dict[str, float]:
        return self.model.outputs(self._state, self.environment, self.equipment, self.controls)

    def _validate_request(self, request: DataRequest) -> None:
        known = set(VARIABLE_UNITS) | set(COMPARTMENT_UNITS)
        known.update(f"resource.{k}" for k in self.resource_units if k in self.resources)
        for name in request.variables:
            if name not in known:
                raise RequestAccessError(f"unknown or unitless requested variable: {name}")
            if name.startswith("resource."):
                key = name.split(".", 1)[1]
                value = self.resources[key]
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not self.resource_units[key]
                ):
                    raise RequestAccessError(
                        f"resource observation requires a finite scalar and unit: {name}"
                    )
        if any(name not in ASSESSMENT_VARIABLES for name in request.assessments):
            raise RequestAccessError("unknown assessment")

    def observe(self, request: DataRequest | None = None) -> Observation:
        request = request or self.requests
        if (
            not set(request.variables) <= set(self.requests.variables)
            or not set(request.assessments) <= set(self.requests.assessments)
            or request.events
            and not self.requests.events
        ):
            raise RequestAccessError("request exceeds engine's declared information set")
        o = self._outputs()
        values: dict[str, float | None] = {}
        units = {}
        for name in request.variables:
            if name in VARIABLE_UNITS:
                value = o[name]
                units[name] = VARIABLE_UNITS[name]
                sigma = self.observation_noise.get(name, 0)
                # Named/time-keyed random draws; request order/access never changes future noise.
                key = f"{self.rng_identity}:observation:{self.time_s.hex()}:{name}"
                value += random.Random(key).gauss(0, sigma)
            elif name in COMPARTMENT_UNITS:
                value = getattr(self._state, name.split(".", 1)[1])
                units[name] = COMPARTMENT_UNITS[name]
            else:
                value = float(self.resources[name.split(".", 1)[1]])
                units[name] = self.resource_units[name.split(".", 1)[1]]
            values[name] = value if self.sensor_available.get(name, True) else None
        return Observation(
            self.time_s,
            self.patient.subject_id,
            values,
            units,
            tuple(self.assess(name) for name in request.assessments),
            tuple(copy.deepcopy(e) for e in self.events if e["time_s"] <= self.time_s)
            if request.events
            else (),
        )

    def assess(self, name: str) -> Assessment:
        if name not in self.requests.assessments:
            raise RequestAccessError(f"undeclared assessment: {name}")
        o = self._outputs()
        names = ASSESSMENT_VARIABLES[name]
        return Assessment(
            name, self.time_s, {n: o[n] for n in names}, {n: VARIABLE_UNITS[n] for n in names}
        )

    def schedule(self, action: PhysiologicalAction) -> int:
        if action.time_s < self.time_s:
            raise ValueError("cannot schedule an action in the past")
        self._validate_action(action)
        number = self._next_action_id
        self._next_action_id += 1
        self._pending.append((number, copy.deepcopy(action)))
        self._pending.sort(key=lambda item: (item[1].time_s, item[0]))
        self.chronology.append(
            {
                "operation": "schedule",
                "time_s": self.time_s,
                "action_id": number,
                "action": asdict(action),
            }
        )
        return number

    def act(self, kind: str, **parameters: Any) -> Observation:
        self.schedule(PhysiologicalAction(self.time_s, kind, parameters))
        return self.advance(0)

    def _control_parameters(self, action: PhysiologicalAction) -> dict[str, Any]:
        aliases = {
            "exercise": {"exercise_mets"},
            "hemorrhage": {"hemorrhage_ml_min"},
            "infusion": {"infusion_ml_min", "maintenance_ml_min"},
            "orthostasis": {"orthostasis"},
            "lbnp": {"lbnp_mmhg"},
            "arrhythmia": {"arrhythmia_rate_bpm", "rhythm_efficiency"},
            "control": set(CONTROL_RANGES),
        }
        if action.kind not in aliases:
            return {}
        if not set(action.parameters) <= aliases[action.kind]:
            raise ValueError(f"unknown parameters for {action.kind}")
        return dict(action.parameters)

    def _validate_action(self, action: PhysiologicalAction) -> None:
        if action.kind not in self.available_actions:
            raise ValueError(f"action unavailable: {action.kind}")
        controls = self._control_parameters(action)
        for name, value in controls.items():
            low, high = CONTROL_RANGES[name]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} outside [{low}, {high}]")
        if action.kind == "environment":
            replace(self.environment, **action.parameters)
        elif action.kind in {"equipment", "oxygen", "ventilation"}:
            allowed = {f.name for f in fields(EquipmentState)}
            if action.kind == "oxygen":
                allowed = {"oxygen_fraction", "enabled"}
            elif action.kind == "ventilation":
                allowed = {"ventilation_l_min", "peep_cmh2o", "enabled"}
            if not set(action.parameters) <= allowed:
                raise ValueError("unknown equipment parameter")
            replace(self.equipment, **action.parameters)
        elif action.kind == "bolus":
            if set(action.parameters) != {"volume_ml"}:
                raise ValueError("bolus requires volume_ml (isotonic saline, 154 mmol/L)")
            v = action.parameters["volume_ml"]
            if not math.isfinite(v) or not 0 <= v <= 2000:
                raise ValueError("bolus outside 0..2000 mL")
        elif action.kind == "sensor":
            if not set(action.parameters) <= set(self.requests.variables) or any(
                not isinstance(v, bool) for v in action.parameters.values()
            ):
                raise ValueError("sensor requires declared variable -> bool availability")
        elif action.kind == "resource":
            if not set(action.parameters) <= set(self.resources):
                raise ValueError("resource action requires existing resource keys")
            _json_copy(action.parameters)
            for name in self.requests.variables:
                if name.startswith("resource.") and name.split(".", 1)[1] in action.parameters:
                    value = action.parameters[name.split(".", 1)[1]]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("observed resource action must remain a finite scalar")
        elif action.kind == "policy":
            _json_copy(action.parameters)

    def _apply_due_actions(self) -> None:
        while self._pending and self._pending[0][1].time_s <= self.time_s:
            number, a = self._pending.pop(0)
            controls = self._control_parameters(a)
            if controls:
                self.controls.update(controls)
            elif a.kind == "environment":
                self.environment = replace(self.environment, **a.parameters)
            elif a.kind in {"equipment", "oxygen", "ventilation"}:
                self.equipment = replace(self.equipment, **a.parameters)
            elif a.kind == "bolus":
                volume = a.parameters["volume_ml"] / 1000
                self._state.plasma_volume_l += volume
                self._state.infused_l += volume
                self._state.sodium_mmol += 154 * volume
                self._state.sodium_input_mmol += 154 * volume
            elif a.kind == "resource":
                self.resources.update(copy.deepcopy(a.parameters))
            elif a.kind == "sensor":
                self.sensor_available.update(a.parameters)
            elif a.kind == "policy":
                self.policy_state.update(copy.deepcopy(a.parameters))
            self.events.append(
                {
                    "action_id": number,
                    "time_s": self.time_s,
                    "kind": a.kind,
                    "parameters": dict(a.parameters),
                    "cause": a.cause,
                    "mechanism": self._mechanism(a.kind),
                    "evidence_basis": a.evidence_basis,
                    "safety_status": a.safety_status,
                    "label": a.label,
                    "evidence_class": "SIMULATION",
                }
            )
        while self._noise_tick <= self.time_s:
            self._forcing = self._rng.gauss(0, self.process_noise_std)
            self._noise_tick += 1

    @staticmethod
    def _mechanism(kind: str) -> str:
        return {
            "exercise": "increased O2 demand, ventilatory drive, autonomic activation and vasodilation",
            "hemorrhage": "loss of plasma/RBC/gases, decreased preload, baroreflex and renal response",
            "infusion": "plasma water/Na input, Hb dilution, altered preload and renal excretion",
            "bolus": "instant plasma water/Na input, Hb dilution and preload change",
            "orthostasis": "venous pooling reduces preload; pressure activates baroreflex",
            "lbnp": "assumed pressure-dependent venous pooling reduces preload",
            "arrhythmia": "rate override and mechanical efficiency change circulating gas transport",
            "oxygen": "inspired O2 changes alveolar store and blood O2 loading",
            "ventilation": "alveolar ventilation changes O2/CO2 flux; PEEP reduces preload",
            "environment": "inspired gas pressure and temperature-dependent metabolic demand",
            "equipment": "gas exchange equipment state and PEEP-mediated preload",
            "sensor": "declared observation availability only",
            "resource": "caller-managed resource state change",
            "policy": "caller-managed policy metadata change",
        }.get(kind, "explicit model control change")

    def _rk4(self, vector: tuple[float, ...], h: float) -> tuple[float, ...]:
        def rhs(v: tuple[float, ...]) -> tuple[float, ...]:
            return self.model.derivative(
                PhysiologyState.from_vector(v),
                self.environment,
                self.equipment,
                self.controls,
                self._forcing,
            )

        a = rhs(vector)
        b = rhs(tuple(x + h * y / 2 for x, y in zip(vector, a)))
        c = rhs(tuple(x + h * y / 2 for x, y in zip(vector, b)))
        d = rhs(tuple(x + h * y for x, y in zip(vector, c)))
        return tuple(
            x + h * (w + 2 * y + 2 * z + v) / 6 for x, w, y, z, v in zip(vector, a, b, c, d)
        )

    def _physical(self, v: tuple[float, ...]) -> bool:
        if not all(math.isfinite(x) and x >= 0 for x in v):
            return False
        s = PhysiologyState.from_vector(v)
        capacity = self.model.lung_capacity(self.environment, self.patient.alveolar_volume_l)
        return (
            s.plasma_volume_l > 0
            and s.red_cell_volume_l > 0
            and s.interstitial_volume_l > 0
            and 0 <= s.sympathetic <= 1
            and 0 <= s.adh <= 1
            and s.alveolar_o2_ml + s.alveolar_co2_ml < capacity
        )

    def advance(self, duration_s: float) -> Observation:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("advance duration must be finite nonnegative seconds")
        return self.advance_to(self.time_s + duration_s)

    def advance_to(self, time_s: float) -> Observation:
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < self.time_s:
            raise ValueError("advance_to requires finite nondecreasing absolute seconds")
        if time_s > 21600:
            raise OperatingEnvelopeError("simulation horizon exceeds 6-hour model envelope")
        start = self.time_s
        self._apply_due_actions()
        while self.time_s < time_s:
            grid = (math.floor(self.time_s / self.max_step_s + 1e-10) + 1) * self.max_step_s
            boundary = min(
                time_s,
                grid,
                float(self._noise_tick),
                self._pending[0][1].time_s if self._pending else time_s,
            )
            remaining = boundary - self.time_s
            if remaining <= 1e-12:
                self.time_s = boundary
                self._apply_due_actions()
                continue
            h = remaining
            vector = self._state.vector()
            while True:
                try:
                    full = self._rk4(vector, h)
                    half = self._rk4(self._rk4(vector, h / 2), h / 2)
                    error = max(
                        abs(a - b) / (15 * (self.atol + self.rtol * max(abs(x), abs(b))))
                        for x, a, b in zip(vector, full, half)
                    )
                    accepted = self._physical(half) and error <= 1
                except (ValueError, ZeroDivisionError, OverflowError):
                    accepted = False
                if accepted:
                    break
                self.solver_statistics["rejected_steps"] += 1
                h /= 2
                if h < 1e-7:
                    raise OperatingEnvelopeError("no positive/error-controlled step available")
            self._state = PhysiologyState.from_vector(half)
            self.time_s = boundary if h == remaining else self.time_s + h
            self.solver_statistics["accepted_steps"] += 1
            self._apply_due_actions()
            self._check_envelope()
        if time_s != start:
            self.chronology.append({"operation": "advance", "from_s": start, "time_s": self.time_s})
        return self.observe()

    def _check_envelope(self) -> None:
        o = self._outputs()
        o["blood_volume_ratio"] = o["blood_volume_l"] / self.patient.blood_volume_l
        for name, (low, high, unit) in self.model.descriptor.operating_envelope.items():
            value = self.time_s if name == "horizon_s" else o[name]
            if not low <= value <= high:
                if self.envelope_policy == "fail":
                    raise OperatingEnvelopeError(f"{name}={value} {unit} outside [{low}, {high}]")
                if name not in self._violated_names:
                    self.envelope_violations.append(
                        {
                            "time_s": self.time_s,
                            "variable": name,
                            "value": value,
                            "unit": unit,
                            "bounds": [low, high],
                        }
                    )
                    self._violated_names.add(name)
                    warnings.warn(
                        f"model envelope crossed: {name}={value} {unit}, declared [{low}, {high}]; "
                        "recorded in provenance",
                        OperatingEnvelopeWarning,
                        stacklevel=2,
                    )

    def _balances(self) -> dict[str, float]:
        s = self._state
        return {
            "fluid_l": s.plasma_volume_l
            + s.red_cell_volume_l
            + s.interstitial_volume_l
            + s.urine_l
            + s.blood_loss_l
            - s.infused_l
            - s.maintenance_l,
            "sodium_mmol": s.sodium_mmol + s.sodium_output_mmol - s.sodium_input_mmol,
            "oxygen_ml": s.arterial_o2_ml
            + s.venous_o2_ml
            + s.alveolar_o2_ml
            + s.oxygen_expired_ml
            + s.oxygen_consumed_ml
            + s.oxygen_lost_ml
            - s.oxygen_inspired_ml,
            "co2_ml": s.arterial_co2_ml
            + s.venous_co2_ml
            + s.alveolar_co2_ml
            + s.co2_expired_ml
            + s.co2_lost_ml
            - s.co2_produced_ml
            - s.co2_inspired_ml,
        }

    def conservation_residuals(self) -> dict[str, float]:
        """Privileged numerical diagnostics, not policy observations."""
        return {k: v - self._initial_balances[k] for k, v in self._balances().items()}

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "physioworld.physiology.manifest/1",
            "model": asdict(self.model.descriptor),
            "patient": asdict(self.patient),
            "seed": self.seed,
            "rng_identity": self.rng_identity,
            "branch_id": self.branch_id,
            "lineage": copy.deepcopy(self.lineage),
            "evidence_class": "SIMULATION",
            "causal_status": "MODEL_COUNTERFACTUAL" if self.lineage else "MODEL_SIMULATION",
            "time_unit": "s",
            "time_s": self.time_s,
            "requests": asdict(self.requests),
            "available_actions": list(self.available_actions),
            "solver": {
                "method": "RK4_step_doubling",
                "max_step_s": self.max_step_s,
                "rtol": self.rtol,
                "atol": self.atol,
                **self.solver_statistics,
            },
            "events": copy.deepcopy(self.events),
            "envelope_violations": copy.deepcopy(self.envelope_violations),
            "conservation_residuals": self.conservation_residuals(),
            "uncertainty": {
                "process_noise_std": self.process_noise_std,
                "observation_noise": dict(self.observation_noise),
                "interpretation": "declared synthetic noise, not confidence intervals",
            },
        }

    def checkpoint(self) -> dict[str, Any]:
        """Full privileged state, including future event queue and RNG Gaussian cache."""
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model_version": MODEL_VERSION,
            "patient": asdict(self.patient),
            "fidelity": self.model.fidelity,
            "environment": asdict(self.environment),
            "equipment": asdict(self.equipment),
            "requests": asdict(self.requests),
            "resources": self.resources,
            "resource_units": self.resource_units,
            "controls": self.controls,
            "time_s": self.time_s,
            "state": asdict(self._state),
            "seed": self.seed,
            "rng_identity": self.rng_identity,
            "rng_state": self._rng.getstate(),
            "forcing": self._forcing,
            "noise_tick": self._noise_tick,
            "process_noise_std": self.process_noise_std,
            "observation_noise": self.observation_noise,
            "max_step_s": self.max_step_s,
            "rtol": self.rtol,
            "atol": self.atol,
            "envelope_policy": self.envelope_policy,
            "available_actions": self.available_actions,
            "sensor_available": self.sensor_available,
            "policy_state": self.policy_state,
            "pending": [{"id": n, "action": asdict(a)} for n, a in self._pending],
            "next_action_id": self._next_action_id,
            "events": self.events,
            "chronology": self.chronology,
            "envelope_violations": self.envelope_violations,
            "violated_names": sorted(self._violated_names),
            "solver_statistics": self.solver_statistics,
            "branch_id": self.branch_id,
            "lineage": self.lineage,
            "initial_balances": self._initial_balances,
        }
        return _json_copy(payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.checkpoint(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> PhysiologyEngine:
        c = _json_copy(checkpoint)
        if c.get("schema") != CHECKPOINT_SCHEMA or c.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported checkpoint/model version")
        obj = cls(
            PatientParameters(**c["patient"]),
            seed=c["seed"],
            fidelity=c["fidelity"],
            environment=EnvironmentConditions(**c["environment"]),
            equipment=EquipmentState(**c["equipment"]),
            requests=DataRequest(**c["requests"]),
            resources=c["resources"],
            resource_units=c["resource_units"],
            max_step_s=c["max_step_s"],
            rtol=c["rtol"],
            atol=c["atol"],
            process_noise_std=c["process_noise_std"],
            observation_noise=c["observation_noise"],
            envelope_policy=c["envelope_policy"],
            available_actions=c["available_actions"],
        )
        obj.time_s, obj._state = float(c["time_s"]), PhysiologyState(**c["state"])
        if (
            not math.isfinite(obj.time_s)
            or not 0 <= obj.time_s <= 21600
            or not obj._physical(obj._state.vector())
        ):
            raise ValueError("checkpoint has nonphysical state/time")
        obj.rng_identity = c["rng_identity"]
        obj._rng.setstate(_tuple_tree(c["rng_state"]))
        obj._forcing, obj._noise_tick = c["forcing"], c["noise_tick"]
        if (
            not math.isfinite(obj._forcing)
            or not isinstance(obj._noise_tick, int)
            or obj._noise_tick != math.floor(obj.time_s) + 1
        ):
            raise ValueError("checkpoint forcing clock inconsistent with simulation time")
        if obj.rng_identity != f"{obj.patient.subject_id}:{obj.seed}":
            raise ValueError("checkpoint RNG identity inconsistent with patient/seed")
        obj.controls = c["controls"]
        if set(obj.controls) != set(CONTROL_RANGES) or any(
            not math.isfinite(value)
            or not CONTROL_RANGES[name][0] <= value <= CONTROL_RANGES[name][1]
            for name, value in obj.controls.items()
        ):
            raise ValueError("checkpoint controls outside envelope")
        obj.sensor_available, obj.policy_state = c["sensor_available"], c["policy_state"]
        obj._pending = [(a["id"], PhysiologicalAction(**a["action"])) for a in c["pending"]]
        if obj._pending != sorted(obj._pending, key=lambda item: (item[1].time_s, item[0])) or len(
            {n for n, _ in obj._pending}
        ) != len(obj._pending):
            raise ValueError("checkpoint action queue not ordered/unique")
        if any(a.time_s < obj.time_s for _, a in obj._pending):
            raise ValueError("checkpoint contains past pending action")
        for _, action in obj._pending:
            obj._validate_action(action)
        obj._next_action_id, obj.events, obj.chronology = (
            c["next_action_id"],
            c["events"],
            c["chronology"],
        )
        if any(e["time_s"] > obj.time_s for e in obj.events) or any(
            b["time_s"] < a["time_s"] for a, b in zip(obj.events, obj.events[1:])
        ):
            raise ValueError("checkpoint event chronology inconsistent")
        obj.envelope_violations, obj._violated_names = (
            c["envelope_violations"],
            set(c["violated_names"]),
        )
        obj.solver_statistics = c["solver_statistics"]
        obj.branch_id, obj.lineage, obj._initial_balances = (
            c["branch_id"],
            c["lineage"],
            c["initial_balances"],
        )
        return obj

    @classmethod
    def load(cls, path: str | Path) -> PhysiologyEngine:
        return cls.from_checkpoint(json.loads(Path(path).read_text(encoding="utf-8")))

    def fork(self, name: str, actions: Sequence[PhysiologicalAction] = ()) -> PhysiologyEngine:
        if not name:
            raise ValueError("fork name required")
        checkpoint = self.checkpoint()
        child = self.from_checkpoint(checkpoint)
        digest = hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest()
        child.branch_id = f"{self.branch_id}/{name}"
        child.lineage.append(
            {
                "parent_branch": self.branch_id,
                "branch": name,
                "time_s": self.time_s,
                "parent_checkpoint_sha256": digest,
                "random_pairing": "COMMON_EXOGENOUS_VARIATES",
                "evidence_class": "SIMULATION",
                "causal_status": "MODEL_COUNTERFACTUAL",
            }
        )
        for action in actions:
            child.schedule(action)
        return child

    branch = fork

    def diff(self, other: PhysiologyEngine) -> dict[str, Any]:
        left, right = self.checkpoint(), other.checkpoint()

        def changes(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
            return {
                k: {"left": a.get(k), "right": b.get(k)}
                for k in sorted(set(a) | set(b))
                if a.get(k) != b.get(k)
            }

        return {
            "model": changes(
                {"fidelity": left["fidelity"], "version": left["model_version"]},
                {"fidelity": right["fidelity"], "version": right["model_version"]},
            ),
            "state": changes(left["state"], right["state"]),
            "parameters": changes(left["patient"], right["patient"]),
            "resources": changes(left["resources"], right["resources"]),
            "policy": changes(left["policy_state"], right["policy_state"]),
            "controls": changes(left["controls"], right["controls"]),
            "environment": changes(left["environment"], right["environment"]),
            "equipment": changes(left["equipment"], right["equipment"]),
            "interventions": changes(
                {"events": left["events"], "pending": left["pending"]},
                {"events": right["events"], "pending": right["pending"]},
            ),
            "chronology": changes(
                {"time_s": left["time_s"], "chronology": left["chronology"]},
                {"time_s": right["time_s"], "chronology": right["chronology"]},
            ),
            "randomness": changes(
                {"identity": left["rng_identity"], "state": left["rng_state"]},
                {"identity": right["rng_identity"], "state": right["rng_state"]},
            ),
            "observability": changes(
                {"requests": left["requests"], "sensors": left["sensor_available"]},
                {"requests": right["requests"], "sensors": right["sensor_available"]},
            ),
            "uncertainty": changes(
                {"process": left["process_noise_std"], "observation": left["observation_noise"]},
                {"process": right["process_noise_std"], "observation": right["observation_noise"]},
            ),
        }

    def coupling_step(
        self,
        time_s: float,
        actions: Sequence[PhysiologicalAction] = (),
    ) -> dict[str, Any]:
        """Native/remote-friendly coupling boundary: explicit inputs, declared JSON outputs.

        Inputs are scheduled actions in simulation seconds; no wall-clock time
        reinterpretation or implicit interpolation. Resource evolution remains
        caller managed through resource actions, separate from physiology.
        """
        if not math.isfinite(time_s) or time_s < self.time_s:
            raise ValueError("coupling clock must be finite and nondecreasing")
        for action in actions:
            if action.time_s < self.time_s:
                raise ValueError("coupling action precedes current simulation time")
            self._validate_action(action)
        for action in actions:
            self.schedule(action)
        return self.advance_to(time_s).to_dict()

    def replay(self, times_s: Sequence[float]) -> Iterator[Observation]:
        for time in times_s:
            yield self.advance_to(time)


CoupledPhysiologicalWorld = PhysiologyEngine
