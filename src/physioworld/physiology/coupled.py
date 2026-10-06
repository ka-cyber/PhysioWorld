"""Experimental, resource-limited adult cardiorespiratory/fluid reduction.

The physiological RHS is the conservative ``models.CoupledCardiorespiratoryModel``:
two well-mixed blood gas pools, one fixed-volume alveolar gas pool, plasma/RBC/
interstitial fluid, pressure, autonomic/ventilatory/ADH controllers, sodium,
lactate and bookkeeping integrals. This module adds *dynamic* compressed-oxygen,
crystalloid and device-energy stores; these are not caller-managed metadata.
Blood volume changes preload and Hb, circulation transports gases, gas chemistry
changes ventilatory/autonomic drive, and pressure/perfusion changes urine output.

Additional equations (time in s, gases in mL STPD, ventilation in L BTPS/min)::

    VA_STPD = 1000 * VA_BTPS * (Patm - 47) / 863
    r = min(1, oxygen_draw_ml_min / VA_STPD)
    FiO2 = ambient_FiO2 + (1 - ambient_FiO2) * r
    FiCO2 = ambient_FiCO2 * (1 - r)
    dO2_supply/dt = -oxygen_draw_ml_min / 60
    dfluid_supply/dt = -delivered_infusion_ml_min / 60000
    denergy/dt = -(12 W * assist_active + 3 W * pump_active)

Pure oxygen is ideally mixed into the *effective alveolar* ventilation stream;
draw above that stream's flow is spilled. Cylinder pressure, masks, dead space,
breath mechanics and lung recruitment are absent. Oxygen flow is passive; the
pump and ventilatory assistance require energy. Empty stores immediately stop
their delivery; energy exhaustion also removes PEEP. Spontaneous ventilatory
drive evolves during assistance and becomes the fallback. This does not model
apnea or an energy-dependent biological metabolism. Maintenance water is an
explicit external input of the inherited resting model (0.8 mL/min).

Resource exhaustion is an endogenous hybrid event with an analytical boundary.
Scheduled changes affect requested settings, never overwrite physiological
outputs. Resource modes are held fixed over each RK interval, including endpoint
stages, then switched at the boundary. No state clipping masks a failed physical
step. Operating bounds are assumed research limits, not empirical evidence.

Example: ``result = resource_limited_scenario().run()``. ``result.manifest``
contains all states/units, equations, bounds, assumptions, input chronology,
solver settings, conservation residuals and scientific provenance.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping

from .information import Observation
from .models import (
    MODEL_VERSION,
    VARIABLE_UNITS,
    CoupledCardiorespiratoryModel,
    EnvironmentConditions,
    EquipmentState,
    PatientParameters,
    PhysiologyState,
)

COUPLED_MODEL_VERSION = "0.1.0"
MATURITY = "EXPERIMENTAL_REDUCED_ORDER"
OPERATING_ENVELOPE_LABEL = "ASSUMED_SYNTHETIC_ADULT_SHORT_HORIZON"
COUPLED_MATURITY = MATURITY
COUPLED_OPERATING_ENVELOPE = OPERATING_ENVELOPE_LABEL
VENTILATOR_POWER_W = 12.0
PUMP_POWER_W = 3.0

COUPLED_EQUATIONS = (
    *CoupledCardiorespiratoryModel(PatientParameters()).descriptor.equations,
    "VA_STPD=1000*VA_BTPS*(Patm-47)/863 mL STPD/min",
    "r=min(1,O2_draw/VA_STPD); FiO2=Fambient+(1-Fambient)*r",
    "FiCO2=ambient_CO2*(1-r); spill=max(0,O2_draw-VA_STPD)",
    "dO2_supply=-O2_draw/60; dO2_drawn=O2_draw/60",
    "dfluid_supply=-infusion/60000; dfluid_delivered=infusion/60000",
    "denergy=-power_W; dbattery_used=power_W",
    "power_W=12*assist_active+3*pump_active; delivery stops at empty stores",
)
COUPLED_ASSUMPTIONS = (
    "Adult nonpregnant lumped compartments; no ventricular valves or beat resolution.",
    "O2 Hill curve fixed pH/P50; linear CO2 content; no acid-base/Haldane model.",
    "Gas stores are effective blood/alveolar stores; hemorrhage carries gases.",
    "Original resource reduction; ideal effective-alveolar oxygen mixing; excess O2 spills.",
    "Fixed assumed 12-W assistance and 3-W infusion pump; passive O2 regulator.",
    "Spontaneous drive remains during assistance; no apnea or breath mechanics.",
    "No replenishment; maintenance fluid 0.8 mL/min is an external unlimited input.",
    "Inventories are equipment resources, not biochemical substrate/ATP stores.",
    "Envelope and coefficients are assumptions; no empirical evidence supplied.",
    "No calibration or validation evidence is supplied; outputs are simulation quantities.",
)
COUPLED_PROVENANCE = (
    {
        "relation": "gas/fluid transport",
        "basis": "mass balance and Fick transport",
        "implementation": "existing conservative cardiorespiratory kernel",
    },
    {
        "relation": "O2 binding",
        "basis": "Hill saturation approximation",
        "implementation": "fixed n=2.7, P50=26.8 mmHg; no pH/temperature shift",
    },
    {
        "relation": "feedback/fluid/CO2 closures",
        "basis": "inherited assumed phenomenology, not fitted measurements",
    },
    {
        "relation": "finite supplies and ideal gas mixing",
        "basis": "original algebraic inventory/mixing reduction; assumed device powers",
    },
)
EVENT_DEFINITIONS = {
    "input_change": "exact-time partial replacement of requested intervention inputs",
    "resource_depleted": "analytical inventory boundary; associated delivery immediately stops",
    "operating_envelope_exit": "first recorded crossing when envelope_policy=record",
}

INPUT_BOUNDS = {
    "exercise_mets": (1.0, 6.0, "1"),
    "hemorrhage_ml_min": (0.0, 300.0, "mL/min"),
    "infusion_request_ml_min": (0.0, 250.0, "mL/min"),
    "oxygen_flow_l_min_stpd": (0.0, 10.0, "L STPD/min"),
    "assisted_ventilation_l_min_btps": (0.0, 20.0, "L BTPS/min"),
    "peep_cmh2o": (0.0, 10.0, "cmH2O"),
}
RESOURCE_BOUNDS = {
    "oxygen_supply_ml": (0.0, 200000.0, "mL STPD"),
    "fluid_supply_l": (0.0, 2.0, "L"),
    "battery_energy_j": (0.0, 100000.0, "J"),
}
ENVIRONMENT_BOUNDS = {
    "oxygen_fraction": (0.16, 0.30, "1"),
    "co2_fraction": (0.0, 0.005, "1"),
    "atmospheric_pressure_mmhg": (600.0, 800.0, "mmHg"),
    "temperature_c": (15.0, 30.0, "degC"),
}
OPERATING_BOUNDS = {
    "blood_volume_ratio": (0.7, 1.25, "1"),
    "mean_pressure_mmhg": (40.0, 160.0, "mmHg"),
    "arterial_po2_mmhg": (30.0, 750.0, "mmHg"),
    "arterial_pco2_mmhg": (20.0, 80.0, "mmHg"),
    "sodium_mmol_l": (125.0, 155.0, "mmol/L"),
    "horizon_s": (0.0, 1800.0, "s"),
}
STATE_UNITS = {
    **{
        name: "mL STPD"
        for name in (
            "arterial_o2_ml",
            "venous_o2_ml",
            "alveolar_o2_ml",
            "arterial_co2_ml",
            "venous_co2_ml",
            "alveolar_co2_ml",
        )
    },
    **{
        name: "L"
        for name in (
            "plasma_volume_l",
            "red_cell_volume_l",
            "interstitial_volume_l",
            "urine_l",
            "blood_loss_l",
            "infused_l",
            "maintenance_l",
        )
    },
    **{
        name: "mL STPD"
        for name in (
            "oxygen_debt_ml",
            "oxygen_inspired_ml",
            "oxygen_expired_ml",
            "oxygen_consumed_ml",
            "oxygen_lost_ml",
            "co2_inspired_ml",
            "co2_expired_ml",
            "co2_produced_ml",
            "co2_lost_ml",
        )
    },
    "mean_pressure_mmhg": "mmHg",
    "sympathetic": "1",
    "adh": "1",
    "ventilation_l_min": "L BTPS/min",
    "sodium_mmol": "mmol",
    "lactate_mmol": "mmol",
    "sodium_input_mmol": "mmol",
    "sodium_output_mmol": "mmol",
    **{name: bounds[2] for name, bounds in RESOURCE_BOUNDS.items()},
    "oxygen_drawn_ml": "mL STPD",
    "fluid_delivered_l": "L",
    "battery_used_j": "J",
}
OUTPUT_UNITS = {
    **VARIABLE_UNITS,
    **{name: bounds[2] for name, bounds in RESOURCE_BOUNDS.items()},
    "inspired_oxygen_fraction": "1",
    "oxygen_draw_ml_min": "mL STPD/min",
    "oxygen_spill_ml_min": "mL STPD/min",
    "infusion_delivered_ml_min": "mL/min",
    "device_power_w": "W",
    "ventilatory_assist_active": "1",
}
_RESOURCE_NAMES = tuple(RESOURCE_BOUNDS)


def _check_bounds(name: str, value: float, bounds: tuple[float, float, str]) -> None:
    low, high, unit = bounds
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite in [{low}, {high}] {unit}")


class CoupledEnvelopeError(ValueError):
    """A hard physical/numerical limit or a configured operating limit was crossed."""


@dataclass(frozen=True)
class CoupledInputs:
    """Requested settings; actual delivery depends on finite resource stores."""

    exercise_mets: float = 1.0
    hemorrhage_ml_min: float = 0.0
    infusion_request_ml_min: float = 0.0
    oxygen_flow_l_min_stpd: float = 0.0
    assisted_ventilation_l_min_btps: float = 0.0
    peep_cmh2o: float = 0.0

    def __post_init__(self) -> None:
        for name, bounds in INPUT_BOUNDS.items():
            _check_bounds(name, getattr(self, name), bounds)


@dataclass(frozen=True)
class CoupledResources:
    """Initial equipment inventories, not physiological substrate stores."""

    oxygen_supply_ml: float = 10000.0
    fluid_supply_l: float = 0.5
    battery_energy_j: float = 3600.0

    def __post_init__(self) -> None:
        for name, bounds in RESOURCE_BOUNDS.items():
            _check_bounds(name, getattr(self, name), bounds)


@dataclass
class CoupledState:
    """Explicit physiological state plus remaining inventories and usage integrals.

    ``physiology`` fields use ``STATE_UNITS``; every vector derivative is per s.
    ``vector`` order is inherited physiological fields, the three inventories,
    then their three usage integrals. Public simulator snapshots are copies.
    """

    physiology: PhysiologyState
    oxygen_supply_ml: float
    fluid_supply_l: float
    battery_energy_j: float
    oxygen_drawn_ml: float = 0.0
    fluid_delivered_l: float = 0.0
    battery_used_j: float = 0.0

    def vector(self) -> tuple[float, ...]:
        return self.physiology.vector() + tuple(
            getattr(self, f.name) for f in fields(self) if f.name != "physiology"
        )


@dataclass(frozen=True)
class CoupledIntervention:
    """A partial requested-input change at an exact, absolute time in seconds.

    Equal-time interventions retain insertion order. Resource exhaustion is
    processed before scheduled input changes at the same time. No implicit
    duration/reset or replenishment is inferred.
    """

    time_s: float
    changes: Mapping[str, float]
    label: str = "declared synthetic intervention"

    def __post_init__(self) -> None:
        _check_bounds("intervention time_s", self.time_s, OPERATING_BOUNDS["horizon_s"])
        changes = dict(self.changes)
        if not changes or not set(changes) <= set(INPUT_BOUNDS):
            raise ValueError("intervention requires known, nonempty requested-input changes")
        for name, value in changes.items():
            _check_bounds(name, value, INPUT_BOUNDS[name])
        if not self.label:
            raise ValueError("intervention label is required")
        object.__setattr__(self, "changes", MappingProxyType(changes))

    def to_dict(self) -> dict[str, Any]:
        return {"time_s": self.time_s, "changes": dict(self.changes), "label": self.label}


@dataclass(frozen=True)
class _Delivery:
    oxygen_ml_min: float
    infusion_ml_min: float
    assist_active: bool
    power_w: float

    def depletion_rates(self) -> tuple[float, float, float]:
        return self.oxygen_ml_min / 60, self.infusion_ml_min / 60000, self.power_w


class CoupledPhysiology:
    """Deterministic hybrid simulation with a reusable, vector-valued RHS.

    ``advance_to`` uses absolute seconds and ``observe`` exposes unit-bearing
    simulation outputs. ``derivative(state, inputs)`` is a pure RHS evaluation.
    Default envelope policy is ``fail``; ``record`` retains the first exit for
    each operating variable in provenance. Hard positivity/finiteness and the
    1800-s horizon always apply. Error-controlled RK4 uses step doubling.
    """

    maturity = MATURITY
    operating_envelope_label = OPERATING_ENVELOPE_LABEL

    def __init__(
        self,
        patient: PatientParameters | None = None,
        *,
        environment: EnvironmentConditions | None = None,
        resources: CoupledResources | None = None,
        inputs: CoupledInputs | None = None,
        max_step_s: float = 0.25,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        envelope_policy: str = "fail",
    ) -> None:
        _check_bounds("max_step_s", max_step_s, (1e-6, 1.0, "s"))
        for name, value in (("rtol", rtol), ("atol", atol)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite positive")
        if envelope_policy not in {"fail", "record"}:
            raise ValueError("envelope_policy must be fail or record")
        self.patient = patient or PatientParameters()
        self.environment = environment or EnvironmentConditions()
        for name, bounds in ENVIRONMENT_BOUNDS.items():
            _check_bounds(f"environment.{name}", getattr(self.environment, name), bounds)
        self.initial_resources = resources or CoupledResources()
        self.inputs = inputs or CoupledInputs()
        self._kernel = CoupledCardiorespiratoryModel(self.patient)
        self._state = CoupledState(
            self._kernel.initial_state(self.environment), **asdict(self.initial_resources)
        )
        self.max_step_s, self.rtol, self.atol = max_step_s, rtol, atol
        self.envelope_policy = envelope_policy
        self.time_s = 0.0
        self._pending: list[CoupledIntervention] = []
        self._scheduled: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []
        self._violated_names: set[str] = set()
        self.solver_statistics = {"accepted_steps": 0, "rejected_steps": 0}
        self._initial_inputs = asdict(self.inputs)
        self._initial_balances = self._balances()
        if not self._physical(self._state.physiology.vector()):
            raise CoupledEnvelopeError("initial state violates hard physical bounds")
        self._check_envelope()

    @property
    def state(self) -> CoupledState:
        return copy.deepcopy(self._state)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

    def _delivery(self, s: CoupledState, c: CoupledInputs) -> _Delivery:
        powered = s.battery_energy_j > 0
        assist = powered and c.assisted_ventilation_l_min_btps > 0
        infusion = c.infusion_request_ml_min if powered and s.fluid_supply_l > 0 else 0.0
        oxygen = c.oxygen_flow_l_min_stpd * 1000 if s.oxygen_supply_ml > 0 else 0.0
        power = VENTILATOR_POWER_W * assist + PUMP_POWER_W * (infusion > 0)
        return _Delivery(oxygen, infusion, assist, power)

    def _boundary_inputs(
        self, s: PhysiologyState, c: CoupledInputs, delivery: _Delivery
    ) -> tuple[EquipmentState, dict[str, float], float]:
        va = c.assisted_ventilation_l_min_btps if delivery.assist_active else s.ventilation_l_min
        va_stpd = va * 1000 * (self.environment.atmospheric_pressure_mmhg - 47) / 863
        mixing_fraction = min(1.0, delivery.oxygen_ml_min / va_stpd)
        fio2 = (
            self.environment.oxygen_fraction
            + (1 - self.environment.oxygen_fraction) * mixing_fraction
        )
        equipment = EquipmentState(
            oxygen_fraction=fio2,
            ventilation_l_min=va if delivery.assist_active else None,
            peep_cmh2o=c.peep_cmh2o if delivery.assist_active else 0.0,
        )
        controls = {
            "exercise_mets": c.exercise_mets,
            "hemorrhage_ml_min": c.hemorrhage_ml_min,
            "infusion_ml_min": delivery.infusion_ml_min,
            "maintenance_ml_min": 0.8,
        }
        return equipment, controls, max(0.0, delivery.oxygen_ml_min - va_stpd)

    def _physiology_rhs(
        self, s: PhysiologyState, c: CoupledInputs, delivery: _Delivery
    ) -> tuple[float, ...]:
        equipment, controls, _ = self._boundary_inputs(s, c, delivery)
        return self._kernel.derivative(s, self.environment, equipment, controls)

    def derivative(self, state: CoupledState, inputs: CoupledInputs) -> tuple[float, ...]:
        """Right-sided derivative in the current hybrid mode; no mutations.

        At an empty resource the corresponding flux is zero. External integrators
        must locate depletion events rather than evaluate through negative stores.
        """
        delivery = self._delivery(state, inputs)
        rates = delivery.depletion_rates()
        return (
            self._physiology_rhs(state.physiology, inputs, delivery)
            + tuple(-rate for rate in rates)
            + rates
        )

    def observe(self) -> Observation:
        s = self._state
        delivery = self._delivery(s, self.inputs)
        equipment, controls, spill = self._boundary_inputs(s.physiology, self.inputs, delivery)
        assert equipment.oxygen_fraction is not None
        values = self._kernel.outputs(s.physiology, self.environment, equipment, controls)
        values.update({name: getattr(s, name) for name in _RESOURCE_NAMES})
        values.update(
            inspired_oxygen_fraction=equipment.oxygen_fraction,
            oxygen_draw_ml_min=delivery.oxygen_ml_min,
            oxygen_spill_ml_min=spill,
            infusion_delivered_ml_min=delivery.infusion_ml_min,
            device_power_w=delivery.power_w,
            ventilatory_assist_active=float(delivery.assist_active),
        )
        return Observation(self.time_s, self.patient.subject_id, values, OUTPUT_UNITS)

    def schedule(self, intervention: CoupledIntervention) -> None:
        if intervention.time_s < self.time_s:
            raise ValueError("cannot schedule an intervention in the past")
        self._pending.append(intervention)
        self._pending.sort(key=lambda event: event.time_s)
        self._scheduled.append(intervention.to_dict())

    def _apply_due(self) -> None:
        while self._pending and self._pending[0].time_s <= self.time_s:
            event = self._pending.pop(0)
            self.inputs = replace(self.inputs, **event.changes)
            self._events.append(
                {
                    "kind": "input_change",
                    **event.to_dict(),
                    "evidence_basis": "DECLARED_SYNTHETIC_INTERVENTION",
                }
            )

    def advance(self, duration_s: float) -> Observation:
        """Advance by a finite nonnegative duration in simulation seconds."""
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite nonnegative seconds")
        return self.advance_to(self.time_s + duration_s)

    def _rk4(self, vector: tuple[float, ...], h: float, delivery: _Delivery) -> tuple[float, ...]:
        def rhs(v: tuple[float, ...]) -> tuple[float, ...]:
            return self._physiology_rhs(PhysiologyState.from_vector(v), self.inputs, delivery)

        a = rhs(vector)
        b = rhs(tuple(x + h * y / 2 for x, y in zip(vector, a)))
        c = rhs(tuple(x + h * y / 2 for x, y in zip(vector, b)))
        d = rhs(tuple(x + h * y for x, y in zip(vector, c)))
        return tuple(
            x + h * (w + 2 * y + 2 * z + v) / 6 for x, w, y, z, v in zip(vector, a, b, c, d)
        )

    def _physical(self, vector: tuple[float, ...]) -> bool:
        if not all(math.isfinite(x) and x >= 0 for x in vector):
            return False
        s = PhysiologyState.from_vector(vector)
        capacity = self._kernel.lung_capacity(self.environment, self.patient.alveolar_volume_l)
        return (
            min(s.plasma_volume_l, s.red_cell_volume_l, s.interstitial_volume_l) > 0
            and s.sympathetic <= 1
            and s.adh <= 1
            and s.alveolar_o2_ml + s.alveolar_co2_ml < capacity
        )

    def advance_to(self, time_s: float) -> Observation:
        _check_bounds("time_s", time_s, OPERATING_BOUNDS["horizon_s"])
        if time_s < self.time_s:
            raise ValueError("advance_to requires nondecreasing absolute seconds")
        self._apply_due()
        while self.time_s < time_s:
            delivery = self._delivery(self._state, self.inputs)
            rates = delivery.depletion_rates()
            deadlines = {
                name: self.time_s + getattr(self._state, name) / rate
                for name, rate in zip(_RESOURCE_NAMES, rates)
                if rate > 0
            }
            grid = (math.floor(self.time_s / self.max_step_s + 1e-10) + 1) * self.max_step_s
            boundary = min(
                time_s,
                grid,
                self._pending[0].time_s if self._pending else time_s,
                *deadlines.values(),
            )
            remaining = boundary - self.time_s
            h = remaining
            vector = self._state.physiology.vector()
            if h > 0:
                while True:
                    try:
                        full = self._rk4(vector, h, delivery)
                        half = self._rk4(self._rk4(vector, h / 2, delivery), h / 2, delivery)
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
                    if h < 1e-8:
                        raise CoupledEnvelopeError("no positive, error-controlled step available")
                self._state.physiology = PhysiologyState.from_vector(half)
                self.solver_statistics["accepted_steps"] += 1
            self.time_s = boundary if h == remaining else self.time_s + h
            for name, used_name, rate in zip(
                _RESOURCE_NAMES, ("oxygen_drawn_ml", "fluid_delivered_l", "battery_used_j"), rates
            ):
                exhausted = name in deadlines and self.time_s == deadlines[name]
                supply = getattr(self._state, name)
                used = supply if exhausted else rate * h
                setattr(self._state, name, 0.0 if exhausted else supply - used)
                setattr(self._state, used_name, getattr(self._state, used_name) + used)
                if exhausted:
                    self._events.append(
                        {
                            "kind": "resource_depleted",
                            "time_s": self.time_s,
                            "resource": name,
                            "mechanism": {
                                "oxygen_supply_ml": "oxygen enrichment stops; ambient gas remains",
                                "fluid_supply_l": "infusion and pump power draw stop",
                                "battery_energy_j": "pump/assistance/PEEP stop; spontaneous drive remains",
                            }[name],
                            "evidence_basis": "ASSUMED_RESOURCE_MASS_ENERGY_BALANCE",
                        }
                    )
            self._apply_due()
            self._check_envelope()
        return self.observe()

    def _check_envelope(self) -> None:
        values = {
            name: float(value) for name, value in self.observe().values.items() if value is not None
        }
        values["blood_volume_ratio"] = values["blood_volume_l"] / self.patient.blood_volume_l
        values["horizon_s"] = self.time_s
        for name, (low, high, unit) in OPERATING_BOUNDS.items():
            value = values[name]
            if not low <= value <= high:
                if self.envelope_policy == "fail":
                    raise CoupledEnvelopeError(f"{name}={value} {unit} outside [{low}, {high}]")
                if name not in self._violated_names:
                    violation = {
                        "variable": name,
                        "value": value,
                        "unit": unit,
                        "bounds": [low, high],
                        "time_s": self.time_s,
                    }
                    self._violations.append(violation)
                    self._events.append({"kind": "operating_envelope_exit", **violation})
                    self._violated_names.add(name)

    def _balances(self) -> dict[str, float]:
        s = self._state.physiology
        r = self._state
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
            "oxygen_supply_ml": r.oxygen_supply_ml + r.oxygen_drawn_ml,
            "fluid_supply_l": r.fluid_supply_l + r.fluid_delivered_l,
            "battery_energy_j": r.battery_energy_j + r.battery_used_j,
            "infusion_transfer_l": s.infused_l - r.fluid_delivered_l,
        }

    def conservation_residuals(self) -> dict[str, float]:
        return {
            name: value - self._initial_balances[name] for name, value in self._balances().items()
        }

    def manifest(self) -> dict[str, Any]:
        """JSON-compatible scientific specification and complete run provenance."""
        return {
            "schema": "physioworld.physiology.coupled/1",
            "model": {
                "name": "resource_limited_cardiorespiratory_fluid",
                "version": COUPLED_MODEL_VERSION,
                "scope": "cardiovascular-respiratory-gas-exchange-fluid/resource reduction",
                "maturity": self.maturity,
                "operating_envelope_label": self.operating_envelope_label,
                "evidence_class": "SIMULATION",
                "validation_status": "NOT_VALIDATED",
                "calibration_status": "NOT_CALIBRATED",
                "parameter_origin": self.patient.parameter_origin,
                "calibration_data": [],
                "empirical_evaluation_evidence": [],
                "kernel": {"module": "physioworld.physiology.models", "version": MODEL_VERSION},
                "equations": list(COUPLED_EQUATIONS),
                "state_units": dict(STATE_UNITS),
                "output_units": dict(OUTPUT_UNITS),
                "state_vector_order": [f.name for f in fields(PhysiologyState)]
                + [f.name for f in fields(CoupledState) if f.name != "physiology"],
                "input_bounds": dict(INPUT_BOUNDS),
                "initial_resource_bounds": dict(RESOURCE_BOUNDS),
                "environment_bounds": dict(ENVIRONMENT_BOUNDS),
                "parameter_bounds": self.patient.ranges(),
                "operating_bounds": dict(OPERATING_BOUNDS),
                "hard_bounds": [
                    "finite nonnegative stores and usage integrals; strictly positive fluid pools",
                    "0<=sympathetic,ADH<=1; alveolar_O2+CO2<dry_gas_capacity",
                    "absolute time in 0..1800 s; positive accepted RK stages/endpoint",
                ],
                "assumptions": list(COUPLED_ASSUMPTIONS),
                "provenance": list(COUPLED_PROVENANCE),
                "event_definitions": dict(EVENT_DEFINITIONS),
            },
            "patient": asdict(self.patient),
            "environment": asdict(self.environment),
            "initial_resources": asdict(self.initial_resources),
            "initial_inputs": dict(self._initial_inputs),
            "current_inputs": asdict(self.inputs),
            "time_s": self.time_s,
            "time_unit": "s",
            "evidence_class": "SIMULATION",
            "state": asdict(self._state),
            "events": copy.deepcopy(self._events),
            "scheduled_interventions": copy.deepcopy(self._scheduled),
            "pending_interventions": [event.to_dict() for event in self._pending],
            "envelope_policy": self.envelope_policy,
            "envelope_violations": copy.deepcopy(self._violations),
            "solver": {
                "method": "RK4_STEP_DOUBLING_WITH_ANALYTICAL_RESOURCE_BOUNDARIES",
                "max_step_s": self.max_step_s,
                "rtol": self.rtol,
                "atol": self.atol,
                **self.solver_statistics,
            },
            "conservation_residuals": self.conservation_residuals(),
        }


@dataclass(frozen=True)
class CoupledScenarioResult:
    observations: tuple[Observation, ...]
    manifest: Mapping[str, Any]
    final_state: CoupledState


@dataclass(frozen=True)
class CoupledScenario:
    """One reproducible bounded scenario specification, with endpoint sampling."""

    name: str
    duration_s: float = 240.0
    sample_interval_s: float = 10.0
    resources: CoupledResources = field(default_factory=CoupledResources)
    interventions: tuple[CoupledIntervention, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name is required")
        _check_bounds("duration_s", self.duration_s, OPERATING_BOUNDS["horizon_s"])
        if not math.isfinite(self.sample_interval_s) or self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be finite positive")
        object.__setattr__(self, "interventions", tuple(self.interventions))
        if any(event.time_s > self.duration_s for event in self.interventions):
            raise ValueError("intervention exceeds scenario duration")

    def run(self, *, max_step_s: float = 0.25) -> CoupledScenarioResult:
        world = CoupledPhysiology(resources=self.resources, max_step_s=max_step_s)
        for event in self.interventions:
            world.schedule(event)
        observations = [world.advance_to(0)]
        index = 1
        while world.time_s < self.duration_s:
            observations.append(
                world.advance_to(min(index * self.sample_interval_s, self.duration_s))
            )
            index += 1
        manifest = world.manifest()
        manifest["scenario"] = {
            "name": self.name,
            "duration_s": self.duration_s,
            "sample_interval_s": self.sample_interval_s,
            "resources": asdict(self.resources),
            "interventions": [event.to_dict() for event in self.interventions],
        }
        return CoupledScenarioResult(tuple(observations), manifest, world.state)


def resource_limited_scenario() -> CoupledScenario:
    """Synthetic exercise/bleeding with finite assistance and crystalloid delivery.

    Oxygen, fluid and battery exhaust at 100, 130 and 150 s respectively under
    the stated constant draws. These are numerical/directional checks of the
    declared reduction, not observations of a human intervention response.
    """
    return CoupledScenario(
        name="exercise_bleeding_and_finite_support",
        resources=CoupledResources(
            oxygen_supply_ml=4000, fluid_supply_l=0.1, battery_energy_j=1500
        ),
        interventions=(
            CoupledIntervention(
                20,
                {"exercise_mets": 3, "hemorrhage_ml_min": 150},
                "exercise and controlled blood loss",
            ),
            CoupledIntervention(
                40,
                {
                    "oxygen_flow_l_min_stpd": 4,
                    "assisted_ventilation_l_min_btps": 8,
                    "peep_cmh2o": 5,
                },
                "finite oxygen and powered ventilatory assistance",
            ),
            CoupledIntervention(70, {"infusion_request_ml_min": 100}, "finite crystalloid pump"),
            CoupledIntervention(120, {"hemorrhage_ml_min": 0}, "blood loss ends"),
            CoupledIntervention(
                170,
                {
                    "exercise_mets": 1,
                    "oxygen_flow_l_min_stpd": 0,
                    "assisted_ventilation_l_min_btps": 0,
                    "peep_cmh2o": 0,
                    "infusion_request_ml_min": 0,
                },
                "rest with spontaneous ventilation",
            ),
        ),
    )
