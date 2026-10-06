"""Conservative, lumped adult cardiopulmonary model (synthetic, not calibrated).

Blood is two well-mixed pools, lungs one alveolar store; circulation transports
O2/CO2 between them. All gas amounts/fluxes use mL STPD. Lung and ventilation
volumes use L BTPS with dry-gas conversion Pdry/863. This is a reduced-order
teaching/research model, not an electrophysiological or clinical patient twin.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Protocol

MODEL_VERSION = "1.0.0"


def bounded(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class PatientParameters:
    subject_id: str = "synthetic-adult"
    mass_kg: float = 70.0
    blood_volume_l: float = 5.0
    hematocrit: float = 0.45
    hemoglobin_g_dl: float = 15.0
    interstitial_volume_l: float = 12.0
    heart_rate_bpm: float = 72.0
    cardiac_output_l_min: float = 5.0
    mean_pressure_mmhg: float = 90.0
    oxygen_consumption_ml_min: float = 245.0
    respiratory_quotient: float = 0.8
    alveolar_volume_l: float = 3.0
    shunt_fraction: float = 0.02
    renal_filtration_ml_min: float = 120.0
    autonomic_tau_s: float = 8.0
    ventilation_tau_s: float = 6.0
    adh_tau_s: float = 90.0
    parameter_origin: str = "ASSUMED_SYNTHETIC"

    def __post_init__(self) -> None:
        ranges = self.ranges()
        for name, (low, high, _) in ranges.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"patient {name} outside [{low}, {high}]")
        if not self.subject_id:
            raise ValueError("subject_id is required")
        if self.parameter_origin not in {"ASSUMED_SYNTHETIC", "FITTED", "MEASURED"}:
            raise ValueError("declare parameter origin")

    @staticmethod
    def ranges() -> dict[str, tuple[float, float, str]]:
        return {
            "mass_kg": (40, 150, "kg"),
            "blood_volume_l": (3, 8, "L"),
            "hematocrit": (0.2, 0.6, "1"),
            "hemoglobin_g_dl": (6, 20, "g/dL"),
            "interstitial_volume_l": (6, 25, "L"),
            "heart_rate_bpm": (40, 110, "1/min"),
            "cardiac_output_l_min": (2.5, 8, "L/min"),
            "mean_pressure_mmhg": (65, 120, "mmHg"),
            "oxygen_consumption_ml_min": (120, 500, "mL STPD/min"),
            "respiratory_quotient": (0.65, 1.0, "1"),
            "alveolar_volume_l": (1.5, 6, "L BTPS"),
            "shunt_fraction": (0, 0.35, "1"),
            "renal_filtration_ml_min": (15, 180, "mL/min"),
            "autonomic_tau_s": (2, 60, "s"),
            "ventilation_tau_s": (2, 60, "s"),
            "adh_tau_s": (20, 600, "s"),
        }


@dataclass(frozen=True)
class EnvironmentConditions:
    oxygen_fraction: float = 0.2095
    co2_fraction: float = 0.0004
    atmospheric_pressure_mmhg: float = 760.0
    temperature_c: float = 22.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("environment must be finite")
        if not 0.1 <= self.oxygen_fraction <= 1 or not 0 <= self.co2_fraction <= 0.03:
            raise ValueError("gas fractions outside model envelope")
        if self.oxygen_fraction + self.co2_fraction > 1:
            raise ValueError("gas fractions sum above one")
        if not 400 <= self.atmospheric_pressure_mmhg <= 800:
            raise ValueError("pressure outside 400..800 mmHg")
        if not 5 <= self.temperature_c <= 40:
            raise ValueError("temperature outside 5..40 C")


@dataclass(frozen=True)
class EquipmentState:
    oxygen_fraction: float | None = None
    ventilation_l_min: float | None = None
    peep_cmh2o: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.oxygen_fraction is not None and (
            not math.isfinite(self.oxygen_fraction) or not 0.1 <= self.oxygen_fraction <= 1
        ):
            raise ValueError("equipment FiO2 outside envelope")
        if self.ventilation_l_min is not None and (
            not math.isfinite(self.ventilation_l_min) or not 0.1 <= self.ventilation_l_min <= 40
        ):
            raise ValueError("equipment alveolar ventilation outside 0.1..40 L/min")
        if not math.isfinite(self.peep_cmh2o) or not 0 <= self.peep_cmh2o <= 20:
            raise ValueError("PEEP outside 0..20 cmH2O")


@dataclass
class PhysiologyState:
    plasma_volume_l: float
    red_cell_volume_l: float
    interstitial_volume_l: float
    arterial_o2_ml: float
    venous_o2_ml: float
    arterial_co2_ml: float
    venous_co2_ml: float
    alveolar_o2_ml: float
    alveolar_co2_ml: float
    mean_pressure_mmhg: float
    sympathetic: float
    adh: float
    ventilation_l_min: float
    sodium_mmol: float
    lactate_mmol: float
    oxygen_debt_ml: float = 0.0
    urine_l: float = 0.0
    blood_loss_l: float = 0.0
    infused_l: float = 0.0
    maintenance_l: float = 0.0
    sodium_input_mmol: float = 0.0
    sodium_output_mmol: float = 0.0
    oxygen_inspired_ml: float = 0.0
    oxygen_expired_ml: float = 0.0
    oxygen_consumed_ml: float = 0.0
    oxygen_lost_ml: float = 0.0
    co2_inspired_ml: float = 0.0
    co2_expired_ml: float = 0.0
    co2_produced_ml: float = 0.0
    co2_lost_ml: float = 0.0

    def vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    @classmethod
    def from_vector(cls, values: tuple[float, ...]) -> PhysiologyState:
        return cls(*values)


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    version: str
    fidelity: str
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    parameter_ranges: Mapping[str, tuple[float, float, str]]
    operating_envelope: Mapping[str, tuple[float, float, str]]
    evidence_class: str = "MODEL"
    reality_level: int = 1
    source: str = "PhysioWorld original lumped reduction of mass balance/Fick/Hill laws"
    calibration_data: tuple[str, ...] = ()
    validation_evidence: tuple[str, ...] = ()
    reduction_error_criterion: str = (
        "No claimed equivalence to a full-order human model; assess requested outputs "
        "by step-halving against declared numerical tolerances."
    )
    coupling: Mapping[str, Any] = field(
        default_factory=lambda: {
            "state": "two blood gas pools, alveolar gas, fluid/RBC/Na, regulation",
            "inputs": "environment, equipment, exercise, bleeding, infusion, posture, rhythm",
            "outputs": "pressure, perfusion, gases, demand, filtration, observations",
            "clock": "seconds, continuous; forcing at integer seconds; actions at exact times",
            "direct_feedthrough": "actions -> flow/demand; gas and fluid stores remain continuous",
            "initialization": "analytical resting gas balance for declared patient/environment",
            "algebraic_constraints": "blood volume=plasma+RBC; CaO2=Hill(PaO2,Hb)",
            "events": "scheduled parameter/mode changes and conservative fluid impulses",
            "solver": "RK4 step-doubling, positivity rejection, exact scheduled boundaries",
        }
    )


class PhysiologyModel(Protocol):
    """Pure RHS boundary for alternate solvers/native bindings."""

    descriptor: ModelDescriptor

    def initial_state(self, environment: EnvironmentConditions) -> PhysiologyState: ...

    def derivative(
        self,
        state: PhysiologyState,
        environment: EnvironmentConditions,
        equipment: EquipmentState,
        controls: Mapping[str, float],
        forcing: float = 0.0,
    ) -> tuple[float, ...]: ...


def oxygen_content(po2_mmhg: float, hemoglobin_g_dl: float) -> float:
    """mL STPD O2/L blood; Hill n=2.7, P50=26.8 mmHg, fixed pH/temperature."""
    p = max(0.0, po2_mmhg)
    saturation = p**2.7 / (26.8**2.7 + p**2.7)
    return 13.4 * hemoglobin_g_dl * saturation + 0.03 * p


def oxygen_pressure(content_ml_l: float, hemoglobin_g_dl: float) -> float:
    # Monotone inversion, including dissolved oxygen above Hb saturation.
    low, high = 0.0, 5000.0
    for _ in range(36):
        middle = (low + high) / 2
        if oxygen_content(middle, hemoglobin_g_dl) < content_ml_l:
            low = middle
        else:
            high = middle
    return (low + high) / 2


VARIABLE_UNITS = {
    "blood_volume_l": "L",
    "plasma_volume_l": "L",
    "red_cell_volume_l": "L",
    "interstitial_volume_l": "L",
    "heart_rate_bpm": "1/min",
    "mean_pressure_mmhg": "mmHg",
    "systolic_pressure_mmhg": "mmHg",
    "diastolic_pressure_mmhg": "mmHg",
    "cardiac_output_l_min": "L/min",
    "stroke_volume_ml": "mL",
    "systemic_resistance_mmhg_min_l": "mmHg min/L",
    "hemoglobin_g_dl": "g/dL",
    "arterial_po2_mmhg": "mmHg",
    "arterial_pco2_mmhg": "mmHg",
    "venous_po2_mmhg": "mmHg",
    "oxygen_saturation": "1",
    "oxygen_delivery_ml_min": "mL STPD/min",
    "oxygen_consumption_ml_min": "mL STPD/min",
    "oxygen_demand_ml_min": "mL STPD/min",
    "alveolar_po2_mmhg": "mmHg",
    "alveolar_pco2_mmhg": "mmHg",
    "ventilation_l_min": "L BTPS/min",
    "respiratory_rate_bpm": "1/min",
    "renal_filtration_ml_min": "mL/min",
    "urine_flow_ml_min": "mL/min",
    "sodium_mmol_l": "mmol/L",
    "lactate_mmol_l": "mmol/L",
    "sympathetic": "1",
    "adh": "1",
    "oxygen_debt_ml": "mL STPD",
    "urine_l": "L",
    "blood_loss_l": "L",
    "infused_l": "L",
}


class CoupledCardiorespiratoryModel:
    """A conservative gas/fluid kernel with explicit, approximate controllers."""

    def __init__(self, patient: PatientParameters, fidelity: str = "coupled"):
        if fidelity not in {"coupled", "open_loop"}:
            raise ValueError("fidelity must be coupled or open_loop")
        self.patient = patient
        self.fidelity = fidelity
        self.descriptor = ModelDescriptor(
            name=f"cardiorespiratory_{fidelity}",
            version=MODEL_VERSION,
            fidelity=fidelity,
            equations=(
                "Q=HR*SV/1000; SV=SV0*preload*contractility*tachycardia_factor",
                "dMAP/dt=(CVP+Q*SVR-MAP)/1.5 s",
                "Ca*= (1-shunt)*C_Hill(PAO2,Hb)+shunt*Cv",
                "dMaO2/dt=Q/60*(Ca*-Ca)-bleed*Ca*arterial_fraction",
                "dMvO2/dt=Q/60*(Ca-Cv)-VO2/60-bleed*Cv*venous_fraction",
                "dMlungO2/dt=VA_STPD/60*(FiO2-FAO2)-Q/60*(Ca*-Cv)",
                "CO2 blood content=max(0,240+6*PCO2) mL/L; analogous gas balances",
                "VO2=min(demand,0.85*Q*Ca); VCO2=RQ*VO2",
                "dVplasma=infusion+maintenance-urine-bleed_plasma-capillary_flux",
                "dVrbc=-bleed_rbc; dVinterstitial=capillary_flux",
                "dNa=intake+154*infusion-[Na]*(urine+bleed_plasma)",
                "baroreflex/chemoreflex and ADH: bounded first-order assumed controllers",
            ),
            assumptions=(
                "Adult nonpregnant lumped compartments; no ventricular valves or beat resolution.",
                "O2 Hill curve fixed pH/P50; linear CO2 content, no acid-base/Haldane model.",
                "Gas stores effective blood stores; RBC loss changes Hb, crystalloid dilutes it.",
                "No dissolved-gas transport with renal/capillary water flux; hemorrhage carries gases.",
                "PEEP reduces preload; no lung recruitment/compliance or ventilator waveform.",
                "Oxygen equipment blends CO2-free gas with ambient air; inspired CO2 scales with ambient fraction.",
                "Renal filtration/ADH/capillary refill and lactate are assumed phenomenology.",
                "Ambient temperature changes metabolic demand, not a body heat-balance model.",
                "open_loop freezes autonomic/ADH feedback and removes chemical ventilatory feedback."
                if fidelity == "open_loop"
                else "Autonomic/ADH/ventilation feedback enabled.",
                "Not calibrated or clinically validated; simulations do not identify human causal effects.",
            ),
            parameter_ranges=patient.ranges(),
            operating_envelope={
                "blood_volume_ratio": (0.45, 1.6, "1"),
                "mean_pressure_mmhg": (20, 180, "mmHg"),
                "arterial_po2_mmhg": (15, 750, "mmHg"),
                "arterial_pco2_mmhg": (10, 120, "mmHg"),
                "sodium_mmol_l": (115, 165, "mmol/L"),
                "horizon_s": (0, 21600, "s"),
            },
        )

    @property
    def resting_ventilation(self) -> float:
        p = self.patient
        return 0.863 * p.oxygen_consumption_ml_min * p.respiratory_quotient / 40

    @staticmethod
    def lung_capacity(environment: EnvironmentConditions, volume_l: float) -> float:
        return volume_l * 1000 * (environment.atmospheric_pressure_mmhg - 47) / 863

    def initial_state(self, environment: EnvironmentConditions) -> PhysiologyState:
        p = self.patient
        dry = environment.atmospheric_pressure_mmhg - 47
        ventilation = self.resting_ventilation * 1000 * dry / 863
        fo2 = environment.oxygen_fraction - p.oxygen_consumption_ml_min / ventilation
        fco2 = environment.co2_fraction + (
            p.oxygen_consumption_ml_min * p.respiratory_quotient / ventilation
        )
        if fo2 <= 0 or fo2 + fco2 >= 1:
            raise ValueError("initial environment cannot support assumed resting gas balance")
        ca = oxygen_content(fo2 * dry, p.hemoglobin_g_dl) - (
            p.shunt_fraction
            * p.oxygen_consumption_ml_min
            / (p.cardiac_output_l_min * (1 - p.shunt_fraction))
        )
        cv = ca - p.oxygen_consumption_ml_min / p.cardiac_output_l_min
        cca = (
            240
            + 6 * fco2 * dry
            + (
                p.shunt_fraction
                * p.oxygen_consumption_ml_min
                * p.respiratory_quotient
                / (p.cardiac_output_l_min * (1 - p.shunt_fraction))
            )
        )
        ccv = cca + p.oxygen_consumption_ml_min * p.respiratory_quotient / p.cardiac_output_l_min
        plasma = p.blood_volume_l * (1 - p.hematocrit)
        ecf = plasma + p.interstitial_volume_l
        capacity = self.lung_capacity(environment, p.alveolar_volume_l)
        return PhysiologyState(
            plasma,
            p.blood_volume_l * p.hematocrit,
            p.interstitial_volume_l,
            ca * p.blood_volume_l * 0.2,
            cv * p.blood_volume_l * 0.8,
            cca * p.blood_volume_l * 0.2,
            ccv * p.blood_volume_l * 0.8,
            fo2 * capacity,
            fco2 * capacity,
            p.mean_pressure_mmhg,
            0.25,
            0.2,
            self.resting_ventilation,
            ecf * 140,
            ecf,
        )

    def outputs(
        self,
        s: PhysiologyState,
        env: EnvironmentConditions,
        eq: EquipmentState,
        c: Mapping[str, float],
    ) -> dict[str, float]:
        p = self.patient
        volume = s.plasma_volume_l + s.red_cell_volume_l
        hb = p.hemoglobin_g_dl * s.red_cell_volume_l / (p.blood_volume_l * p.hematocrit)
        hb *= p.blood_volume_l / volume
        ca, cv = s.arterial_o2_ml / (0.2 * volume), s.venous_o2_ml / (0.8 * volume)
        pco2 = (s.arterial_co2_ml / (0.2 * volume) - 240) / 6
        dry = env.atmospheric_pressure_mmhg - 47
        capacity = self.lung_capacity(env, p.alveolar_volume_l)
        pao2, paco2 = s.alveolar_o2_ml / capacity * dry, s.alveolar_co2_ml / capacity * dry
        pool = 0.10 * c.get("orthostasis", 0) + 0.002 * c.get("lbnp_mmhg", 0)
        peep = eq.peep_cmh2o if eq.enabled else 0.0
        preload = bounded((volume / p.blood_volume_l - pool - 0.007 * peep) ** 3, 0.05, 1.7)
        heart_rate = c.get("arrhythmia_rate_bpm", 0) or (
            p.heart_rate_bpm * (0.85 + 0.6 * s.sympathetic) + 8 * (c.get("exercise_mets", 1) - 1)
        )
        stroke = p.cardiac_output_l_min * 1000 / p.heart_rate_bpm * preload
        stroke *= (1 + 0.4 * (s.sympathetic - 0.25)) * c.get("contractility", 1)
        stroke *= c.get("rhythm_efficiency", 1) * min(1.0, 100 / heart_rate)
        flow = heart_rate * stroke / 1000
        workload = c.get("exercise_mets", 1)
        resistance = (p.mean_pressure_mmhg - 5) / p.cardiac_output_l_min
        resistance *= (0.8 + 0.8 * s.sympathetic) / (1 + 0.15 * (workload - 1))
        demand = p.oxygen_consumption_ml_min * workload * (1 + 0.006 * abs(env.temperature_c - 22))
        consumption = min(demand, max(0.0, 0.85 * flow * ca))
        filtration = p.renal_filtration_ml_min * bounded((s.mean_pressure_mmhg - 25) / 65, 0, 1.5)
        filtration *= bounded(flow / p.cardiac_output_l_min, 0, 1.3)
        ecf = s.plasma_volume_l + s.interstitial_volume_l
        excess = max(0.0, ecf - (p.blood_volume_l * (1 - p.hematocrit) + p.interstitial_volume_l))
        urine = (0.8 + 4 * excess) * (1.2 - s.adh) * filtration / p.renal_filtration_ml_min
        po2 = oxygen_pressure(ca, hb)
        saturation = po2**2.7 / (26.8**2.7 + po2**2.7)
        pulse_pressure = stroke / 1.6  # assumed lumped arterial compliance mL/mmHg
        ventilation = (
            eq.ventilation_l_min
            if eq.enabled and eq.ventilation_l_min is not None
            else s.ventilation_l_min
        )
        return {
            "blood_volume_l": volume,
            "plasma_volume_l": s.plasma_volume_l,
            "red_cell_volume_l": s.red_cell_volume_l,
            "interstitial_volume_l": s.interstitial_volume_l,
            "heart_rate_bpm": heart_rate,
            "mean_pressure_mmhg": s.mean_pressure_mmhg,
            "systolic_pressure_mmhg": s.mean_pressure_mmhg + 2 * pulse_pressure / 3,
            "diastolic_pressure_mmhg": s.mean_pressure_mmhg - pulse_pressure / 3,
            "stroke_volume_ml": stroke,
            "cardiac_output_l_min": flow,
            "systemic_resistance_mmhg_min_l": resistance,
            "hemoglobin_g_dl": hb,
            "arterial_po2_mmhg": po2,
            "arterial_pco2_mmhg": pco2,
            "venous_po2_mmhg": oxygen_pressure(cv, hb),
            "oxygen_saturation": saturation,
            "oxygen_delivery_ml_min": flow * ca,
            "oxygen_consumption_ml_min": consumption,
            "oxygen_demand_ml_min": demand,
            "alveolar_po2_mmhg": pao2,
            "alveolar_pco2_mmhg": paco2,
            "ventilation_l_min": ventilation,
            "respiratory_rate_bpm": 12
            * math.sqrt(max(0.01, ventilation / self.resting_ventilation)),
            "renal_filtration_ml_min": filtration,
            "urine_flow_ml_min": urine,
            "sodium_mmol_l": s.sodium_mmol / ecf,
            "lactate_mmol_l": s.lactate_mmol / ecf,
            "sympathetic": s.sympathetic,
            "adh": s.adh,
            "oxygen_debt_ml": s.oxygen_debt_ml,
            "urine_l": s.urine_l,
            "blood_loss_l": s.blood_loss_l,
            "infused_l": s.infused_l,
        }

    def derivative(
        self,
        s: PhysiologyState,
        env: EnvironmentConditions,
        eq: EquipmentState,
        c: Mapping[str, float],
        forcing: float = 0.0,
    ) -> tuple[float, ...]:
        p = self.patient
        o = self.outputs(s, env, eq, c)
        volume = o["blood_volume_l"]
        ca, cv = s.arterial_o2_ml / (0.2 * volume), s.venous_o2_ml / (0.8 * volume)
        cca, ccv = s.arterial_co2_ml / (0.2 * volume), s.venous_co2_ml / (0.8 * volume)
        q = o["cardiac_output_l_min"] / 60
        shunt = p.shunt_fraction
        cap_o2 = (1 - shunt) * oxygen_content(
            o["alveolar_po2_mmhg"], o["hemoglobin_g_dl"]
        ) + shunt * cv
        cap_co2 = (1 - shunt) * max(0, 240 + 6 * o["alveolar_pco2_mmhg"]) + shunt * ccv
        lung_o2, lung_co2 = q * (cap_o2 - cv), q * (ccv - cap_co2)
        consumed = o["oxygen_consumption_ml_min"] / 60
        produced = consumed * p.respiratory_quotient
        ventilation = (
            o["ventilation_l_min"] * 1000 * (env.atmospheric_pressure_mmhg - 47) / 863 / 60
        )
        fio2 = (
            eq.oxygen_fraction
            if eq.enabled and eq.oxygen_fraction is not None
            else env.oxygen_fraction
        )
        fico2 = env.co2_fraction
        if eq.enabled and eq.oxygen_fraction is not None:
            # CO2-free oxygen enrichment (or nitrogen dilution) of ambient air.
            ambient_fraction = (
                (1 - fio2) / (1 - env.oxygen_fraction)
                if fio2 >= env.oxygen_fraction and env.oxygen_fraction < 1
                else fio2 / env.oxygen_fraction
            )
            fico2 *= ambient_fraction
        capacity = self.lung_capacity(env, p.alveolar_volume_l)
        oi, oe = ventilation * fio2, ventilation * s.alveolar_o2_ml / capacity
        ci, ce = ventilation * fico2, ventilation * s.alveolar_co2_ml / capacity
        bleed = c.get("hemorrhage_ml_min", 0) / 60000
        bp, br = bleed * s.plasma_volume_l / volume, bleed * s.red_cell_volume_l / volume
        infusion = c.get("infusion_ml_min", 0) / 60000
        maintenance = c.get("maintenance_ml_min", 0.8) / 60000
        urine = o["urine_flow_ml_min"] / 60000
        baseline_plasma = p.blood_volume_l * (1 - p.hematocrit)
        exchange = 0.003 * (
            s.plasma_volume_l
            - baseline_plasma
            - 0.15 * (s.interstitial_volume_l - p.interstitial_volume_l)
        )
        hypoxia = max(0.0, 0.94 - o["oxygen_saturation"])
        sym_target = bounded(
            0.25
            + 0.8 * (p.mean_pressure_mmhg - s.mean_pressure_mmhg) / p.mean_pressure_mmhg
            + 1.5 * hypoxia
            + 0.04 * (c.get("exercise_mets", 1) - 1)
            + forcing,
            0.02,
            0.98,
        )
        adh_target = bounded(
            0.2 + 1.5 * (1 - volume / p.blood_volume_l) + 0.8 * (o["sodium_mmol_l"] / 140 - 1),
            0.01,
            0.99,
        )
        # Fixed normoxic reference, rather than moving the chemoreflex setpoint
        # with an environmental CO2 intervention.
        resting_pco2 = (
            40.2852
            + p.shunt_fraction
            * p.oxygen_consumption_ml_min
            * p.respiratory_quotient
            / (6 * p.cardiac_output_l_min * (1 - p.shunt_fraction))
        )
        ventilation_target = self.resting_ventilation * (
            1
            + 0.65 * (c.get("exercise_mets", 1) - 1)
            + (
                0.035 * (o["arterial_pco2_mmhg"] - resting_pco2) + 3 * hypoxia
                if self.fidelity == "coupled"
                else 0
            )
        )
        ecf = s.plasma_volume_l + s.interstitial_volume_l
        deficit = max(0.0, o["oxygen_demand_ml_min"] - o["oxygen_consumption_ml_min"]) / 60
        na_in, na_out = 154 * infusion + 140 * maintenance, o["sodium_mmol_l"] * (urine + bp)
        d = {
            "plasma_volume_l": infusion + maintenance - urine - bp - exchange,
            "red_cell_volume_l": -br,
            "interstitial_volume_l": exchange,
            "arterial_o2_ml": q * (cap_o2 - ca) - 0.2 * bleed * ca,
            "venous_o2_ml": q * (ca - cv) - consumed - 0.8 * bleed * cv,
            "arterial_co2_ml": q * (cap_co2 - cca) - 0.2 * bleed * cca,
            "venous_co2_ml": q * (cca - ccv) + produced - 0.8 * bleed * ccv,
            "alveolar_o2_ml": oi - oe - lung_o2,
            "alveolar_co2_ml": ci - ce + lung_co2,
            "mean_pressure_mmhg": (
                5 * volume / p.blood_volume_l
                + q * 60 * o["systemic_resistance_mmhg_min_l"]
                - s.mean_pressure_mmhg
            )
            / 1.5,
            "sympathetic": (sym_target - s.sympathetic) / p.autonomic_tau_s
            if self.fidelity == "coupled"
            else 0,
            "adh": (adh_target - s.adh) / p.adh_tau_s if self.fidelity == "coupled" else 0,
            "ventilation_l_min": (bounded(ventilation_target, 0.1, 40) - s.ventilation_l_min)
            / p.ventilation_tau_s,
            "sodium_mmol": na_in - na_out,
            "lactate_mmol": 0.008 * deficit
            - (s.lactate_mmol - ecf)
            * bounded(o["cardiac_output_l_min"] / p.cardiac_output_l_min, 0, 1)
            / 300,
            "oxygen_debt_ml": deficit,
            "urine_l": urine,
            "blood_loss_l": bleed,
            "infused_l": infusion,
            "maintenance_l": maintenance,
            "sodium_input_mmol": na_in,
            "sodium_output_mmol": na_out,
            "oxygen_inspired_ml": oi,
            "oxygen_expired_ml": oe,
            "oxygen_consumed_ml": consumed,
            "oxygen_lost_ml": bleed * (0.2 * ca + 0.8 * cv),
            "co2_inspired_ml": ci,
            "co2_expired_ml": ce,
            "co2_produced_ml": produced,
            "co2_lost_ml": bleed * (0.2 * cca + 0.8 * ccv),
        }
        return tuple(d[f.name] for f in fields(s))


MODEL_REGISTRY = {
    name: CoupledCardiorespiratoryModel(PatientParameters(), name).descriptor
    for name in ("coupled", "open_loop")
}
