"""JSON scenarios, bounded-memory sampling, restartable populations and sweeps."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .engine import PhysiologicalAction, PhysiologyEngine
from .information import DataRequest, Observation
from .models import EnvironmentConditions, EquipmentState, PatientParameters, bounded


@dataclass(frozen=True)
class ScenarioResult:
    observations: tuple[Observation, ...]
    manifest: Mapping[str, Any]
    final_checkpoint: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "manifest": dict(self.manifest),
            "final_checkpoint": dict(self.final_checkpoint),
        }


@dataclass(frozen=True)
class PhysiologicalScenario:
    name: str = "resting_adult"
    duration_s: float = 60.0
    sample_interval_s: float = 1.0
    patient: PatientParameters = field(default_factory=PatientParameters)
    environment: EnvironmentConditions = field(default_factory=EnvironmentConditions)
    equipment: EquipmentState = field(default_factory=EquipmentState)
    requests: DataRequest = field(default_factory=DataRequest)
    actions: tuple[PhysiologicalAction, ...] = ()
    seed: int = 0
    fidelity: str = "coupled"
    solver: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    resource_units: Mapping[str, str] = field(default_factory=dict)
    process_noise_std: float = 0.0
    observation_noise: Mapping[str, float] = field(default_factory=dict)
    envelope_policy: str = "record"
    available_actions: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name required")
        if not math.isfinite(self.duration_s) or not 0 <= self.duration_s <= 21600:
            raise ValueError("scenario duration outside 0..21600 seconds")
        if not math.isfinite(self.sample_interval_s) or self.sample_interval_s <= 0:
            raise ValueError("sample interval must be finite positive seconds")
        if any(a.time_s > self.duration_s for a in self.actions):
            raise ValueError("action scheduled beyond scenario duration")
        if not set(self.solver) <= {"max_step_s", "rtol", "atol"}:
            raise ValueError("unknown solver settings")
        object.__setattr__(self, "actions", tuple(self.actions))

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> PhysiologicalScenario:
        c = copy.deepcopy(dict(config))
        schema = c.pop("schema", "physioworld.physiology.scenario/1")
        if schema != "physioworld.physiology.scenario/1":
            raise ValueError("unsupported scenario schema")
        for name, constructor in (
            ("patient", PatientParameters),
            ("environment", EnvironmentConditions),
            ("equipment", EquipmentState),
            ("requests", DataRequest),
        ):
            if name in c:
                if name == "requests" and isinstance(c[name], list):
                    c[name] = {"variables": c[name]}
                c[name] = constructor(**c[name])
        if "actions" in c:
            c["actions"] = tuple(PhysiologicalAction(**a) for a in c["actions"])
        if c.get("available_actions") is not None:
            c["available_actions"] = tuple(c["available_actions"])
        return cls(**c)

    @classmethod
    def from_json(cls, source: str | Path) -> PhysiologicalScenario:
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            config = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            config = json.loads(source)
        return cls.from_dict(config)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "physioworld.physiology.scenario/1", **asdict(self)}

    def create_engine(self) -> PhysiologyEngine:
        engine = PhysiologyEngine(
            self.patient,
            seed=self.seed,
            fidelity=self.fidelity,
            environment=self.environment,
            equipment=self.equipment,
            requests=self.requests,
            resources=self.resources,
            resource_units=self.resource_units,
            process_noise_std=self.process_noise_std,
            observation_noise=self.observation_noise,
            envelope_policy=self.envelope_policy,
            available_actions=self.available_actions,
            **self.solver,
        )
        for action in self.actions:
            engine.schedule(action)
        return engine

    def runner(self) -> ScenarioRunner:
        return ScenarioRunner(self)

    def run(self) -> ScenarioResult:
        runner = self.runner()
        observations = tuple(o for chunk in runner.chunks(256) for o in chunk)
        return ScenarioResult(observations, runner.manifest(), runner.engine.checkpoint())


class ScenarioRunner:
    """Fixed absolute sampling chronology independent of chunk size and restart.

    ``run_chunk`` retains no observation history; the consumer chooses storage.
    Runner checkpoints additionally retain the next sample index/end-point flag.
    """

    def __init__(self, scenario: PhysiologicalScenario):
        self.scenario = scenario
        self.engine = scenario.create_engine()
        self.next_sample_index = 0
        self.finished = False

    def run_chunk(self, max_samples: int = 256) -> tuple[Observation, ...]:
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        result: list[Observation] = []
        while not self.finished and len(result) < max_samples:
            time = min(
                self.next_sample_index * self.scenario.sample_interval_s, self.scenario.duration_s
            )
            result.append(self.engine.advance_to(time))
            self.next_sample_index += 1
            self.finished = time == self.scenario.duration_s
        return tuple(result)

    def chunks(self, max_samples: int = 256) -> Iterator[tuple[Observation, ...]]:
        while not self.finished:
            yield self.run_chunk(max_samples)

    def manifest(self) -> dict[str, Any]:
        config = self.scenario.to_dict()
        return {
            **self.engine.manifest(),
            "scenario": config,
            "scenario_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "next_sample_index": self.next_sample_index,
            "finished": self.finished,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": "physioworld.physiology.runner/1",
            "scenario": self.scenario.to_dict(),
            "engine": self.engine.checkpoint(),
            "next_sample_index": self.next_sample_index,
            "finished": self.finished,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.checkpoint(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> ScenarioRunner:
        if checkpoint.get("schema") != "physioworld.physiology.runner/1":
            raise ValueError("unsupported runner checkpoint")
        obj = cls.__new__(cls)
        obj.scenario = PhysiologicalScenario.from_dict(checkpoint["scenario"])
        obj.engine = PhysiologyEngine.from_checkpoint(checkpoint["engine"])
        obj.next_sample_index = checkpoint["next_sample_index"]
        obj.finished = checkpoint["finished"]
        if not isinstance(obj.next_sample_index, int) or obj.next_sample_index < 0:
            raise ValueError("invalid sample cursor")
        expected = min(
            max(0, obj.next_sample_index - 1) * obj.scenario.sample_interval_s,
            obj.scenario.duration_s,
        )
        if obj.engine.time_s != expected or obj.finished != (
            obj.next_sample_index > 0 and expected == obj.scenario.duration_s
        ):
            raise ValueError("runner time/cursor mismatch")
        return obj

    @classmethod
    def load(cls, path: str | Path) -> ScenarioRunner:
        return cls.from_checkpoint(json.loads(Path(path).read_text(encoding="utf-8")))


def run_scenario(source: PhysiologicalScenario | Mapping[str, Any] | str | Path) -> ScenarioResult:
    if isinstance(source, PhysiologicalScenario):
        scenario = source
    elif isinstance(source, Mapping):
        scenario = PhysiologicalScenario.from_dict(source)
    else:
        scenario = PhysiologicalScenario.from_json(source)
    return scenario.run()


def sample_population(count: int, *, seed: int = 0) -> tuple[PatientParameters, ...]:
    """Assumed correlated synthetic adult population, not sampled participants."""
    if not isinstance(count, int) or count < 0:
        raise ValueError("population count must be nonnegative integer")
    rng = random.Random(seed)
    patients = []
    for i in range(count):
        mass = bounded(rng.gauss(75, 12), 45, 115)
        volume = bounded(mass * 0.0714 * rng.uniform(0.94, 1.06), 3, 8)
        hematocrit = bounded(rng.gauss(0.44, 0.025), 0.36, 0.52)
        patients.append(
            PatientParameters(
                subject_id=f"synthetic-{seed}-{i:04d}",
                mass_kg=mass,
                blood_volume_l=volume,
                hematocrit=hematocrit,
                hemoglobin_g_dl=hematocrit * 33.3,
                interstitial_volume_l=mass * 12 / 70,
                heart_rate_bpm=bounded(rng.gauss(72, 7), 50, 95),
                cardiac_output_l_min=5 * (mass / 70) ** 0.75,
                mean_pressure_mmhg=bounded(rng.gauss(90, 6), 70, 110),
                oxygen_consumption_ml_min=3.5 * mass,
                autonomic_tau_s=rng.uniform(5, 13),
                renal_filtration_ml_min=rng.uniform(80, 140),
            )
        )
    return tuple(patients)


def run_population(
    scenario: PhysiologicalScenario,
    patients: Sequence[PatientParameters],
    *,
    seeds: Sequence[int] | None = None,
) -> Iterator[ScenarioResult]:
    """Lazy patient-by-patient execution; deterministic and dependency-free."""
    if seeds is not None and len(seeds) != len(patients):
        raise ValueError("one seed per patient required")
    for index, patient in enumerate(patients):
        yield replace(
            scenario, patient=patient, seed=seeds[index] if seeds is not None else scenario.seed
        ).run()


def parameter_sweep(
    scenario: PhysiologicalScenario,
    grid: Mapping[str, Sequence[Any]],
) -> Iterator[tuple[dict[str, Any], ScenarioResult]]:
    """Cartesian sweep of named ``patient.*``, ``environment.*``, ``solver.*`` or seed.

    A common seed is retained by default; subject identity stays fixed so random
    pairing is retained across parameter counterfactuals. Results state their
    model-counterfactual regime; no observational causal identification implied.
    """
    names = tuple(sorted(grid))
    for values in itertools.product(*(grid[n] for n in names)):
        config = scenario.to_dict()
        parameters = dict(zip(names, values))
        for name, value in parameters.items():
            if name == "seed":
                config[name] = value
            else:
                prefix, key = name.split(".", 1)
                if prefix not in {"patient", "environment", "solver"}:
                    raise ValueError(f"unsupported sweep path: {name}")
                config[prefix][key] = value
        result = PhysiologicalScenario.from_dict(config).run()
        manifest = {
            **result.manifest,
            "sweep_parameters": parameters,
            "causal_status": "MODEL_COUNTERFACTUAL",
        }
        yield parameters, replace(result, manifest=manifest)


def golden_scenarios(duration_s: float = 60) -> dict[str, PhysiologicalScenario]:
    """Minimal physiology fixtures with directional/invariant expected behavior."""
    onset = duration_s / 4
    baseline = PhysiologicalScenario(duration_s=duration_s)

    def variant(name: str, kind: str, parameters: Mapping[str, Any]) -> PhysiologicalScenario:
        return replace(baseline, name=name, actions=(PhysiologicalAction(onset, kind, parameters),))

    return {
        "ideal": baseline,
        "noisy": replace(
            baseline,
            name="noisy",
            process_noise_std=0.01,
            observation_noise={"mean_pressure_mmhg": 0.5},
        ),
        "missing_modality": variant("missing_modality", "sensor", {"oxygen_saturation": False}),
        "orthostasis": variant("orthostasis", "orthostasis", {"orthostasis": 1}),
        "exercise": variant("exercise", "exercise", {"exercise_mets": 3}),
        "hemorrhage": variant("hemorrhage", "hemorrhage", {"hemorrhage_ml_min": 150}),
        "oxygen": variant("oxygen", "oxygen", {"oxygen_fraction": 0.4}),
        "arrhythmia": variant(
            "arrhythmia", "arrhythmia", {"arrhythmia_rate_bpm": 150, "rhythm_efficiency": 0.6}
        ),
        "resource_depletion": replace(
            baseline,
            name="resource_depletion",
            resources={"energy_j": 1.0},
            resource_units={"energy_j": "J"},
            requests=DataRequest(("resource.energy_j",)),
            actions=(PhysiologicalAction(onset, "resource", {"energy_j": 0}),),
        ),
    }


Scenario = PhysiologicalScenario
