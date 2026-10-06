"""Requirements, independent reference cases, current run receipts, and credibility."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Callable, Sequence
from uuid import uuid4

from .falsification import CounterexampleRegistry, FalsificationResult, FalsificationTemplate
from .numerics import Tolerance
from .validation import ValidationComparison


class FidelityLevel(IntEnum):
    MATHEMATICAL = 0
    SYNTHETIC = 1
    RETROSPECTIVE_REPLAY = 2
    HYBRID = 3
    HARDWARE_IN_THE_LOOP = 4
    PHYSICAL_TESTBED = 5
    HUMAN_OBSERVATIONAL = 6
    CONTROLLED_HUMAN_INTERVENTION = 7
    CLINICAL = 8
    DEPLOYMENT = 9


@dataclass(frozen=True)
class ReferenceCase:
    test_id: str
    context: str
    input_artifact: tuple[float, ...]
    implementation: Callable[[tuple[float, ...]], Sequence[float]]
    independent_reference: Callable[[tuple[float, ...]], Sequence[float]]
    reference_method: str
    tolerance: Tolerance
    requirement_ids: tuple[str, ...]
    scientific_status: str = "NUMERICAL_VERIFICATION"

    def __post_init__(self) -> None:
        if (
            not self.test_id
            or not self.context
            or not self.reference_method
            or not self.requirement_ids
            or self.implementation is self.independent_reference
            or any(not math.isfinite(v) for v in self.input_artifact)
            or self.scientific_status != "NUMERICAL_VERIFICATION"
        ):
            raise ValueError("reference case requires distinct implementations and traceable scope")

    @property
    def input_hash(self) -> str:
        encoded = json.dumps(
            (
                self.test_id,
                self.context,
                self.input_artifact,
                self.reference_method,
                self.tolerance.absolute,
                self.tolerance.relative,
            ),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReferenceResult:
    test_id: str
    input_hash: str
    observed: tuple[float, ...]
    reference: tuple[float, ...]
    residuals: tuple[float, ...]
    passed: bool
    status: str
    failure: str | None = None


@dataclass(frozen=True)
class VerificationRun:
    run_id: str
    executed_at: str
    software_version: str
    results: tuple[ReferenceResult, ...]
    falsifications: tuple[FalsificationResult, ...] = ()


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    statement: str
    test_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...] = ()


class ReferenceRegistry:
    def __init__(self) -> None:
        self.cases: dict[str, ReferenceCase] = {}
        self.requirements: dict[str, Requirement] = {}
        self.counterexamples = CounterexampleRegistry()

    def add_case(self, case: ReferenceCase) -> None:
        if case.test_id in self.cases:
            raise ValueError("duplicate reference test ID")
        self.cases[case.test_id] = case

    def add_requirement(self, requirement: Requirement) -> None:
        if (
            not requirement.requirement_id
            or not requirement.statement
            or not requirement.test_ids
            or requirement.requirement_id in self.requirements
        ):
            raise ValueError("unique traceable requirement required")
        self.requirements[requirement.requirement_id] = requirement

    def add_falsification(self, template: FalsificationTemplate) -> None:
        if any(test_id not in self.cases for test_id in template.reference_test_ids) or any(
            r not in self.requirements for r in template.requirement_ids
        ):
            raise ValueError(
                "counterexample links must reference registered tests and requirements"
            )
        self.counterexamples.register(template)

    def run(
        self, *, software_version: str, test_ids: Sequence[str] | None = None
    ) -> VerificationRun:
        """Always execute actual implementations and independent references now."""
        if not software_version:
            raise ValueError("software version is required")
        selected = tuple(self.cases) if test_ids is None else tuple(test_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("nonempty unique reference tests required")
        results = []
        for test_id in selected:
            if test_id not in self.cases:
                raise ValueError(f"unknown reference test: {test_id}")
            case = self.cases[test_id]
            try:
                reference = tuple(case.independent_reference(case.input_artifact))
                observed = tuple(case.implementation(case.input_artifact))
                valid = (
                    bool(reference)
                    and len(observed) == len(reference)
                    and all(math.isfinite(v) for v in (*reference, *observed))
                )
                passed = valid and all(
                    case.tolerance.accepts(v, r) for v, r in zip(observed, reference)
                )
                residuals = tuple(v - r for v, r in zip(observed, reference))
                results.append(
                    ReferenceResult(
                        test_id,
                        case.input_hash,
                        observed,
                        reference,
                        residuals,
                        passed,
                        "PASSED_CURRENT_REFERENCE" if passed else "FAILED",
                        None if valid else "invalid shape or nonfinite result",
                    )
                )
            except Exception as error:
                results.append(
                    ReferenceResult(
                        test_id,
                        case.input_hash,
                        (),
                        (),
                        (),
                        False,
                        "FAILED",
                        f"{type(error).__name__}: {error}",
                    )
                )
        falsifications = tuple(
            template.run()
            for template in self.counterexamples.templates.values()
            if set(template.reference_test_ids) <= set(selected)
        )
        return VerificationRun(
            str(uuid4()),
            datetime.now(timezone.utc).isoformat(),
            software_version,
            tuple(results),
            falsifications,
        )


@dataclass(frozen=True)
class EnvelopeBound:
    name: str
    lower: float
    upper: float
    unit: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.unit
            or not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower > self.upper
        ):
            raise ValueError("named finite operating envelope bounds with units required")


@dataclass(frozen=True)
class ContextOfUse:
    context_id: str
    intended_use: str
    consequence_of_error: str
    risk_class: str
    coordinates: tuple[tuple[str, float, str], ...]
    fidelity: FidelityLevel = FidelityLevel.MATHEMATICAL


@dataclass(frozen=True)
class ValidationEvidence:
    evidence_id: str
    context_id: str
    source: str
    evidence_class: str
    data_support: str
    envelope: tuple[EnvelopeBound, ...]
    limitations: tuple[str, ...]
    comparison: ValidationComparison | None = None
    verification_test_ids: tuple[str, ...] = ()
    fidelity: FidelityLevel = FidelityLevel.MATHEMATICAL


@dataclass(frozen=True)
class CredibilityAssessment:
    context: ContextOfUse
    status: str
    reasons: tuple[str, ...]
    verification: VerificationRun
    validation: tuple[ValidationEvidence, ...]
    uncertainty_refs: tuple[str, ...]
    discrepancy_refs: tuple[str, ...]
    sensitivity_refs: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CredibilityCase:
    context: ContextOfUse
    requirement_ids: tuple[str, ...]
    validation: tuple[ValidationEvidence, ...]
    uncertainty_refs: tuple[str, ...]
    discrepancy_refs: tuple[str, ...]
    sensitivity_refs: tuple[str, ...]
    limitations: tuple[str, ...]

    def assess(
        self, registry: ReferenceRegistry, *, software_version: str
    ) -> CredibilityAssessment:
        """Run fresh checks and evaluate evidence only in this declared context of use."""
        context = self.context
        if (
            not context.context_id
            or not context.intended_use
            or not context.consequence_of_error
            or context.risk_class not in ("low", "medium", "high")
            or not isinstance(context.fidelity, FidelityLevel)
            or not self.requirement_ids
        ):
            raise ValueError("explicit risk-informed context and requirements required")
        tests = []
        for requirement_id in self.requirement_ids:
            if requirement_id not in registry.requirements:
                raise ValueError(f"unregistered requirement: {requirement_id}")
            requirement = registry.requirements[requirement_id]
            for template_id in requirement.counterexample_ids:
                template = registry.counterexamples.templates.get(template_id)
                if template is None or requirement_id not in template.requirement_ids:
                    raise ValueError("requirement/counterexample linkage is incomplete")
                for test_id in template.reference_test_ids:
                    if test_id not in tests:
                        tests.append(test_id)
            for test_id in requirement.test_ids:
                if (
                    test_id not in registry.cases
                    or requirement_id not in registry.cases[test_id].requirement_ids
                ):
                    raise ValueError("requirement/reference linkage is incomplete")
                if test_id not in tests:
                    tests.append(test_id)
        run = registry.run(software_version=software_version, test_ids=tests)
        reasons = [
            f"reference failed: {result.test_id}" for result in run.results if not result.passed
        ]
        reasons += [
            f"requirement falsified: {result.claim_id}"
            for result in run.falsifications
            if result.claim_id in self.requirement_ids and result.status == "FALSIFIED"
        ]
        coordinates = {name: (value, unit) for name, value, unit in context.coordinates}
        if (
            len(coordinates) != len(context.coordinates)
            or not coordinates
            or any(
                not name or not unit or not math.isfinite(value)
                for name, value, unit in context.coordinates
            )
        ):
            raise ValueError("unique finite context coordinates with units required")
        supported = []
        passed_tests = {r.test_id for r in run.results if r.passed}
        for evidence in self.validation:
            if (
                not evidence.evidence_id
                or not evidence.source
                or not evidence.data_support
                or evidence.context_id != context.context_id
                or not evidence.envelope
                or not isinstance(evidence.fidelity, FidelityLevel)
                or evidence.fidelity < context.fidelity
                or len({b.name for b in evidence.envelope}) != len(evidence.envelope)
            ):
                continue
            if (
                evidence.evidence_class in ("THEOREM", "NUMERICAL_VERIFICATION")
                and evidence.fidelity != FidelityLevel.MATHEMATICAL
            ):
                continue
            if (
                evidence.evidence_class == "SIMULATION"
                and evidence.fidelity > FidelityLevel.SYNTHETIC
            ):
                continue
            if evidence.comparison is not None and (
                not evidence.comparison.passed
                or evidence.comparison.software_version != software_version
                or evidence.comparison.evidence_class != evidence.evidence_class
            ):
                continue
            if evidence.evidence_class in ("EMPIRICAL_ESTIMATE", "EXTERNAL_VALIDATION"):
                comparison = evidence.comparison
                if (
                    comparison is None
                    or not comparison.passed
                    or comparison.software_version != software_version
                    or comparison.evidence_class != evidence.evidence_class
                ):
                    continue
            elif evidence.evidence_class in ("THEOREM", "NUMERICAL_VERIFICATION", "SIMULATION"):
                if (
                    not evidence.verification_test_ids
                    or not set(evidence.verification_test_ids) <= passed_tests
                ):
                    continue
            else:
                continue
            if set(coordinates) != {b.name for b in evidence.envelope}:
                continue
            if all(
                b.unit == coordinates[b.name][1] and b.lower <= coordinates[b.name][0] <= b.upper
                for b in evidence.envelope
            ):
                supported.append(evidence)
        if not supported:
            reasons.append(
                "no validation data support covering this context and operating envelope"
            )
        if context.risk_class == "high" and not any(
            e.evidence_class in ("EMPIRICAL_ESTIMATE", "EXTERNAL_VALIDATION") for e in supported
        ):
            reasons.append("high-consequence use lacks observed validation evidence")
        for name, refs in (
            ("uncertainty", self.uncertainty_refs),
            ("discrepancy", self.discrepancy_refs),
            ("sensitivity", self.sensitivity_refs),
        ):
            if not refs:
                reasons.append(f"no declared {name} evidence")
        status = "SUPPORTED_WITHIN_DECLARED_CONTEXT" if not reasons else "CONTEXT_CLAIM_UNSUPPORTED"
        return CredibilityAssessment(
            context,
            status,
            tuple(reasons),
            run,
            tuple(supported),
            self.uncertainty_refs,
            self.discrepancy_refs,
            self.sensitivity_refs,
            self.limitations,
        )
