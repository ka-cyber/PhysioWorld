"""Explicit acyclic coupling with declared unit conversion and lagged feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from physioworld.instrumentation.units import unit

from .models import FirstOrderModel, StepResult


@dataclass(frozen=True)
class Connection:
    source_model: str
    source_variable: str
    target_model: str
    target_variable: str
    conversion_factor: float = 1.0
    lagged: bool = False


class CoupledSystem:
    """Start-point ZOH macro steps; lagged edges add one communication-step delay.

    Each input is ZOH during a model step. No hidden algebraic iteration. Models
    integrate at all their internal events/clocks, which need not share a period.
    """

    def __init__(
        self, models: Mapping[str, FirstOrderModel], connections: tuple[Connection, ...]
    ) -> None:
        if not models or len({id(m) for m in models.values()}) != len(models):
            raise ValueError("nonempty distinct model instances required")
        self.models, self.connections = dict(models), connections
        targets = set()
        for edge in connections:
            source = next(
                v for v in models[edge.source_model].variables if v.name == edge.source_variable
            )
            target = next(
                v for v in models[edge.target_model].variables if v.name == edge.target_variable
            )
            expected_factor = unit(source.unit).factor_to(unit(target.unit))
            if (
                source.causality != "output"
                or target.causality != "input"
                or edge.conversion_factor != expected_factor
            ):
                raise ValueError(
                    "connection requires output->input and explicit correct unit conversion factor"
                )
            key = (edge.target_model, edge.target_variable)
            if key in targets:
                raise ValueError("multiple drivers of one model input")
            targets.add(key)
        order, active, done = [], set(), set()

        def visit(name: str) -> None:
            if name in active:
                raise ValueError("coupling cycle requires an explicitly lagged edge")
            if name in done:
                return
            active.add(name)
            for edge in connections:
                if edge.target_model == name and not edge.lagged:
                    visit(edge.source_model)
            active.remove(name)
            done.add(name)
            order.append(name)

        for name in models:
            visit(name)
        self.order = tuple(order)
        self._previous_outputs: dict[tuple[str, str], float] = {}

    def initialize(self, start_time: float = 0.0) -> None:
        for model in self.models.values():
            model.initialize(start_time)
        self._previous_outputs = {
            (e.source_model, e.source_variable): self.models[e.source_model].get(e.source_variable)
            for e in self.connections
        }

    def step(self, step_size: float) -> dict[str, StepResult]:
        times = {m.time for m in self.models.values()}
        if len(times) != 1:
            raise ValueError(
                "models must share a communication point, not necessarily internal clocks"
            )
        snapshot = {
            (e.source_model, e.source_variable): self.models[e.source_model].get(e.source_variable)
            for e in self.connections
        }
        results = {}
        for name in self.order:
            model = self.models[name]
            for edge in self.connections:
                if edge.target_model == name:
                    key = (edge.source_model, edge.source_variable)
                    value = self._previous_outputs[key] if edge.lagged else snapshot[key]
                    target = next(v for v in model.variables if v.name == edge.target_variable)
                    model.set(target.name, value * edge.conversion_factor, target.unit)
            results[name] = model.do_step(model.time, step_size)
        self._previous_outputs = snapshot
        return results
