"""Multiplicative physical units; conversions are always explicit.

Unknown atomic names are treated as distinct dimensions, never as dimensionless.
Affine temperature units are deliberately excluded from arithmetic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    dimensions: tuple[tuple[str, int], ...] = ()
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("unit scale must be finite and positive")
        powers: dict[str, int] = {}
        for name, power in self.dimensions:
            if not name or not isinstance(power, int):
                raise ValueError("unit dimensions need named integer powers")
            powers[name] = powers.get(name, 0) + power
        object.__setattr__(
            self, "dimensions", tuple(sorted((n, p) for n, p in powers.items() if p))
        )

    def __mul__(self, other: Unit) -> Unit:
        powers = dict(self.dimensions)
        for name, power in other.dimensions:
            powers[name] = powers.get(name, 0) + power
        return Unit(tuple(sorted((n, p) for n, p in powers.items() if p)), self.scale * other.scale)

    def __pow__(self, power: int) -> Unit:
        return Unit(
            tuple((n, p * power) for n, p in self.dimensions if p * power), self.scale**power
        )

    def __truediv__(self, other: Unit) -> Unit:
        return self * other**-1

    def factor_to(self, other: Unit) -> float:
        if self.dimensions != other.dimensions:
            raise ValueError(f"incompatible units: {self} and {other}")
        return self.scale / other.scale

    def __str__(self) -> str:
        # A numeric scale is retained, so derived mV/s does not silently become V/s.
        terms = [n if p == 1 else f"{n}^{p}" for n, p in self.dimensions]
        if self.scale != 1:
            terms.insert(0, repr(self.scale))
        return "*".join(terms) or "1"


_BASE = {
    "1": Unit(),
    "%": Unit((), 0.01),
    "s": Unit((("s", 1),)),
    "ms": Unit((("s", 1),), 0.001),
    "min": Unit((("s", 1),), 60),
    "Hz": Unit((("s", -1),)),
    "bpm": Unit((("s", -1),), 1 / 60),
    "m": Unit((("m", 1),)),
    "cm": Unit((("m", 1),), 0.01),
    "kg": Unit((("kg", 1),)),
    "g": Unit((("kg", 1),), 0.001),
    "A": Unit((("A", 1),)),
    "mA": Unit((("A", 1),), 0.001),
    "uA": Unit((("A", 1),), 1e-6),
    "K": Unit((("K", 1),)),
}
_BASE["V"] = _BASE["kg"] * _BASE["m"] ** 2 / _BASE["s"] ** 3 / _BASE["A"]
for _name, _scale in (("mV", 0.001), ("uV", 1e-6)):
    _BASE[_name] = Unit(_BASE["V"].dimensions, _scale)
_BASE["Pa"] = _BASE["kg"] / _BASE["m"] / _BASE["s"] ** 2
_BASE["mmHg"] = Unit(_BASE["Pa"].dimensions, 133.322387415)
_BASE["L"] = Unit((("m", 3),), 0.001)
_BASE["mL"] = Unit((("m", 3),), 1e-6)
_BASE["C"] = _BASE["A"] * _BASE["s"]
_BASE["J"] = _BASE["kg"] * _BASE["m"] ** 2 / _BASE["s"] ** 2


def unit(text: str | Unit) -> Unit:
    if isinstance(text, Unit):
        return text
    text = text.replace("µ", "u").replace(" ", "")
    if not text:
        raise ValueError("a unit must be declared (use '1' for dimensionless)")
    result = Unit()
    sign = 1
    for token in re.split(r"([*/])", text):
        if token in ("*", "/"):
            sign = -1 if token == "/" else 1
            continue
        match = re.fullmatch(r"([^\^]+)(?:\^(-?\d+))?", token)
        if not match:
            raise ValueError(f"unsupported unit expression {text!r}")
        name, power = match.groups()
        if name in {"degC", "°C", "degF"}:
            raise ValueError("affine temperature conversion must be an explicit transform")
        try:
            scale = float(name)
        except ValueError:
            atom = _BASE.get(name, Unit(((name, 1),)))
        else:
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("unit scale must be finite and positive")
            atom = Unit((), scale)
        result = result * atom ** (sign * int(power or 1))
    return result
