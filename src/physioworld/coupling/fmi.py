"""FMI 3.0.2 metadata subset and tool-required Python co-simulation boundary.

The archive has FMI 3 modelDescription/units/Float64/Clock metadata and a
declarative Python model resource. It advertises needsExecutionTool=true and
requires this runtime, not a generic FMI C-ABI importer. Archive code is never
executed. Generic native Co-Simulation is a separate optional FMPy boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from physioworld.instrumentation.units import Unit, unit

from .models import FirstOrderModel, ScheduledClock, StepResult, Variable


class UnsafeArchiveError(ValueError):
    pass


class UnsupportedFMUError(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 1024
    max_total_bytes: int = 64 << 20
    max_member_bytes: int = 32 << 20
    max_compression_ratio: float = 1000.0

    def __post_init__(self) -> None:
        if (
            min(self.max_members, self.max_total_bytes, self.max_member_bytes) < 1
            or not math.isfinite(self.max_compression_ratio)
            or self.max_compression_ratio <= 0
        ):
            raise ValueError("archive limits must be finite and positive")


@contextmanager
def _open_archive(archive: str | Path):
    try:
        with zipfile.ZipFile(archive) as package:
            yield package
    except (
        OSError,
        NotImplementedError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise UnsafeArchiveError("invalid FMU ZIP archive") from exc


def safe_extract_fmu(
    archive: str | Path, destination: str | Path, limits: ArchiveLimits = ArchiveLimits()
) -> Path:
    """Validate the entire archive before extracting to a new/empty real directory."""
    destination = Path(destination)
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise UnsafeArchiveError("FMU extraction requires an empty non-symlink directory")
    if any(parent.is_symlink() for parent in destination.parents):
        raise UnsafeArchiveError("symlink ancestors are not allowed for extraction")
    with _open_archive(archive) as package:
        members = package.infolist()
        if (
            len(members) > limits.max_members
            or sum(i.file_size for i in members) > limits.max_total_bytes
        ):
            raise UnsafeArchiveError("FMU archive exceeds member/expanded-size limits")
        seen: set[str] = set()
        files: set[str] = set()
        directories: set[str] = set()
        for info in members:
            # Check both spellings: platform-specific ZIP implementations may
            # expose a normalized filename while retaining the original name.
            original_name = info.orig_filename
            exposed_name = info.filename
            name = original_name
            path = PurePosixPath(name)
            parts = name.rstrip("/").split("/")
            mode = info.external_attr >> 16
            if (
                not name
                or "\\" in original_name
                or "\\" in exposed_name
                or ":" in name
                or "\x00" in name
                or path.is_absolute()
                or any(p in {"", ".", ".."} for p in parts)
                or any(
                    p.endswith((".", " "))
                    or re.fullmatch(
                        r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", p.split(".")[0], re.IGNORECASE
                    )
                    for p in parts
                )
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})
                or (stat.S_ISDIR(mode) and not info.is_dir())
                or (info.is_dir() and info.file_size != 0)
                or info.flag_bits & 1
                or info.file_size > limits.max_member_bytes
                or info.file_size / max(1, info.compress_size) > limits.max_compression_ratio
            ):
                raise UnsafeArchiveError(f"unsafe FMU member {name!r}")
            normalized = "/".join(parts).casefold()
            if normalized in seen:
                raise UnsafeArchiveError("duplicate/case-colliding FMU archive paths")
            seen.add(normalized)
            (directories if info.is_dir() else files).add(normalized)
        for filename in files | directories:
            ancestors = filename.split("/")[:-1]
            if any("/".join(ancestors[: i + 1]) in files for i in range(len(ancestors))):
                raise UnsafeArchiveError("file/directory collision in FMU archive")
        if package.testzip() is not None:
            raise UnsafeArchiveError("FMU archive failed CRC validation")
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        for info in members:
            target = root.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(info) as source, target.open("xb") as output:
                count = 0
                while True:
                    chunk = source.read(65536)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > info.file_size or count > limits.max_member_bytes:
                        raise UnsafeArchiveError("expanded FMU member exceeds declared size")
                    output.write(chunk)
    return destination


def _xml_root(payload: bytes) -> ET.Element:
    # Restricted UTF-8 subset avoids DTD/entity expansion and UTF-16 bypasses.
    text = payload.decode("utf-8-sig")
    if "\x00" in text or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.IGNORECASE):
        raise UnsafeArchiveError("DTD/entity declarations are forbidden in modelDescription")
    # DTD/entity declarations are rejected above; the parser only receives a
    # UTF-8 FMI model description with no external entity mechanism enabled.
    root = ET.fromstring(text)  # nosec B314
    if root.tag != "fmiModelDescription" or root.get("fmiVersion") not in {"3.0", "3.0.1", "3.0.2"}:
        raise UnsupportedFMUError("only FMI 3.0 through 3.0.2 model descriptions are supported")
    if not root.get("modelName") or not root.get("instantiationToken"):
        raise ValueError("model description is missing name/token")
    return root


def _xml_boolean(value: str) -> bool:
    if value.strip() not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid XML boolean {value!r}")
    return value.strip() in {"true", "1"}


_COMMON_FLAGS = (
    "needsExecutionTool",
    "canBeInstantiatedOnlyOncePerProcess",
    "canGetAndSetFMUState",
    "canSerializeFMUState",
    "providesDirectionalDerivatives",
    "providesAdjointDerivatives",
    "providesPerElementDependencies",
)
_INTERFACE_FLAGS = {
    "ModelExchange": ("needsCompletedIntegratorStep", "providesEvaluateDiscreteStates"),
    "CoSimulation": (
        "canHandleVariableCommunicationStepSize",
        "providesIntermediateUpdate",
        "mightReturnEarlyFromDoStep",
        "canReturnEarlyAfterIntermediateUpdate",
        "hasEventMode",
        "providesEvaluateDiscreteStates",
    ),
    "ScheduledExecution": (),
}


@dataclass(frozen=True)
class InterfaceCapabilities:
    """Advertised metadata, not evidence that this runtime implements the interface.

    Known boolean flags include FMI defaults (false); attributes preserve the
    original declaration, including numeric and vendor-specific attributes.
    """

    interface: str
    model_identifier: str
    boolean_flags: tuple[tuple[str, bool], ...]
    attributes: tuple[tuple[str, str], ...]

    @property
    def needs_execution_tool(self) -> bool:
        return dict(self.boolean_flags)["needsExecutionTool"]


def _interface_capabilities(root: ET.Element) -> tuple[InterfaceCapabilities, ...]:
    result = []
    for interface, flags in _INTERFACE_FLAGS.items():
        declarations = root.findall(interface)
        if len(declarations) > 1:
            raise ValueError(f"duplicate {interface} declaration")
        if not declarations:
            continue
        element = declarations[0]
        identifier = element.get("modelIdentifier", "")
        if not identifier:
            raise ValueError("interface requires a nonempty modelIdentifier")
        result.append(
            InterfaceCapabilities(
                interface,
                identifier,
                tuple(
                    (flag, _xml_boolean(element.get(flag, "false")))
                    for flag in _COMMON_FLAGS + flags
                ),
                tuple(sorted(element.attrib.items())),
            )
        )
    if not result:
        raise ValueError("model description must declare an FMI interface")
    return tuple(result)


def _declared_units(root: ET.Element) -> set[str]:
    """Avoid interpreting XML-defined units using conflicting symbolic units."""
    names: set[str] = set()
    dimensions = {"kg", "m", "s", "A", "K", "mol", "cd", "rad"}
    for definition in root.findall("UnitDefinitions/Unit"):
        name = definition.get("name", "")
        if not name or name in names:
            raise ValueError("unit definitions require unique nonempty names")
        names.add(name)
        bases = definition.findall("BaseUnit")
        if len(bases) > 1:
            raise ValueError("duplicate BaseUnit definition")
        if bases:
            base = bases[0]
            if set(base.attrib) - dimensions - {"factor", "offset"}:
                raise ValueError("unknown BaseUnit attribute")
            offset = float(base.get("offset", "0"))
            if not math.isfinite(offset):
                raise ValueError("unit offset must be finite")
            if offset != 0:
                raise UnsupportedFMUError("affine BaseUnit offsets are outside the unit subset")
            declared = Unit(
                tuple((d, int(base.get(d, "0"))) for d in sorted(dimensions)),
                float(base.get("factor", "1")),
            )
            symbolic = unit(name)
            if declared.dimensions != symbolic.dimensions or not math.isclose(
                declared.scale, symbolic.scale, rel_tol=1e-12, abs_tol=0
            ):
                raise UnsupportedFMUError("XML BaseUnit conflicts with supported symbolic unit")
        if definition.findall("DisplayUnit"):
            raise UnsupportedFMUError("DisplayUnit definitions are outside the unit subset")
    return names


@dataclass(frozen=True)
class ModelDescription:
    fmi_version: str
    model_name: str
    instantiation_token: str
    interfaces: tuple[str, ...]
    variables: tuple[Variable, ...]
    clocks: tuple[ScheduledClock, ...]
    needs_execution_tool: bool
    interface_capabilities: tuple[InterfaceCapabilities, ...] = ()


def parse_model_description(payload: bytes) -> ModelDescription:
    """Parse the owned metadata subset, not a complete FMI schema/semantic validator.

    Scalar Float64 uses project symbolic units and finite starts (omitted starts
    map to zero). Clocks are constant decimal input clocks, not native SE partitions.
    """
    root = _xml_root(payload)
    capabilities = _interface_capabilities(root)
    variables, clocks, names, references = [], [], set(), set()
    declared_units = _declared_units(root)
    if root.find("ModelVariables") is None or root.find("ModelStructure") is None:
        raise ValueError("model description requires ModelVariables and ModelStructure")
    for element in root.findall("ModelVariables/*"):
        name, reference_text = element.get("name"), element.get("valueReference")
        if name is None or reference_text is None:
            raise ValueError("variable requires name and valueReference")
        try:
            reference = int(reference_text)
        except ValueError as exc:
            raise ValueError("valueReference must be an integer") from exc
        if not name or not 0 <= reference < 2**32:
            raise ValueError("invalid variable name/valueReference")
        if name in names or reference in references:
            raise ValueError("duplicate variable name/valueReference")
        names.add(name)
        references.add(reference)
        if element.get("declaredType") is not None or element.find("Alias") is not None:
            raise UnsupportedFMUError("declaredType inheritance and aliases are outside the subset")
        if element.get("clocks") is not None:
            raise UnsupportedFMUError(
                "clocked variables/native SE partitions are outside the subset"
            )
        if element.get("causality", "local") not in {
            "input",
            "output",
            "parameter",
            "local",
            "independent",
        }:
            raise UnsupportedFMUError(
                "calculated/structural parameter variables are outside the subset"
            )
        if element.tag == "Clock":
            if (
                element.get("intervalVariability") != "constant"
                or element.get("causality", "local") != "input"
                or element.get("variability") != "discrete"
                or _xml_boolean(element.get("supportsFraction", "false"))
                or _xml_boolean(element.get("canBeDeactivated", "false"))
                or any(
                    a in element.attrib
                    for a in ("resolution", "intervalCounter", "shiftCounter", "priority")
                )
            ):
                raise UnsupportedFMUError(
                    "Python clock subset requires constant decimal input clocks without priority/deactivation"
                )
            if "intervalDecimal" not in element.attrib:
                raise ValueError("constant decimal Clock requires intervalDecimal")
            clocks.append(
                ScheduledClock(
                    name,
                    float(element.attrib["intervalDecimal"]),
                    float(element.get("shiftDecimal", "0")),
                )
            )
        elif element.tag == "Float64" and not element.findall("Dimension"):
            declared = element.get("unit", "1")
            if "unit" in element.attrib and declared not in declared_units:
                raise ValueError("variable unit not declared in UnitDefinitions")
            if element.get("displayUnit") is not None:
                raise UnsupportedFMUError("displayUnit conversion is outside the unit subset")
            if element.get("variability", "continuous") not in {
                "constant",
                "fixed",
                "tunable",
                "discrete",
                "continuous",
            }:
                raise ValueError("invalid Float64 variability")
            variables.append(
                Variable(
                    name,
                    reference,
                    declared,
                    element.get("causality", "local"),
                    float(element.get("start", "0")),
                    element.get("variability", "continuous"),
                )
            )
        else:
            raise UnsupportedFMUError(
                "description API supports scalar Float64 and constant Clock only"
            )
    output_references = [
        int(e.attrib["valueReference"]) for e in root.findall("ModelStructure/Output")
    ]
    outputs = set(output_references)
    if len(outputs) != len(output_references) or outputs != {
        v.value_reference for v in variables if v.causality == "output"
    }:
        raise ValueError("ModelStructure outputs do not match variables")
    return ModelDescription(
        root.attrib["fmiVersion"],
        root.attrib["modelName"],
        root.attrib["instantiationToken"],
        tuple(c.interface for c in capabilities),
        tuple(variables),
        tuple(clocks),
        any(c.interface == "CoSimulation" and c.needs_execution_tool for c in capabilities),
        capabilities,
    )


def _canonical_spec(model: FirstOrderModel) -> bytes:
    return json.dumps(
        model.specification(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def model_description_xml(model: FirstOrderModel) -> bytes:
    token = hashlib.sha256(_canonical_spec(model)).hexdigest()
    root = ET.Element(
        "fmiModelDescription",
        fmiVersion="3.0.2",
        modelName=model.name,
        instantiationToken=token,
        generationTool="PhysioWorld Python FMU runtime",
        description="Tool-required declarative Python co-simulation; no native FMI binary",
    )
    ET.SubElement(
        root,
        "CoSimulation",
        modelIdentifier="physioworld_python",
        needsExecutionTool="true",
        canHandleVariableCommunicationStepSize="true",
        hasEventMode="true",
    )
    definitions = ET.SubElement(root, "UnitDefinitions")
    for name in sorted({v.unit for v in model.variables}):
        parsed = unit(name)
        if not set(dict(parsed.dimensions)) <= {"kg", "m", "s", "A", "K", "mol", "cd", "rad"}:
            raise ValueError("FMU export requires units expressible as FMI SI BaseUnit dimensions")
        definition = ET.SubElement(definitions, "Unit", name=name)
        base_unit_attributes: dict[str, str] = {n: str(p) for n, p in parsed.dimensions}
        base_unit_attributes["factor"] = repr(parsed.scale)
        ET.SubElement(definition, "BaseUnit", attrib=base_unit_attributes)
    variables = ET.SubElement(root, "ModelVariables")
    for variable in model.variables:
        attributes = {
            "name": variable.name,
            "valueReference": str(variable.value_reference),
            "causality": variable.causality,
            "variability": variable.variability,
            "unit": variable.unit,
        }
        if variable.causality != "independent":
            attributes["start"] = repr(variable.start)
            if variable.causality != "input":
                attributes["initial"] = "exact"
        ET.SubElement(variables, "Float64", attrib=attributes)
    for index, clock in enumerate(model.clocks, start=len(model.variables)):
        ET.SubElement(
            variables,
            "Clock",
            name=clock.name,
            valueReference=str(index),
            causality="input",
            variability="discrete",
            intervalVariability="constant",
            intervalDecimal=repr(clock.period_s),
            shiftDecimal=repr(clock.phase_s),
        )
    structure = ET.SubElement(root, "ModelStructure")
    for variable in model.variables:
        if variable.causality == "output":
            ET.SubElement(
                structure, "Output", valueReference=str(variable.value_reference), dependencies=""
            )
    annotations = ET.SubElement(root, "Annotations")
    annotation = ET.SubElement(annotations, "Annotation", type="org.physioworld.python-fmu/1")
    ET.SubElement(
        annotation,
        "PythonModel",
        resource="resources/physioworld.json",
        sha256=token,
        execution="trusted-built-in-model-only",
        eventOrder="actions-then-clocks",
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def export_fmu(model: FirstOrderModel, path: str | Path) -> Path:
    """Package the supported Python model; deterministic resources, no executable code."""
    if type(model) is not FirstOrderModel:
        raise UnsupportedFMUError("export currently supports FirstOrderModel only")
    payloads = {
        "modelDescription.xml": model_description_xml(model),
        "resources/physioworld.json": _canonical_spec(model),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        for name, payload in payloads.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            package.writestr(info, payload)
    return Path(path)


class PythonFMU:
    """Executable scalar co-simulation facade, including timed hybrid/clock events."""

    def __init__(self, model: FirstOrderModel, description: ModelDescription) -> None:
        self.model, self.description = model, description
        self.terminated = False

    def initialize(self, start_time: float = 0.0) -> None:
        if self.terminated:
            raise RuntimeError("FMU is terminated")
        self.model.initialize(start_time)

    def set_float64(self, name: str | int, value: float, declared_unit: str) -> None:
        if self.terminated:
            raise RuntimeError("FMU is terminated")
        self.model.set(self._name(name), value, declared_unit)

    def get_float64(self, name: str | int) -> float:
        if self.terminated:
            raise RuntimeError("FMU is terminated")
        return self.model.get(self._name(name))

    def _name(self, reference: str | int) -> str:
        for variable in self.description.variables:
            if reference == variable.name or reference == variable.value_reference:
                return variable.name
        raise KeyError(reference)

    def do_step(self, current_time: float, step_size: float) -> StepResult:
        if self.terminated:
            raise RuntimeError("FMU is terminated")
        return self.model.do_step(current_time, step_size)

    def terminate(self) -> None:
        self.terminated = True


def import_fmu(path: str | Path, *, limits: ArchiveLimits = ArchiveLimits()) -> PythonFMU:
    with tempfile.TemporaryDirectory(prefix="physioworld-fmu-") as directory:
        root = safe_extract_fmu(path, directory, limits)
        if not (root / "modelDescription.xml").is_file():
            raise ValueError("FMU requires modelDescription.xml at the archive root")
        payload = (root / "modelDescription.xml").read_bytes()
        description = parse_model_description(payload)
        resource = root / "resources" / "physioworld.json"
        if (
            not resource.is_file()
            or description.interfaces != ("CoSimulation",)
            or not description.needs_execution_tool
        ):
            raise UnsupportedFMUError(
                "not a PhysioWorld tool-required Python FMU; use optional NativeCoSimulation for binaries"
            )
        model = FirstOrderModel.from_specification(json.loads(resource.read_text(encoding="utf-8")))
        # Verify ALL metadata, units, starts, clocks and implementation token, not just model name.
        if ET.canonicalize(payload.decode("utf-8-sig")) != ET.canonicalize(
            model_description_xml(model).decode()
        ):
            raise ValueError("FMU modelDescription does not match declarative model resource")
        return PythonFMU(model, description)


class NativeCoSimulation:
    """Optional FMPy FMI 3.0.2 scalar Float64 CS runtime.

    Native FMUs run native code and must be trusted by the caller. Extraction is
    checked before the optional runtime is invoked. This CS-only boundary does
    not expose ME/SE, arrays, non-Float64 variables, tool-required FMUs, event
    mode, state serialization, intermediate updates, early return, or importer
    completeness.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            from fmpy import read_model_description
            from fmpy.fmi3 import FMU3Slave
        except ImportError as exc:
            raise ImportError(
                "native FMI 3 Co-Simulation requires optional package fmpy and a compatible FMU binary"
            ) from exc
        self._temporary = tempfile.TemporaryDirectory(prefix="physioworld-native-fmu-")
        self.fmu: Any = None
        self._instantiated = False
        try:
            root = safe_extract_fmu(path, self._temporary.name)
            _xml_root((root / "modelDescription.xml").read_bytes())
            self.description = read_model_description(str(root), validate=True)
            description = self.description
            if (
                description.fmiVersion not in {"3.0", "3.0.1", "3.0.2"}
                or description.coSimulation is None
                or description.coSimulation.needsExecutionTool
                or description.coSimulation.hasEventMode
                or any(
                    getattr(description.coSimulation, flag, False)
                    for flag in (
                        "providesIntermediateUpdate",
                        "mightReturnEarlyFromDoStep",
                        "canReturnEarlyAfterIntermediateUpdate",
                    )
                )
            ):
                raise UnsupportedFMUError(
                    "native subset is FMI 3 CS without tool-required, event mode, intermediate update or early return"
                )
            if any(
                v.type != "Float64" or getattr(v, "dimensions", ())
                for v in description.modelVariables
            ):
                raise UnsupportedFMUError("native subset supports scalar Float64 variables only")
            self.fmu = FMU3Slave(
                guid=description.guid,
                unzipDirectory=str(root),
                modelIdentifier=description.coSimulation.modelIdentifier,
                instanceName="physioworld",
            )
            self.fmu.instantiate()
            self._instantiated = True
            self.time, self.initialized = 0.0, False
        except Exception:
            self.close()
            raise

    def initialize(self, start_time: float = 0.0) -> None:
        self.fmu.enterInitializationMode(startTime=start_time)
        self.fmu.exitInitializationMode()
        self.time, self.initialized = start_time, True

    def _variable(self, name: str) -> Any:
        return next(v for v in self.description.modelVariables if v.name == name)

    def set_float64(self, name: str, value: float, declared_unit: str) -> None:
        variable = self._variable(name)
        if (
            variable.causality != "input"
            or not math.isfinite(value)
            or unit(variable.unit or "1") != unit(declared_unit)
        ):
            raise ValueError("input/unit/value mismatch")
        self.fmu.setFloat64([variable.valueReference], [value])

    def get_float64(self, name: str) -> float:
        return float(self.fmu.getFloat64([self._variable(name).valueReference])[0])

    def do_step(self, current_time: float, step_size: float) -> StepResult:
        if (
            not self.initialized
            or current_time != self.time
            or not math.isfinite(step_size)
            or step_size <= 0
        ):
            raise ValueError(
                "initialized matching communication point and positive finite step required"
            )
        event, terminate, early_return, last_time = self.fmu.doStep(
            currentCommunicationPoint=current_time, communicationStepSize=step_size
        )
        if event or terminate or early_return:
            raise UnsupportedFMUError(
                "native event/termination/early-return handling is outside this subset"
            )
        self.time = current_time + step_size
        return StepResult(self.time, False, ())

    def close(self) -> None:
        if self.fmu is not None:
            try:
                if getattr(self, "initialized", False):
                    self.fmu.terminate()
            finally:
                if self._instantiated:
                    self.fmu.freeInstance()
                self.fmu = None
        self._temporary.cleanup()

    def __enter__(self) -> NativeCoSimulation:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
