# Declarative physiology protocols

`physioworld.experiments.protocol` is a small, typed execution kernel for
repeatable laboratory and simulation workflows.  A protocol is data made from
typed nodes, an explicit seconds clock, a seed, metadata, bounded control flow,
and an optional emergency sequence.  It is suitable for configuration files,
Python builders, notebooks, and published manifests.

The runtime does not interpret protocol text as Python.  Node handler names are
lookups in an injected `HandlerRegistry`; a handler is ordinary Python code
owned by the integration layer and receives only the typed node plus an
`ExecutionContext`.  This keeps device, recording, physiology, computation,
and export integration explicit while leaving ordering and traceability to the
protocol runtime.

## A minimal protocol

```python
from physioworld.experiments.protocol import (
    Event,
    ExecutionLimits,
    HandlerRegistry,
    ProtocolBuilder,
    ProtocolRuntime,
)


def event_handler(node, context):
    # Call a native device/recording integration here if appropriate.
    return None


protocol = (
    ProtocolBuilder(
        "baseline-mark",
        horizon_s=5.0,
        seed=17,
        limits=ExecutionLimits(max_commands=20),
        metadata={"purpose": "clock and event example"},
    )
    .add(Event("baseline", "baseline", handler="event"))
    .build()
)
result = ProtocolRuntime(HandlerRegistry({"event": event_handler})).run(protocol)
assert result.status == "completed"
```

`ProtocolBuilder` is convenience syntax.  `Setup`, `Calibration`, `Baseline`,
`Stimulus`, `Acquire`, `Event`, `Wait`, `Computation`, `Decision`,
`Intervention`, `Resource`, `Measurement`, `Stop`, and `Export` are distinct
typed command nodes.  `Loop`, `If`, `Schedule`, and `ParameterSweep` are
typed control-flow nodes.  They are not aliases for a generic operation.

## Time, units, and state

* Protocol time is a monotone simulation coordinate in seconds.  `horizon_s`
  is mandatory, durations are positive, and `at_s` cannot move the clock into
  the past or beyond the horizon.
* Numeric command parameters require a declared unit in the node's `units`
  mapping.  `Quantity(value, unit)` carries a unit with a returned or compared
  value.  Conversion uses the existing `instrumentation.units` implementation;
  incompatible dimensions fail explicitly.
* `Reference("measurement.name")`, `Reference("resource.name")`, and named
  variable references resolve only values already present in the execution
  context.  A future measurement is an error.
* A handler result for `Measurement` or `Computation` must be a scalar,
  `{ "value": ..., "unit": ... }`, or `Quantity`.  The runtime converts it to
  the node's declared output unit and stores it in the result manifest.

The protocol clock is independent from wall time.  A handler may advance an
existing `PhysiologyEngine`, `AcquisitionRunner`, or other native component to
the same declared boundary, as shown in `examples/protocol_lab.py`.

## Native handler boundary

Handlers are injected rather than named imports in protocol data:

```python
from physioworld.experiments.protocol import HandlerContract, HandlerRegistry


def acquire_handler(node, context):
    # Use an existing AcquisitionRunner or Device contract here.
    # Return a declared result only if the node defines one.
    return None


registry = HandlerRegistry(
    {"acquire": acquire_handler},
    contracts={
        "acquire": HandlerContract(
            "acquire",
            required_parameters=("device_id",),
            allowed_parameters=("device_id",),
        )
    },
)
```

The existing native APIs remain the hardware boundary:

* `physioworld.devices.Device` provides discovery, configuration, calibration,
  read, stop, emergency stop, and stimulation contracts.
* `AcquisitionRunner` retains explicit device clock maps and records acquired
  blocks in `instrumentation.recording.Recording`.
* `PhysiologyEngine` provides a reduced-order simulation clock, declared
  `DataRequest` observations, scheduled `PhysiologicalAction`s, and simulation
  provenance.

The protocol layer does not claim that a handler call produced physical
measurement.  Handler provenance and the result's evidence class should retain
whether a value is hardware-acquired, replayed, or simulated.

## Control flow and reproducibility

`Schedule` supports `sequential`, seeded `randomized`, and seeded
`counterbalanced` case orders.  Randomized ordering is derived from the
protocol seed and schedule node ID, so identical declarations produce identical
orders.  `ParameterSweep` enumerates the Cartesian product in sorted parameter
name order.  `If` and `Decision` compare literals and prior values only; there
is no expression evaluator.

Every protocol has finite `ExecutionLimits`:

* `max_commands` bounds ordinary command dispatches;
* `max_loop_iterations` bounds each loop;
* `max_sweep_cases` bounds a sweep's Cartesian product;
* `max_branch_depth` bounds nested control flow; and
* `max_emergency_commands` bounds emergency dispatches.

When a handler calls `context.abort(reason)`, the run gets status `aborted` and
the declared `emergency` nodes execute.  A handler exception or protocol limit
produces `ProtocolExecutionError` with a failed `ExecutionResult` attached.
The result contains chronology, actions, events, measurement units, and
provenance entries including emergency status.  An emergency sequence is a
recorded fallback, not an implicit safety certification; hardware handlers
must still use their native device safe-state and stimulation bounds.

## Machine-readable form

`schemas/protocol.schema.json` is the source-checkout copy and
`physioworld/schemas/protocol.schema.json` is the identical installed resource.
The schema is Draft 2020-12, requires protocol schema version `1.0.0`, uses a
closed root and closed node objects, and rejects unknown operations and unknown
node fields.  Parameter object keys remain open because their contract is
declared by the injected handler; `HandlerContract` can close those keys for a
specific integration.

Round-trip a declaration with:

```python
payload = protocol.to_dict()
restored = Protocol.from_dict(payload)
assert restored.to_dict() == payload
```

`Protocol.from_dict` performs strict native validation in addition to the JSON
schema's structural validation.  It rejects unsupported schema versions,
malformed units, duplicate sibling node IDs, non-finite values, missing time
units, executable-looking handler names, and unsupported operations.

## Reduced-order lab example and evidence scope

Run:

```text
PYTHONPATH=src python examples/protocol_lab.py
```

The example constructs `PhysiologyEngine` with an explicit `DataRequest`,
creates a two-point `devices.Calibration` artifact from declared demo points,
advances an exercise intervention and recovery window, and computes a derived
shock-index quantity from two engine observations.  Its output is labeled
`SIMULATION`.  It does not stand in for a sensor calibration study, hardware
qualification, a clinical result, or external measurement evidence.
