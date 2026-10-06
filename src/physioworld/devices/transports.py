"""Optional real transport boundaries. Importing this module requires no extras."""

from __future__ import annotations

import importlib.util
import socket
from typing import Any, Protocol


def optional_capabilities() -> dict[str, dict[str, Any]]:
    dependencies = {
        "serial": ("pyserial", "serial"),
        "bluetooth": ("bleak", "bleak"),
        "lsl": ("pylsl", "pylsl"),
        "native_fmi_cosimulation": ("fmpy", "fmpy"),
    }
    return {
        cap: {
            "package": distribution,
            "import_name": module,
            "installed": importlib.util.find_spec(module) is not None,
        }
        for cap, (distribution, module) in dependencies.items()
    }


class LineTransport(Protocol):
    def exchange(self, payload: bytes) -> bytes: ...
    def close(self) -> None: ...


class NetworkTransport:
    """Bounded newline-delimited RPC over TCP; timeout is never disabled."""

    def __init__(
        self, host: str, port: int, timeout_s: float = 2.0, max_message_bytes: int = 1 << 20
    ) -> None:
        if timeout_s <= 0 or max_message_bytes < 1:
            raise ValueError("positive timeout/message limit required")
        self.socket = socket.create_connection((host, port), timeout_s)
        self.limit = max_message_bytes
        self.pending = b""

    def exchange(self, payload: bytes) -> bytes:
        if len(payload) > self.limit or b"\n" in payload:
            raise ValueError("RPC payload must be one bounded line")
        self.socket.sendall(payload + b"\n")
        while b"\n" not in self.pending:
            chunk = self.socket.recv(min(4096, self.limit + 1 - len(self.pending)))
            if not chunk:
                raise IOError("network device disconnected")
            self.pending += chunk
            if len(self.pending) > self.limit:
                raise IOError("network device message exceeds limit")
        line, self.pending = self.pending.split(b"\n", 1)
        return line

    def close(self) -> None:
        self.socket.close()


class SerialTransport:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout_s: float = 2.0,
        max_message_bytes: int = 1 << 20,
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "serial transport requires optional package pyserial (import serial)"
            ) from exc
        if timeout_s <= 0 or max_message_bytes < 1:
            raise ValueError("positive timeout/message limit required")
        self.port = serial.serial_for_url(
            port, baudrate=baudrate, timeout=timeout_s, write_timeout=timeout_s
        )
        self.limit = max_message_bytes

    def exchange(self, payload: bytes) -> bytes:
        if b"\n" in payload or len(payload) > self.limit:
            raise ValueError("RPC payload must be one bounded line")
        self.port.write(payload + b"\n")
        self.port.flush()
        result = self.port.read_until(b"\n", self.limit + 1)
        if len(result) > self.limit or not result.endswith(b"\n"):
            raise IOError("serial response incomplete, timed out, or exceeded message limit")
        return result[:-1]

    def close(self) -> None:
        self.port.close()


class BluetoothBoundary:
    """Async BLE GATT reads/writes/discovery using bleak; not a vendor codec."""

    def __init__(self, address: str, timeout_s: float = 5.0) -> None:
        try:
            from bleak import BleakClient
        except ImportError as exc:
            raise ImportError("Bluetooth GATT requires optional package bleak") from exc
        self.client = BleakClient(address, timeout=timeout_s)

    async def connect(self) -> None:
        await self.client.connect()

    async def read(self, characteristic: str) -> bytes:
        return bytes(await self.client.read_gatt_char(characteristic))

    async def write(self, characteristic: str, payload: bytes) -> None:
        await self.client.write_gatt_char(characteristic, payload, response=True)

    async def close(self) -> None:
        await self.client.disconnect()

    @staticmethod
    async def discover(timeout_s: float = 5.0) -> list[Any]:
        try:
            from bleak import BleakScanner
        except ImportError as exc:
            raise ImportError("Bluetooth discovery requires optional package bleak") from exc
        return list(await BleakScanner.discover(timeout=timeout_s))


class LSLBoundary:
    """LSL timestamps remain in the LSL clock until caller applies time_correction."""

    def __init__(self, stream_info: Any) -> None:
        try:
            from pylsl import StreamInlet
        except ImportError as exc:
            raise ImportError(
                "LSL acquisition requires optional package pylsl and its liblsl runtime"
            ) from exc
        self.inlet = StreamInlet(stream_info)

    @staticmethod
    def discover(property: str = "type", value: str = "EEG", timeout_s: float = 2.0) -> list[Any]:
        try:
            from pylsl import resolve_byprop
        except ImportError as exc:
            raise ImportError("LSL discovery requires optional package pylsl") from exc
        return list(resolve_byprop(property, value, timeout=timeout_s))

    def read(self, max_samples: int = 256, timeout_s: float = 1.0) -> tuple[Any, Any]:
        return self.inlet.pull_chunk(timeout=timeout_s, max_samples=max_samples)

    def time_correction(self, timeout_s: float = 1.0) -> float:
        return float(self.inlet.time_correction(timeout=timeout_s))

    def close(self) -> None:
        self.inlet.close_stream()
