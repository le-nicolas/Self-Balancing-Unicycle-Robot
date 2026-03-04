import json
import socket
from typing import TYPE_CHECKING, Any

try:
    import serial  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    serial = None

if TYPE_CHECKING:
    from runtime_config import RuntimeConfig


class _UdpTelemetrySink:
    def __init__(self, host: str, port: int):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def send(self, frame: dict[str, Any]) -> None:
        payload = (json.dumps(frame, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def close(self) -> None:
        self._sock.close()


class _SerialTelemetrySink:
    def __init__(self, port: str, baud: int):
        if serial is None:
            raise RuntimeError("pyserial is required for serial telemetry. Install with: pip install pyserial")
        self._serial = serial.Serial(port=port, baudrate=baud, timeout=0.0, write_timeout=0.0)

    def send(self, frame: dict[str, Any]) -> None:
        payload = (json.dumps(frame, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        self._serial.write(payload)

    def close(self) -> None:
        self._serial.close()


class TelemetryPublisher:
    def __init__(self, sink: _UdpTelemetrySink | _SerialTelemetrySink | None, *, rate_hz: float, endpoint: str):
        self._sink = sink
        self._min_period_s = 0.0 if rate_hz <= 0.0 else (1.0 / rate_hz)
        self._last_sim_time_s: float | None = None
        self.endpoint = endpoint
        self.sent_frames = 0
        self.dropped_frames = 0

    @property
    def active(self) -> bool:
        return self._sink is not None

    def publish(self, *, sim_time_s: float, frame: dict[str, Any]) -> None:
        if self._sink is None:
            return
        if self._min_period_s > 0.0 and self._last_sim_time_s is not None:
            # MuJoCo time can jump backwards when we reset after crashes.
            if sim_time_s < self._last_sim_time_s:
                self._last_sim_time_s = None
            elif (sim_time_s - self._last_sim_time_s) < self._min_period_s:
                return
        try:
            self._sink.send(frame)
            self.sent_frames += 1
        except Exception:
            self.dropped_frames += 1
        self._last_sim_time_s = sim_time_s

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()


def create_telemetry_publisher(cfg: "RuntimeConfig") -> TelemetryPublisher:
    if not cfg.telemetry_enabled:
        return TelemetryPublisher(None, rate_hz=cfg.telemetry_rate_hz, endpoint="disabled")

    try:
        if cfg.telemetry_transport == "serial":
            if not cfg.telemetry_serial_port:
                print("Warning: telemetry serial transport selected but no serial port was provided; telemetry disabled.")
                return TelemetryPublisher(None, rate_hz=cfg.telemetry_rate_hz, endpoint="disabled")
            sink = _SerialTelemetrySink(cfg.telemetry_serial_port, cfg.telemetry_serial_baud)
            endpoint = f"serial://{cfg.telemetry_serial_port}@{cfg.telemetry_serial_baud}"
        else:
            sink = _UdpTelemetrySink(cfg.telemetry_udp_host, cfg.telemetry_udp_port)
            endpoint = f"udp://{cfg.telemetry_udp_host}:{cfg.telemetry_udp_port}"
        return TelemetryPublisher(sink, rate_hz=cfg.telemetry_rate_hz, endpoint=endpoint)
    except Exception as exc:
        print(f"Warning: telemetry initialization failed ({exc}); telemetry disabled.")
        return TelemetryPublisher(None, rate_hz=cfg.telemetry_rate_hz, endpoint="disabled")
