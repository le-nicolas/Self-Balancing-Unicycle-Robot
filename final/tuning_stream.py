from __future__ import annotations

import json
import math
import socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime_config import RuntimeConfig


class _UdpTuningSource:
    def __init__(self, bind_host: str, bind_port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            # REUSEADDR is a best-effort convenience only.
            pass
        self._sock.bind((bind_host, bind_port))
        self._sock.setblocking(False)

    def recv_packets(self, max_packets: int) -> list[bytes]:
        packets: list[bytes] = []
        for _ in range(max(int(max_packets), 1)):
            try:
                payload, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            packets.append(payload)
        return packets

    def close(self) -> None:
        self._sock.close()


def _decode_json_object(raw: bytes) -> dict[str, Any] | None:
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _extract_updates(payload: dict[str, Any]) -> dict[str, float]:
    values = payload.get("values", payload)
    if not isinstance(values, dict):
        return {}
    updates: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            updates[key] = numeric
    return updates


class TuningReceiver:
    def __init__(self, source: _UdpTuningSource | None, endpoint: str):
        self._source = source
        self.endpoint = endpoint
        self.received_packets = 0
        self.parse_errors = 0
        self.received_updates = 0

    @property
    def active(self) -> bool:
        return self._source is not None

    def drain_updates(self, max_packets: int = 64) -> dict[str, float]:
        if self._source is None:
            return {}
        updates: dict[str, float] = {}
        for packet in self._source.recv_packets(max_packets=max_packets):
            self.received_packets += 1
            packet_had_object = False
            packet_had_updates = False
            for raw in packet.splitlines() or [packet]:
                obj = _decode_json_object(raw)
                if obj is None:
                    continue
                packet_had_object = True
                parsed = _extract_updates(obj)
                if parsed:
                    updates.update(parsed)
                    packet_had_updates = True
            if not packet_had_object or not packet_had_updates:
                self.parse_errors += 1
        self.received_updates += len(updates)
        return updates

    def close(self) -> None:
        if self._source is not None:
            self._source.close()


def create_tuning_receiver(cfg: "RuntimeConfig") -> TuningReceiver:
    if not cfg.live_tuning_enabled:
        return TuningReceiver(None, endpoint="disabled")
    try:
        source = _UdpTuningSource(cfg.live_tuning_udp_bind, cfg.live_tuning_udp_port)
        endpoint = f"udp://{cfg.live_tuning_udp_bind}:{cfg.live_tuning_udp_port}"
        return TuningReceiver(source, endpoint=endpoint)
    except Exception as exc:
        print(f"Warning: live tuning receiver failed to initialize ({exc}); tuning disabled.")
        return TuningReceiver(None, endpoint="disabled")
