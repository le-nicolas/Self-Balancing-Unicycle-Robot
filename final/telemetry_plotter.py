from __future__ import annotations

import argparse
import json
import math
import socket
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import serial  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    serial = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "matplotlib is required for telemetry plotting. Install with: pip install matplotlib"
    ) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live telemetry plotter for MuJoCo sim or hardware serial stream.")
    parser.add_argument("--source", choices=["udp", "serial"], default="udp", help="Telemetry input source.")
    parser.add_argument("--udp-bind", type=str, default="0.0.0.0", help="UDP bind address when --source udp.")
    parser.add_argument("--udp-port", type=int, default=9871, help="UDP bind port when --source udp.")
    parser.add_argument("--serial-port", type=str, default=None, help="Serial COM/TTY path when --source serial.")
    parser.add_argument("--serial-baud", type=int, default=115200, help="Serial baud when --source serial.")
    parser.add_argument("--window-s", type=float, default=12.0, help="Sliding plot window length in seconds.")
    parser.add_argument("--refresh-ms", type=int, default=50, help="Plot refresh interval in milliseconds.")
    parser.add_argument("--max-drain", type=int, default=400, help="Max frames to drain per refresh tick.")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.18,
        help="EWMA smoothing alpha for traces in [0,1]. Set 0 for raw.",
    )
    args = parser.parse_args(argv)
    if args.source == "serial" and not args.serial_port:
        parser.error("--serial-port is required when --source serial")
    args.smooth_alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
    return args


def _first_float(frame: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in frame:
            continue
        try:
            value = float(frame[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def _safe_json_line_to_frame(raw: bytes) -> dict[str, Any] | None:
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


class _UdpReader:
    def __init__(self, bind_host: str, bind_port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, bind_port))
        self._sock.setblocking(False)

    def read_frames(self, max_frames: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max_frames):
            try:
                payload, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            for raw_line in payload.splitlines():
                frame = _safe_json_line_to_frame(raw_line)
                if frame is not None:
                    out.append(frame)
        return out

    def close(self) -> None:
        self._sock.close()


class _SerialReader:
    def __init__(self, port: str, baud: int):
        if serial is None:
            raise RuntimeError("pyserial is required for serial plotting. Install with: pip install pyserial")
        self._ser = serial.Serial(port=port, baudrate=baud, timeout=0.01)

    def read_frames(self, max_frames: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max_frames):
            payload = self._ser.readline()
            if not payload:
                break
            frame = _safe_json_line_to_frame(payload)
            if frame is not None:
                out.append(frame)
        return out

    def close(self) -> None:
        self._ser.close()


@dataclass
class _History:
    t: deque[float]
    pitch_deg: deque[float]
    roll_deg: deque[float]
    pitch_rate_dps: deque[float]
    roll_rate_dps: deque[float]
    wheel_rpm: deque[float]
    u_rw: deque[float]
    u_bx: deque[float]
    u_by: deque[float]
    u_drive: deque[float]

    @classmethod
    def create(cls, maxlen: int = 20000) -> "_History":
        return cls(
            t=deque(maxlen=maxlen),
            pitch_deg=deque(maxlen=maxlen),
            roll_deg=deque(maxlen=maxlen),
            pitch_rate_dps=deque(maxlen=maxlen),
            roll_rate_dps=deque(maxlen=maxlen),
            wheel_rpm=deque(maxlen=maxlen),
            u_rw=deque(maxlen=maxlen),
            u_bx=deque(maxlen=maxlen),
            u_by=deque(maxlen=maxlen),
            u_drive=deque(maxlen=maxlen),
        )

    def append_frame(self, frame: dict[str, Any]) -> None:
        time_s = _first_float(frame, "sim_time_s", "time_s", "t_s", "time")
        if not math.isfinite(time_s):
            time_s = (self.t[-1] + 1e-3) if self.t else 0.0

        pitch_rad = _first_float(frame, "pitch_rad", "pitch")
        roll_rad = _first_float(frame, "roll_rad", "roll")
        pitch_deg = _first_float(frame, "pitch_deg")
        roll_deg = _first_float(frame, "roll_deg")
        if not math.isfinite(pitch_deg):
            pitch_deg = math.degrees(pitch_rad) if math.isfinite(pitch_rad) else float("nan")
        if not math.isfinite(roll_deg):
            roll_deg = math.degrees(roll_rad) if math.isfinite(roll_rad) else float("nan")

        pitch_rate_dps = _first_float(frame, "pitch_rate_dps")
        if not math.isfinite(pitch_rate_dps):
            pitch_rate = _first_float(frame, "pitch_rate_rad_s", "pitch_rate")
            pitch_rate_dps = math.degrees(pitch_rate) if math.isfinite(pitch_rate) else float("nan")

        roll_rate_dps = _first_float(frame, "roll_rate_dps")
        if not math.isfinite(roll_rate_dps):
            roll_rate = _first_float(frame, "roll_rate_rad_s", "roll_rate")
            roll_rate_dps = math.degrees(roll_rate) if math.isfinite(roll_rate) else float("nan")

        wheel_rpm = _first_float(frame, "wheel_rate_rpm", "reaction_speed_rpm")
        if not math.isfinite(wheel_rpm):
            wheel_rate_rad_s = _first_float(frame, "wheel_rate_rad_s", "wheel_rate")
            if math.isfinite(wheel_rate_rad_s):
                wheel_rpm = wheel_rate_rad_s * 60.0 / (2.0 * math.pi)
            else:
                reaction_speed_dps = _first_float(frame, "reaction_speed")
                wheel_rpm = reaction_speed_dps / 6.0 if math.isfinite(reaction_speed_dps) else float("nan")

        u_rw = _first_float(frame, "u_rw_cmd", "u_rw", "rw_cmd_norm", "rt")
        u_bx = _first_float(frame, "u_bx_cmd", "u_bx")
        u_by = _first_float(frame, "u_by_cmd", "u_by")
        u_drive = _first_float(frame, "drive_cmd_norm", "u_drive", "dt")
        if not math.isfinite(u_drive) and math.isfinite(u_bx):
            u_drive = u_bx

        self.t.append(time_s)
        self.pitch_deg.append(pitch_deg)
        self.roll_deg.append(roll_deg)
        self.pitch_rate_dps.append(pitch_rate_dps)
        self.roll_rate_dps.append(roll_rate_dps)
        self.wheel_rpm.append(wheel_rpm)
        self.u_rw.append(u_rw)
        self.u_bx.append(u_bx)
        self.u_by.append(u_by)
        self.u_drive.append(u_drive)

    def prune(self, window_s: float) -> None:
        if not self.t:
            return
        cutoff = self.t[-1] - max(window_s, 1e-3)
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.pitch_deg.popleft()
            self.roll_deg.popleft()
            self.pitch_rate_dps.popleft()
            self.roll_rate_dps.popleft()
            self.wheel_rpm.popleft()
            self.u_rw.popleft()
            self.u_bx.popleft()
            self.u_by.popleft()
            self.u_drive.popleft()


def _axis_autoscale(ax, series: list[deque[float]], *, floor_span: float = 1.0) -> None:
    values: list[float] = []
    for s in series:
        values.extend(v for v in s if math.isfinite(v))
    if not values:
        return
    arr = np.asarray(values, dtype=float)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi <= lo:
        center = float(np.mean(arr))
        span = max(abs(center) * 0.3, floor_span)
        ax.set_ylim(center - span, center + span)
        return
    span = hi - lo
    pad = max(0.12 * span, floor_span * 0.12)
    ax.set_ylim(lo - pad, hi + pad)


def _smooth(values: deque[float], alpha: float) -> list[float]:
    raw = list(values)
    if alpha <= 0.0 or len(raw) < 2:
        return raw
    out: list[float] = []
    prev: float | None = None
    a = float(np.clip(alpha, 0.0, 1.0))
    for v in raw:
        if not math.isfinite(v):
            out.append(v)
            continue
        if prev is None or (not math.isfinite(prev)):
            prev = v
        else:
            prev = a * v + (1.0 - a) * prev
        out.append(prev)
    return out


def _fmt(v: float, fmt: str = "{:+.2f}") -> str:
    return fmt.format(v) if math.isfinite(v) else "nan"


def _last_finite(values: deque[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return float(v)
    return float("nan")


def main(argv=None):
    args = parse_args(argv)
    if args.source == "udp":
        reader = _UdpReader(args.udp_bind, int(args.udp_port))
        source_label = f"udp://{args.udp_bind}:{args.udp_port}"
    else:
        reader = _SerialReader(args.serial_port, int(args.serial_baud))
        source_label = f"serial://{args.serial_port}@{args.serial_baud}"

    history = _History.create()
    total_frames = 0

    fig = plt.figure(figsize=(14, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_att = fig.add_subplot(gs[0, 0])
    ax_rates = fig.add_subplot(gs[1, 0], sharex=ax_att)
    ax_cmd = fig.add_subplot(gs[0, 1], sharex=ax_att)
    ax_wheel = fig.add_subplot(gs[1, 1], sharex=ax_att)

    fig.patch.set_facecolor("#f4f7fb")
    for ax in (ax_att, ax_rates, ax_cmd, ax_wheel):
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#d8dee9", alpha=0.7, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#b0b8c4")
        ax.tick_params(colors="#2b3442", labelsize=9)

    fig.suptitle("Telemetry Dashboard", fontsize=16, fontweight="bold", color="#1d2530")
    fig.text(
        0.01,
        0.965,
        f"Source: {source_label}   |   Window: {args.window_s:.1f}s   |   Smooth alpha: {args.smooth_alpha:.2f}",
        fontsize=9,
        color="#4b5563",
    )

    ax_att.set_title("Attitude", fontsize=11, fontweight="bold")
    ax_rates.set_title("Body Rates", fontsize=11, fontweight="bold")
    ax_cmd.set_title("Commands", fontsize=11, fontweight="bold")
    ax_wheel.set_title("Reaction Wheel", fontsize=11, fontweight="bold")

    ax_att.set_ylabel("deg")
    ax_rates.set_ylabel("deg/s")
    ax_cmd.set_ylabel("command")
    ax_wheel.set_ylabel("rpm")
    ax_rates.set_xlabel("sim time (s)")
    ax_wheel.set_xlabel("sim time (s)")

    ax_att.axhline(0.0, color="#8b949e", linewidth=1.0, alpha=0.8)
    ax_rates.axhline(0.0, color="#8b949e", linewidth=1.0, alpha=0.8)
    ax_cmd.axhline(0.0, color="#8b949e", linewidth=1.0, alpha=0.8)
    ax_wheel.axhline(0.0, color="#8b949e", linewidth=1.0, alpha=0.8)

    # Reference band near upright.
    ax_att.axhspan(-5.0, 5.0, color="#e8f5e9", alpha=0.55, zorder=0)

    line_pitch, = ax_att.plot([], [], color="#0f4c81", linewidth=2.1, label="pitch (deg)")
    line_roll, = ax_att.plot([], [], color="#c0392b", linewidth=2.1, label="roll (deg)")
    line_pitch_rate, = ax_rates.plot([], [], color="#00897b", linewidth=1.8, label="pitch rate (deg/s)")
    line_roll_rate, = ax_rates.plot([], [], color="#f57c00", linewidth=1.8, label="roll rate (deg/s)")
    line_u_rw, = ax_cmd.plot([], [], color="#6d4c41", linewidth=1.8, label="u_rw")
    line_u_bx, = ax_cmd.plot([], [], color="#1e88e5", linewidth=1.6, label="u_bx/drive")
    line_u_by, = ax_cmd.plot([], [], color="#43a047", linewidth=1.6, label="u_by")
    line_wheel, = ax_wheel.plot([], [], color="#5d4037", linewidth=1.9, label="wheel speed (rpm)")

    ax_att.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax_rates.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax_cmd.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax_wheel.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)

    status_text = fig.text(0.01, 0.01, "", fontsize=9, color="#2d3748")

    plotted_lines = [
        line_pitch,
        line_roll,
        line_pitch_rate,
        line_roll_rate,
        line_u_rw,
        line_u_bx,
        line_u_by,
        line_wheel,
    ]

    def _refresh(_frame_id: int):
        nonlocal total_frames
        frames = reader.read_frames(max_frames=int(max(args.max_drain, 1)))
        total_frames += len(frames)
        for frame in frames:
            history.append_frame(frame)
        if not history.t:
            return plotted_lines
        history.prune(args.window_s)
        x = list(history.t)

        pitch = _smooth(history.pitch_deg, args.smooth_alpha)
        roll = _smooth(history.roll_deg, args.smooth_alpha)
        pitch_rate = _smooth(history.pitch_rate_dps, args.smooth_alpha)
        roll_rate = _smooth(history.roll_rate_dps, args.smooth_alpha)
        u_rw = _smooth(history.u_rw, args.smooth_alpha)
        u_bx = _smooth(history.u_bx, args.smooth_alpha)
        u_by = _smooth(history.u_by, args.smooth_alpha)
        wheel = _smooth(history.wheel_rpm, args.smooth_alpha)

        line_pitch.set_data(x, pitch)
        line_roll.set_data(x, roll)
        line_pitch_rate.set_data(x, pitch_rate)
        line_roll_rate.set_data(x, roll_rate)
        line_u_rw.set_data(x, u_rw)
        line_u_bx.set_data(x, u_bx)
        line_u_by.set_data(x, u_by)
        line_wheel.set_data(x, wheel)

        x_max = x[-1]
        x_min = max(0.0, x_max - max(args.window_s, 1e-3))
        for ax in (ax_att, ax_rates, ax_cmd, ax_wheel):
            ax.set_xlim(x_min, x_max if x_max > x_min else x_min + 1e-3)

        _axis_autoscale(ax_att, [history.pitch_deg, history.roll_deg], floor_span=4.0)
        _axis_autoscale(ax_rates, [history.pitch_rate_dps, history.roll_rate_dps], floor_span=20.0)
        _axis_autoscale(ax_cmd, [history.u_rw, history.u_bx, history.u_by], floor_span=0.2)
        _axis_autoscale(ax_wheel, [history.wheel_rpm], floor_span=20.0)

        status_text.set_text(
            "frames={frames}  t={t:.2f}s  pitch={pitch}  roll={roll}  pr={pr}  rr={rr}  wheel={wheel}  u_rw={u_rw}".format(
                frames=total_frames,
                t=x_max,
                pitch=_fmt(_last_finite(history.pitch_deg), "{:+.2f}deg"),
                roll=_fmt(_last_finite(history.roll_deg), "{:+.2f}deg"),
                pr=_fmt(_last_finite(history.pitch_rate_dps), "{:+.1f}deg/s"),
                rr=_fmt(_last_finite(history.roll_rate_dps), "{:+.1f}deg/s"),
                wheel=_fmt(_last_finite(history.wheel_rpm), "{:+.1f}rpm"),
                u_rw=_fmt(_last_finite(history.u_rw), "{:+.3f}"),
            )
        )
        return plotted_lines

    anim = FuncAnimation(
        fig,
        _refresh,
        interval=max(int(args.refresh_ms), 20),
        blit=False,
        cache_frame_data=False,
    )
    _ = anim
    try:
        plt.show()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
