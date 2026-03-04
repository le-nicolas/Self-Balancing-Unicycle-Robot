from __future__ import annotations

import argparse
import csv
import json
import shlex
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from control_core import (
    apply_control_delay,
    apply_upright_postprocess,
    base_commands_with_limits,
    compute_control_command,
    reset_controller_buffers,
    update_disturbance_observer,
    update_gain_schedule,
    wheel_command_with_limits,
)
from runtime_config import build_config, parse_args as runtime_parse_args
from runtime_model import (
    build_kalman_gain,
    build_measurement_noise_cov,
    build_partial_measurement_matrix,
    enforce_planar_root_attitude,
    lookup_model_ids,
    reset_state,
)

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    MATPLOTLIB_AVAILABLE = False
    animation = None
    plt = None
    Slider = None
    Button = None


def _solve_discrete_are_robust(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    reg_steps = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    eye = np.eye(R.shape[0], dtype=float)
    last_exc: Exception | None = None
    for eps in reg_steps:
        try:
            P = solve_discrete_are(A, B, Q, R + eps * eye)
            if np.all(np.isfinite(P)):
                return 0.5 * (P + P.T)
        except Exception as exc:  # pragma: no cover - numerical fallback
            last_exc = exc
    raise RuntimeError(f"DARE failed after regularization fallback: {last_exc}") from last_exc


def _solve_linear_robust(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("Linear solve returned non-finite result.")
        return sol


def _lift_discrete_dynamics(A: np.ndarray, B: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(int(steps), 1)
    if n == 1:
        return A.copy(), B.copy()
    A_lift = np.linalg.matrix_power(A, n)
    acc = np.zeros_like(A)
    Ak = np.eye(A.shape[0], dtype=float)
    for _ in range(n):
        acc += Ak
        Ak = A @ Ak
    B_lift = acc @ B
    return A_lift, B_lift


@dataclass
class SensorSample:
    ax: float
    ay: float
    az: float
    roll_rate_rad_s: float
    pitch_rate_rad_s: float
    reaction_speed_rad_s: float
    ts_us: int | None
    seq: int | None


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    vmin: float
    vmax: float
    vinit: float
    step: float | None = None


class ComplementaryAttitude:
    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.pitch_rad = 0.0
        self.roll_rad = 0.0

    def update(
        self,
        ax: float,
        ay: float,
        az: float,
        roll_rate_rad_s: float,
        pitch_rate_rad_s: float,
        dt: float,
    ) -> tuple[float, float, float, float]:
        accel_pitch = float(np.arctan2(ax, np.sqrt(ay * ay + az * az)))
        accel_roll = float(np.arctan2(ay, np.sqrt(ax * ax + az * az)))
        self.pitch_rad = self.alpha * (self.pitch_rad + pitch_rate_rad_s * dt) + (1.0 - self.alpha) * accel_pitch
        self.roll_rad = self.alpha * (self.roll_rad + roll_rate_rad_s * dt) + (1.0 - self.alpha) * accel_roll
        return self.pitch_rad, self.roll_rad, pitch_rate_rad_s, roll_rate_rad_s


class StubControlBackend:
    outputs_normalized = True

    def __init__(self, imu_alpha: float):
        self.att = ComplementaryAttitude(alpha=imu_alpha)
        self.x_est = np.zeros(9, dtype=float)
        self.u_eff_applied = np.zeros(3, dtype=float)
        self.kp_pitch = 8.0
        self.kd_pitch = 0.6
        self.kp_roll = 6.0
        self.kd_roll = 0.5

    def step(self, sample: SensorSample, dt: float) -> tuple[np.ndarray, np.ndarray]:
        pitch, roll, pitch_rate, roll_rate = self.att.update(
            sample.ax,
            sample.ay,
            sample.az,
            sample.roll_rate_rad_s,
            sample.pitch_rate_rad_s,
            dt,
        )
        self.x_est[:] = [pitch, roll, pitch_rate, roll_rate, sample.reaction_speed_rad_s, 0.0, 0.0, 0.0, 0.0]
        rw_cmd = float(np.clip(-(self.kp_roll * roll + self.kd_roll * roll_rate), -1.0, 1.0))
        drive_cmd = float(np.clip(-(self.kp_pitch * pitch + self.kd_pitch * pitch_rate), -1.0, 1.0))
        self.u_eff_applied[:] = [rw_cmd, drive_cmd, 0.0]
        return self.x_est.copy(), self.u_eff_applied.copy()

    def apply_live_tuning(self, updates: dict[str, float]):
        if "stub_kp_pitch" in updates:
            self.kp_pitch = float(max(updates["stub_kp_pitch"], 0.0))
        if "stub_kd_pitch" in updates:
            self.kd_pitch = float(max(updates["stub_kd_pitch"], 0.0))
        if "stub_kp_roll" in updates:
            self.kp_roll = float(max(updates["stub_kp_roll"], 0.0))
        if "stub_kd_roll" in updates:
            self.kd_roll = float(max(updates["stub_kd_roll"], 0.0))

    def live_tuning_values(self) -> dict[str, float]:
        return {
            "stub_kp_pitch": float(self.kp_pitch),
            "stub_kd_pitch": float(self.kd_pitch),
            "stub_kp_roll": float(self.kp_roll),
            "stub_kd_roll": float(self.kd_roll),
        }


class RuntimeControlBackend:
    outputs_normalized = False

    def __init__(self, runtime_args: str, imu_alpha: float):
        parsed = runtime_parse_args(shlex.split(runtime_args))
        self.cfg = build_config(parsed)
        self.att = ComplementaryAttitude(alpha=imu_alpha)

        self.A, self.B = self._build_discrete_model(self.cfg)
        self.NX = self.A.shape[0]
        self.NU = self.B.shape[1]
        self.B_pinv = np.linalg.pinv(self.B)

        A_aug = np.block([[self.A, self.B], [np.zeros((self.NU, self.NX)), np.eye(self.NU)]])
        B_aug = np.vstack([self.B, np.eye(self.NU)])
        Q_aug = np.block([[self.cfg.qx, np.zeros((self.NX, self.NU))], [np.zeros((self.NU, self.NX)), self.cfg.qu]])
        P_aug = _solve_discrete_are_robust(A_aug, B_aug, Q_aug, self.cfg.r_du)
        self.K_du = _solve_linear_robust(B_aug.T @ P_aug @ B_aug + self.cfg.r_du, B_aug.T @ P_aug @ A_aug)

        self.K_paper_pitch = None
        self.K_wheel_only = None
        need_paper_pitch = self.cfg.wheel_only or (self.cfg.controller_family == "paper_split_baseline")
        if need_paper_pitch:
            A_w = self.A[np.ix_([0, 2, 4], [0, 2, 4])]
            B_w = self.B[np.ix_([0, 2, 4], [0])]
            Q_w = np.diag([260.0, 35.0, 0.6])
            R_w = np.array([[0.08]])
            try:
                P_w = _solve_discrete_are_robust(A_w, B_w, Q_w, R_w)
                self.K_paper_pitch = _solve_linear_robust(B_w.T @ P_w @ B_w + R_w, B_w.T @ P_w @ A_w)
            except RuntimeError:
                self.K_paper_pitch = None
        if self.cfg.wheel_only:
            if self.K_paper_pitch is not None:
                self.K_wheel_only = self.K_paper_pitch.copy()
            else:
                self.K_wheel_only = np.array(
                    [[self.cfg.wheel_only_pitch_kp, self.cfg.wheel_only_pitch_kd, 0.0]],
                    dtype=float,
                )

        control_dt = max(1.0 / max(self.cfg.control_hz, 1e-6), 1e-6)
        self.wheel_lsb = (2.0 * np.pi) / (self.cfg.wheel_encoder_ticks_per_rev * control_dt)
        self.C = build_partial_measurement_matrix(self.cfg)
        Qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        Rn = build_measurement_noise_cov(self.cfg, self.wheel_lsb)
        self._Qn_base = Qn.copy()
        self._Rn_base = Rn.copy()
        self.estimator_q_scale = 1.0
        self.estimator_r_scale = 1.0
        self.L = build_kalman_gain(self.A, Qn, self.C, Rn)

        queue_len = 1 if not self.cfg.hardware_realistic else self.cfg.control_delay_steps + 1
        (
            self.x_est,
            self.u_applied,
            self.u_eff_applied,
            self.base_int,
            self.wheel_pitch_int,
            self.wheel_momentum_bias_int,
            self.base_ref,
            self.base_authority_state,
            self.u_base_smooth,
            self.balance_phase,
            self.recovery_time_s,
            self.high_spin_active,
            self.cmd_queue,
        ) = reset_controller_buffers(self.NX, self.NU, queue_len)

        self.du_hits = np.zeros(3, dtype=int)
        self.sat_hits = np.zeros(3, dtype=int)
        self.dob_hat = np.zeros(3, dtype=float)
        self.dob_raw = np.zeros(3, dtype=float)
        self.dob_prev_x_est: np.ndarray | None = None
        self.gain_schedule_scale_state = 1.0
        self.disturbance_level = 0.0
        self.upright_blend = 0.0
        self.despin_gain = 0.25

    def _build_discrete_model(self, cfg) -> tuple[np.ndarray, np.ndarray]:
        xml_path = Path(__file__).with_name("final.xml")
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        ids = lookup_model_ids(model)
        reset_state(
            model,
            data,
            ids.q_pitch,
            ids.q_roll,
            pitch_eq=cfg.linearize_pitch_rad,
            roll_eq=cfg.linearize_roll_rad,
        )
        if cfg.lock_root_attitude:
            enforce_planar_root_attitude(model, data, ids)

        nx_full = 2 * model.nv + model.na
        A_full = np.zeros((nx_full, nx_full))
        B_full = np.zeros((nx_full, model.nu))
        mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)

        idx = [
            ids.v_pitch,
            ids.v_roll,
            model.nv + ids.v_pitch,
            model.nv + ids.v_roll,
            model.nv + ids.v_rw,
            ids.v_base_x,
            ids.v_base_y,
            model.nv + ids.v_base_x,
            model.nv + ids.v_base_y,
        ]
        A_step = A_full[np.ix_(idx, idx)]
        B_step = B_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
        control_steps = (
            1
            if not cfg.hardware_realistic
            else max(1, int(round(1.0 / max(model.opt.timestep * cfg.control_hz, 1e-9))))
        )
        return _lift_discrete_dynamics(A_step, B_step, control_steps)

    def _set_cfg_float(self, key: str, value: float):
        object.__setattr__(self.cfg, key, float(value))

    def _recompute_kalman_gain(self):
        q_scale = float(np.clip(self.estimator_q_scale, 1e-4, 1e4))
        r_scale = float(np.clip(self.estimator_r_scale, 1e-4, 1e4))
        self.estimator_q_scale = q_scale
        self.estimator_r_scale = r_scale
        self.L = build_kalman_gain(self.A, self._Qn_base * q_scale, self.C, self._Rn_base * r_scale)

    def apply_live_tuning(self, updates: dict[str, float]):
        cfg_keys = {
            "base_pitch_kp",
            "base_pitch_kd",
            "base_roll_kp",
            "base_roll_kd",
            "wheel_momentum_k",
            "wheel_momentum_thresh_frac",
            "u_bleed",
        }
        for key, value in updates.items():
            if key in cfg_keys:
                self._set_cfg_float(key, float(value))
            elif key == "estimator_q_scale":
                self.estimator_q_scale = float(value)
            elif key == "estimator_r_scale":
                self.estimator_r_scale = float(value)

        if ("estimator_q_scale" in updates) or ("estimator_r_scale" in updates):
            self._recompute_kalman_gain()

    def live_tuning_values(self) -> dict[str, float]:
        return {
            "base_pitch_kp": float(self.cfg.base_pitch_kp),
            "base_pitch_kd": float(self.cfg.base_pitch_kd),
            "base_roll_kp": float(self.cfg.base_roll_kp),
            "base_roll_kd": float(self.cfg.base_roll_kd),
            "wheel_momentum_k": float(self.cfg.wheel_momentum_k),
            "wheel_momentum_thresh_frac": float(self.cfg.wheel_momentum_thresh_frac),
            "u_bleed": float(self.cfg.u_bleed),
            "estimator_q_scale": float(self.estimator_q_scale),
            "estimator_r_scale": float(self.estimator_r_scale),
        }

    def step(self, sample: SensorSample, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = float(np.clip(dt, 1e-4, 0.05))
        pitch, roll, pitch_rate, roll_rate = self.att.update(
            sample.ax,
            sample.ay,
            sample.az,
            sample.roll_rate_rad_s,
            sample.pitch_rate_rad_s,
            dt,
        )
        y = np.array([pitch, roll, pitch_rate, roll_rate, sample.reaction_speed_rad_s], dtype=float)
        if self.cfg.base_state_from_sensors:
            y = np.concatenate([y, np.zeros(4, dtype=float)])

        x_pred = self.A @ self.x_est + self.B @ self.u_eff_applied
        self.x_est = x_pred + self.L @ (y - self.C @ x_pred)
        if not self.cfg.base_state_from_sensors:
            self.x_est[5:] = 0.0

        angle_mag = max(abs(float(self.x_est[0])), abs(float(self.x_est[1])))
        rate_mag = max(abs(float(self.x_est[2])), abs(float(self.x_est[3])))
        if self.balance_phase == "recovery":
            if angle_mag < self.cfg.hold_enter_angle_rad and rate_mag < self.cfg.hold_enter_rate_rad_s:
                self.balance_phase = "hold"
                self.recovery_time_s = 0.0
        else:
            if angle_mag > self.cfg.hold_exit_angle_rad or rate_mag > self.cfg.hold_exit_rate_rad_s:
                self.balance_phase = "recovery"
                self.recovery_time_s = 0.0
        if self.balance_phase == "recovery":
            self.recovery_time_s += dt

        if self.cfg.dob_enabled:
            self.dob_hat, self.dob_raw = update_disturbance_observer(
                cfg=self.cfg,
                A=self.A,
                B=self.B,
                B_pinv=self.B_pinv,
                x_prev=self.dob_prev_x_est,
                u_prev=self.u_eff_applied,
                x_curr=self.x_est,
                dob_hat=self.dob_hat,
                control_dt=dt,
            )
            if self.cfg.gain_schedule_enabled:
                self.gain_schedule_scale_state, self.disturbance_level = update_gain_schedule(
                    cfg=self.cfg,
                    dob_hat=self.dob_hat,
                    gain_schedule_state=self.gain_schedule_scale_state,
                    control_dt=dt,
                )
            else:
                self.gain_schedule_scale_state = 1.0
                self.disturbance_level = float(np.linalg.norm(self.cfg.gain_schedule_weights * self.dob_hat))
        else:
            self.dob_hat[:] = 0.0
            self.dob_raw[:] = 0.0
            self.gain_schedule_scale_state = 1.0
            self.disturbance_level = 0.0

        (
            u_cmd,
            self.base_int,
            self.base_ref,
            self.base_authority_state,
            self.u_base_smooth,
            self.wheel_pitch_int,
            self.wheel_momentum_bias_int,
            rw_u_limit,
            _wheel_over_budget,
            _wheel_over_hard,
            self.high_spin_active,
            _control_terms,
        ) = compute_control_command(
            cfg=self.cfg,
            x_est=self.x_est,
            x_true=self.x_est,
            u_eff_applied=self.u_eff_applied,
            base_int=self.base_int,
            base_ref=self.base_ref,
            base_authority_state=self.base_authority_state,
            u_base_smooth=self.u_base_smooth,
            wheel_pitch_int=self.wheel_pitch_int,
            wheel_momentum_bias_int=self.wheel_momentum_bias_int,
            balance_phase=self.balance_phase,
            recovery_time_s=self.recovery_time_s,
            high_spin_active=self.high_spin_active,
            control_dt=dt,
            K_du=self.K_du,
            K_wheel_only=self.K_wheel_only,
            K_paper_pitch=self.K_paper_pitch,
            du_hits=self.du_hits,
            sat_hits=self.sat_hits,
            dob_compensation=self.dob_hat,
            gain_schedule_scale=self.gain_schedule_scale_state,
            disturbance_level=self.disturbance_level,
            mpc_controller=None,
        )
        self.dob_prev_x_est = self.x_est.copy()

        u_cmd, self.upright_blend = apply_upright_postprocess(
            cfg=self.cfg,
            u_cmd=u_cmd,
            x_est=self.x_est,
            x_true=self.x_est,
            upright_blend=self.upright_blend,
            balance_phase=self.balance_phase,
            high_spin_active=self.high_spin_active,
            despin_gain=self.despin_gain,
            rw_u_limit=rw_u_limit,
        )
        self.u_applied = apply_control_delay(self.cfg, self.cmd_queue, u_cmd)

        wheel_cmd = wheel_command_with_limits(
            cfg=self.cfg,
            wheel_speed=float(self.x_est[4]),
            wheel_cmd_requested=float(self.u_applied[0]),
        )
        if self.cfg.allow_base_motion:
            base_x_cmd, base_y_cmd = base_commands_with_limits(
                cfg=self.cfg,
                base_x_speed=float(self.x_est[7]),
                base_y_speed=float(self.x_est[8]),
                base_x=float(self.x_est[5]),
                base_y=float(self.x_est[6]),
                base_x_request=float(self.u_applied[1]),
                base_y_request=float(self.u_applied[2]),
            )
        else:
            base_x_cmd = 0.0
            base_y_cmd = 0.0

        self.u_eff_applied[:] = [wheel_cmd, base_x_cmd, base_y_cmd]
        return self.x_est.copy(), self.u_eff_applied.copy()


class HILBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.loop_s = 1.0 / max(args.loop_hz, 1e-6)
        self.running = False
        self.estop = False

        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.bind((args.pc_ip, args.pc_port))
        self.rx_sock.settimeout(max(args.rx_timeout_ms, 1.0) / 1000.0)
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.esp_addr = (args.esp32_ip, args.esp32_port)

        self.backend = StubControlBackend(imu_alpha=args.imu_alpha) if args.stub_control else RuntimeControlBackend(
            runtime_args=args.runtime_args,
            imu_alpha=args.imu_alpha,
        )
        self.backend_lock = threading.Lock()

        self.rw_cmd_scale = float(
            args.rw_cmd_scale
            if args.rw_cmd_scale is not None
            else (
                float(getattr(getattr(self.backend, "cfg", None), "max_u", np.array([1.0]))[0])
                if hasattr(self.backend, "cfg")
                else 1.0
            )
        )
        self.drive_cmd_scale = float(
            args.drive_cmd_scale
            if args.drive_cmd_scale is not None
            else (
                float(getattr(getattr(self.backend, "cfg", None), "max_u", np.array([1.0, 1.0]))[1])
                if hasattr(self.backend, "cfg")
                else 1.0
            )
        )
        self.rw_cmd_scale = max(abs(self.rw_cmd_scale), 1e-6)
        self.drive_cmd_scale = max(abs(self.drive_cmd_scale), 1e-6)

        self.last_ts_us: int | None = None
        self.last_rx_wall: float | None = None
        self.last_seq: int | None = None
        self.missed_packets = 0
        self.timeout_packets = 0
        self.bad_json_packets = 0
        self.runtime_errors = 0
        self.cmd_seq = 0

        self.start_time = time.perf_counter()
        self.last_console_print = self.start_time
        self.last_state = np.zeros(9, dtype=float)
        self.last_cmd_norm = np.zeros(2, dtype=float)

        n_log = int(max(2, round(args.plot_window_s * args.loop_hz)))
        self.log_lock = threading.Lock()
        self.log_time = deque(maxlen=n_log)
        self.log_pitch_deg = deque(maxlen=n_log)
        self.log_roll_deg = deque(maxlen=n_log)
        self.log_reaction_norm = deque(maxlen=n_log)
        self.log_drive_norm = deque(maxlen=n_log)
        self.log_reaction_speed_dps = deque(maxlen=n_log)

        self.csv_file = None
        self.csv_writer = None
        if args.csv_log:
            csv_path = Path(args.csv_log).expanduser()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "wall_time_s",
                    "pitch_deg",
                    "roll_deg",
                    "pitch_rate_dps",
                    "roll_rate_dps",
                    "reaction_speed_dps",
                    "rw_cmd_norm",
                    "drive_cmd_norm",
                    "missed_packets",
                    "timeout_packets",
                    "estop",
                ],
            )
            self.csv_writer.writeheader()

    def _scale_range(self, value: float) -> tuple[float, float]:
        v = float(max(abs(value), 1e-6))
        return max(v * 0.1, 1e-4), max(v * 3.0, v + 0.05)

    def slider_specs(self) -> list[SliderSpec]:
        rw_min, rw_max = self._scale_range(self.rw_cmd_scale)
        drive_min, drive_max = self._scale_range(self.drive_cmd_scale)
        specs: list[SliderSpec] = [
            SliderSpec("rw_cmd_scale", "rw_scale", rw_min, rw_max, float(self.rw_cmd_scale)),
            SliderSpec("drive_cmd_scale", "drive_scale", drive_min, drive_max, float(self.drive_cmd_scale)),
        ]
        if self.args.stub_control:
            specs.extend(
                [
                    SliderSpec("stub_kp_pitch", "kp_pitch", 0.0, 40.0, float(self.backend.kp_pitch)),
                    SliderSpec("stub_kd_pitch", "kd_pitch", 0.0, 8.0, float(self.backend.kd_pitch)),
                    SliderSpec("stub_kp_roll", "kp_roll", 0.0, 40.0, float(self.backend.kp_roll)),
                    SliderSpec("stub_kd_roll", "kd_roll", 0.0, 8.0, float(self.backend.kd_roll)),
                ]
            )
        else:
            cfg = self.backend.cfg
            specs.extend(
                [
                    SliderSpec("base_pitch_kp", "base_p_kp", 0.0, 180.0, float(cfg.base_pitch_kp)),
                    SliderSpec("base_pitch_kd", "base_p_kd", 0.0, 40.0, float(cfg.base_pitch_kd)),
                    SliderSpec("base_roll_kp", "base_r_kp", 0.0, 120.0, float(cfg.base_roll_kp)),
                    SliderSpec("base_roll_kd", "base_r_kd", 0.0, 30.0, float(cfg.base_roll_kd)),
                    SliderSpec("wheel_momentum_k", "rw_mom_k", 0.0, 2.0, float(cfg.wheel_momentum_k)),
                    SliderSpec(
                        "wheel_momentum_thresh_frac",
                        "rw_mom_thr",
                        0.05,
                        0.90,
                        float(cfg.wheel_momentum_thresh_frac),
                    ),
                    SliderSpec("u_bleed", "u_bleed", 0.60, 1.00, float(cfg.u_bleed)),
                    SliderSpec("estimator_q_scale", "kf_q_scale", 0.05, 10.0, 1.0),
                    SliderSpec("estimator_r_scale", "kf_r_scale", 0.05, 10.0, 1.0),
                ]
            )
        return specs

    def live_tuning_values(self) -> dict[str, float]:
        with self.backend_lock:
            values = self.backend.live_tuning_values()
            values["rw_cmd_scale"] = float(self.rw_cmd_scale)
            values["drive_cmd_scale"] = float(self.drive_cmd_scale)
            return values

    def apply_live_tuning(self, updates: dict[str, float]) -> dict[str, float]:
        with self.backend_lock:
            backend_updates: dict[str, float] = {}
            for key, value in updates.items():
                if key == "rw_cmd_scale":
                    self.rw_cmd_scale = float(max(abs(value), 1e-6))
                elif key == "drive_cmd_scale":
                    self.drive_cmd_scale = float(max(abs(value), 1e-6))
                else:
                    backend_updates[key] = float(value)
            if backend_updates:
                self.backend.apply_live_tuning(backend_updates)
            values = self.backend.live_tuning_values()
            values["rw_cmd_scale"] = float(self.rw_cmd_scale)
            values["drive_cmd_scale"] = float(self.drive_cmd_scale)
            return values

    def clear_estop(self):
        self.estop = False

    def _parse_packet(self, packet: dict) -> SensorSample:
        reaction_speed_dps = float(packet.get("reaction_speed", packet.get("wheel_speed", 0.0)))
        ts_us = packet.get("ts", None)
        seq = packet.get("seq", None)
        return SensorSample(
            ax=float(packet.get("ax", 0.0)) * float(self.args.accel_x_sign),
            ay=float(packet.get("ay", 0.0)) * float(self.args.accel_y_sign),
            az=float(packet.get("az", 0.0)) * float(self.args.accel_z_sign),
            roll_rate_rad_s=np.radians(float(packet.get("gx", 0.0)) * float(self.args.roll_rate_sign)),
            pitch_rate_rad_s=np.radians(float(packet.get("gy", 0.0)) * float(self.args.pitch_rate_sign)),
            reaction_speed_rad_s=np.radians(reaction_speed_dps * float(self.args.reaction_speed_sign)),
            ts_us=int(ts_us) if ts_us is not None else None,
            seq=int(seq) if seq is not None else None,
        )

    def _compute_dt(self, sample: SensorSample, now: float) -> float:
        if sample.ts_us is not None and self.last_ts_us is not None:
            if sample.ts_us >= self.last_ts_us:
                delta_us = sample.ts_us - self.last_ts_us
            else:
                delta_us = (2**32 - self.last_ts_us) + sample.ts_us
            dt = 1e-6 * float(delta_us)
        elif self.last_rx_wall is not None:
            dt = now - self.last_rx_wall
        else:
            dt = self.loop_s
        self.last_ts_us = sample.ts_us if sample.ts_us is not None else self.last_ts_us
        self.last_rx_wall = now
        return float(np.clip(dt, 1e-4, 0.05))

    def _update_seq_stats(self, sample: SensorSample):
        if sample.seq is None:
            return
        if self.last_seq is not None:
            delta = (sample.seq - self.last_seq) & 0xFFFFFFFF
            if delta > 1:
                self.missed_packets += int(delta - 1)
        self.last_seq = sample.seq

    def _map_drive_channel(self, u_eff: np.ndarray) -> float:
        if self.args.drive_channel == "bx":
            return float(u_eff[1])
        if self.args.drive_channel == "by":
            return float(u_eff[2])
        return float(u_eff[1] + u_eff[2])

    def _normalize_commands(self, u_eff: np.ndarray) -> tuple[float, float]:
        if self.backend.outputs_normalized:
            rw_norm = float(u_eff[0])
            drive_norm = self._map_drive_channel(u_eff)
        else:
            rw_norm = float(u_eff[0]) / self.rw_cmd_scale
            drive_norm = self._map_drive_channel(u_eff) / self.drive_cmd_scale

        rw_norm = float(
            np.clip(
                rw_norm * float(self.args.reaction_sign),
                -float(self.args.max_reaction_norm),
                float(self.args.max_reaction_norm),
            )
        )
        drive_norm = float(
            np.clip(
                drive_norm * float(self.args.drive_sign),
                -float(self.args.max_drive_norm),
                float(self.args.max_drive_norm),
            )
        )
        return rw_norm, drive_norm

    def _send_command(self, rw_norm: float, drive_norm: float):
        if self.estop:
            rw_norm = 0.0
            drive_norm = 0.0
        payload = {
            "rt": rw_norm,
            "dt": drive_norm,
            "estop": int(self.estop),
            "seq": int(self.cmd_seq),
        }
        self.cmd_seq = (self.cmd_seq + 1) & 0xFFFFFFFF
        message = json.dumps(payload, separators=(",", ":"))
        self.tx_sock.sendto(message.encode("utf-8"), self.esp_addr)
        self.last_cmd_norm[:] = [rw_norm, drive_norm]

    def _log_state(self, x_est: np.ndarray, sample: SensorSample, rw_norm: float, drive_norm: float):
        now = time.perf_counter()
        t_rel = now - self.start_time
        pitch_deg = float(np.degrees(x_est[0]))
        roll_deg = float(np.degrees(x_est[1]))
        reaction_speed_dps = float(np.degrees(sample.reaction_speed_rad_s))
        self.last_state = x_est.copy()

        with self.log_lock:
            self.log_time.append(t_rel)
            self.log_pitch_deg.append(pitch_deg)
            self.log_roll_deg.append(roll_deg)
            self.log_reaction_norm.append(rw_norm)
            self.log_drive_norm.append(drive_norm)
            self.log_reaction_speed_dps.append(reaction_speed_dps)

        if self.csv_writer is not None:
            self.csv_writer.writerow(
                {
                    "wall_time_s": f"{t_rel:.6f}",
                    "pitch_deg": f"{pitch_deg:.6f}",
                    "roll_deg": f"{roll_deg:.6f}",
                    "pitch_rate_dps": f"{np.degrees(sample.pitch_rate_rad_s):.6f}",
                    "roll_rate_dps": f"{np.degrees(sample.roll_rate_rad_s):.6f}",
                    "reaction_speed_dps": f"{reaction_speed_dps:.6f}",
                    "rw_cmd_norm": f"{rw_norm:.6f}",
                    "drive_cmd_norm": f"{drive_norm:.6f}",
                    "missed_packets": self.missed_packets,
                    "timeout_packets": self.timeout_packets,
                    "estop": int(self.estop),
                }
            )
            self.csv_file.flush()

        if now - self.last_console_print >= 1.0:
            print(
                f"[{t_rel:7.2f}s] pitch={pitch_deg:+7.2f}deg roll={roll_deg:+7.2f}deg "
                f"rw={rw_norm:+.3f} drive={drive_norm:+.3f} "
                f"missed={self.missed_packets} timeout={self.timeout_packets} estop={int(self.estop)}"
            )
            self.last_console_print = now

    def control_loop(self):
        print(f"HIL bridge running at {self.args.loop_hz:.1f} Hz")
        print(
            f"UDP listen={self.args.pc_ip}:{self.args.pc_port} "
            f"send={self.args.esp32_ip}:{self.args.esp32_port}"
        )
        print("Live parameter ID mode: tuning controller/estimator on real hardware using the existing stack.")
        if not self.args.stub_control:
            print(f"Runtime args: {self.args.runtime_args}")
            print(f"Command scales: rw={self.rw_cmd_scale:.6f} drive={self.drive_cmd_scale:.6f}")
        print("Press Ctrl+C to stop.")

        while self.running:
            loop_start = time.perf_counter()
            try:
                data, _addr = self.rx_sock.recvfrom(1024)
                packet = json.loads(data.decode("utf-8"))
                sample = self._parse_packet(packet)
                self._update_seq_stats(sample)
                dt = self._compute_dt(sample, now=time.perf_counter())
                with self.backend_lock:
                    x_est, u_eff = self.backend.step(sample, dt)

                pitch_deg = float(np.degrees(x_est[0]))
                roll_deg = float(np.degrees(x_est[1]))
                if abs(pitch_deg) >= self.args.pitch_estop_deg or abs(roll_deg) >= self.args.roll_estop_deg:
                    self.estop = True

                rw_norm, drive_norm = self._normalize_commands(u_eff)
                self._send_command(rw_norm, drive_norm)
                self._log_state(x_est, sample, rw_norm, drive_norm)
            except socket.timeout:
                self.timeout_packets += 1
                if self.args.zero_on_timeout:
                    self._send_command(0.0, 0.0)
                if self.last_rx_wall is not None:
                    if (time.perf_counter() - self.last_rx_wall) > self.args.comm_estop_s:
                        self.estop = True
            except json.JSONDecodeError:
                self.bad_json_packets += 1
            except Exception as exc:  # pragma: no cover - runtime guard
                self.runtime_errors += 1
                print(f"[error] {exc}")
                self._send_command(0.0, 0.0)

            elapsed = time.perf_counter() - loop_start
            sleep_s = self.loop_s - elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def start(self):
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def stop(self):
        self.running = False
        try:
            self._send_command(0.0, 0.0)
        except Exception:
            pass
        if hasattr(self, "control_thread"):
            self.control_thread.join(timeout=1.0)
        self.rx_sock.close()
        self.tx_sock.close()
        if self.csv_file is not None:
            self.csv_file.close()
        print(
            "Stopped: "
            f"missed={self.missed_packets} timeout={self.timeout_packets} "
            f"bad_json={self.bad_json_packets} runtime_errors={self.runtime_errors}"
        )


def run_with_plot(bridge: HILBridge):
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for --plot. Install with: pip install matplotlib")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("WiFi HIL Live Telemetry")
    fig.text(
        0.08,
        0.955,
        "Real-hardware parameter identification using your existing MuJoCo stack",
        fontsize=10,
        fontweight="bold",
    )
    ax0, ax1, ax2 = axes
    ax0.set_ylabel("angle (deg)")
    ax1.set_ylabel("command (norm)")
    ax2.set_ylabel("rw speed (deg/s)")
    ax2.set_xlabel("time (s)")

    line_pitch, = ax0.plot([], [], "b", label="pitch")
    line_roll, = ax0.plot([], [], "r", label="roll")
    line_rw, = ax1.plot([], [], "g", label="rw_cmd")
    line_drive, = ax1.plot([], [], "m", label="drive_cmd")
    line_rspd, = ax2.plot([], [], "orange", label="rw_speed")
    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper left", fontsize=8)

    ax0.set_ylim(-60, 60)
    ax1.set_ylim(-1.1, 1.1)
    ax2.set_ylim(-1200, 1200)

    def _format_status(values: dict[str, float]) -> str:
        fields = [
            ("rw_cmd_scale", "rw_scale"),
            ("drive_cmd_scale", "drive_scale"),
        ]
        if bridge.args.stub_control:
            fields.extend(
                [
                    ("stub_kp_pitch", "kp_p"),
                    ("stub_kd_pitch", "kd_p"),
                    ("stub_kp_roll", "kp_r"),
                    ("stub_kd_roll", "kd_r"),
                ]
            )
        else:
            fields.extend(
                [
                    ("base_pitch_kp", "bp_kp"),
                    ("base_pitch_kd", "bp_kd"),
                    ("base_roll_kp", "br_kp"),
                    ("base_roll_kd", "br_kd"),
                    ("wheel_momentum_k", "rw_mk"),
                    ("wheel_momentum_thresh_frac", "rw_thr"),
                    ("u_bleed", "u_bleed"),
                    ("estimator_q_scale", "kf_q"),
                    ("estimator_r_scale", "kf_r"),
                ]
            )
        chunks = []
        for key, label in fields:
            if key in values:
                chunks.append(f"{label}={values[key]:.3f}")
        return "Live tuning: " + "  |  ".join(chunks)

    specs = bridge.slider_specs() if bridge.args.tuning_panel else []
    if specs and Slider is None:
        raise RuntimeError("matplotlib widgets are unavailable; disable with --no-tuning-panel")
    slider_objs: dict[str, Slider] = {}
    status_text = None
    estop_text = None
    reset_button = None
    estop_button = None
    if specs:
        rows = int(np.ceil(len(specs) / 2.0))
        bottom_space = float(np.clip(0.16 + rows * 0.055, 0.20, 0.58))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=bottom_space)
        status_y = max(0.02, bottom_space - 0.055)
        status_text = fig.text(0.08, status_y, "", fontsize=8)
        estop_text = fig.text(0.74, status_y, "ESTOP: 0", fontsize=9, color="green", fontweight="bold")

        y0 = bottom_space - 0.045
        for idx, spec in enumerate(specs):
            col = idx % 2
            row = idx // 2
            x = 0.08 + col * 0.46
            y = y0 - row * 0.052
            ax_slider = fig.add_axes([x, y, 0.38, 0.024], facecolor="#f6f6f6")
            initial = bridge.live_tuning_values().get(spec.key, spec.vinit)
            slider = Slider(
                ax=ax_slider,
                label=spec.label,
                valmin=spec.vmin,
                valmax=spec.vmax,
                valinit=float(initial),
                valstep=spec.step,
            )
            slider_objs[spec.key] = slider

            def _make_handler(key: str):
                def _handler(val):
                    values = bridge.apply_live_tuning({key: float(val)})
                    if status_text is not None:
                        status_text.set_text(_format_status(values))

                return _handler

            slider.on_changed(_make_handler(spec.key))

        ax_reset = fig.add_axes([0.08, 0.02, 0.12, 0.03])
        reset_button = Button(ax_reset, "Reset Sliders")

        def _on_reset(_event):
            for slider in slider_objs.values():
                slider.reset()
            values = bridge.live_tuning_values()
            if status_text is not None:
                status_text.set_text(_format_status(values))

        reset_button.on_clicked(_on_reset)

        ax_estop = fig.add_axes([0.22, 0.02, 0.12, 0.03])
        estop_button = Button(ax_estop, "Clear ESTOP")

        def _on_clear_estop(_event):
            bridge.clear_estop()

        estop_button.on_clicked(_on_clear_estop)

        if status_text is not None:
            status_text.set_text(_format_status(bridge.live_tuning_values()))
    else:
        plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10)

    def update(_):
        with bridge.log_lock:
            if len(bridge.log_time) < 2:
                return
            t = list(bridge.log_time)
            pitch = list(bridge.log_pitch_deg)
            roll = list(bridge.log_roll_deg)
            rw = list(bridge.log_reaction_norm)
            drive = list(bridge.log_drive_norm)
            rspd = list(bridge.log_reaction_speed_dps)

        line_pitch.set_data(t, pitch)
        line_roll.set_data(t, roll)
        line_rw.set_data(t, rw)
        line_drive.set_data(t, drive)
        line_rspd.set_data(t, rspd)
        t0 = max(0.0, t[-1] - bridge.args.plot_window_s)
        t1 = t[-1] + 0.2
        for ax in axes:
            ax.set_xlim(t0, t1)
        if estop_text is not None:
            estop_text.set_text(f"ESTOP: {int(bridge.estop)}")
            estop_text.set_color("red" if bridge.estop else "green")
        return line_pitch, line_roll, line_rw, line_drive, line_rspd

    _ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    bridge.start()
    try:
        plt.tight_layout()
        plt.show()
    finally:
        bridge.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESP32 WiFi HIL bridge (PC side).")
    parser.add_argument("--pc-ip", type=str, default="0.0.0.0", help="PC bind IP for UDP receive.")
    parser.add_argument("--pc-port", type=int, default=5005, help="PC UDP receive port.")
    parser.add_argument("--esp32-ip", type=str, required=True, help="ESP32 IP for command UDP send.")
    parser.add_argument("--esp32-port", type=int, default=5006, help="ESP32 UDP command port.")
    parser.add_argument("--loop-hz", type=float, default=250.0, help="Bridge loop target frequency.")
    parser.add_argument("--rx-timeout-ms", type=float, default=10.0, help="UDP receive timeout in ms.")
    parser.add_argument(
        "--stub-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PD stub controller instead of final runtime stack.",
    )
    parser.add_argument(
        "--runtime-args",
        type=str,
        default="--mode robust --hardware-safe --control-hz 250 --controller-family current",
        help="Argument string forwarded to final/runtime_config.py parser.",
    )
    parser.add_argument("--imu-alpha", type=float, default=0.98, help="Complementary filter alpha.")
    parser.add_argument("--rw-cmd-scale", type=float, default=None, help="Wheel command units per +1 normalized command.")
    parser.add_argument(
        "--drive-cmd-scale",
        type=float,
        default=None,
        help="Drive command units per +1 normalized command.",
    )
    parser.add_argument(
        "--drive-channel",
        choices=["bx", "by", "sum"],
        default="bx",
        help="Which runtime command channel maps to drive command.",
    )
    parser.add_argument("--reaction-sign", type=float, default=1.0, help="Sign multiplier for outgoing rw command.")
    parser.add_argument("--drive-sign", type=float, default=1.0, help="Sign multiplier for outgoing drive command.")
    parser.add_argument("--pitch-rate-sign", type=float, default=1.0, help="Sign multiplier for incoming gy->pitch_rate.")
    parser.add_argument("--roll-rate-sign", type=float, default=1.0, help="Sign multiplier for incoming gx->roll_rate.")
    parser.add_argument("--reaction-speed-sign", type=float, default=1.0, help="Sign multiplier for incoming reaction_speed.")
    parser.add_argument("--accel-x-sign", type=float, default=1.0, help="Sign multiplier for accelerometer ax.")
    parser.add_argument("--accel-y-sign", type=float, default=1.0, help="Sign multiplier for accelerometer ay.")
    parser.add_argument("--accel-z-sign", type=float, default=1.0, help="Sign multiplier for accelerometer az.")
    parser.add_argument("--pitch-estop-deg", type=float, default=45.0, help="Pitch estop threshold.")
    parser.add_argument("--roll-estop-deg", type=float, default=45.0, help="Roll estop threshold.")
    parser.add_argument("--comm-estop-s", type=float, default=0.25, help="No-packet duration to force estop.")
    parser.add_argument(
        "--zero-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send zero command when receive timeout occurs.",
    )
    parser.add_argument("--max-reaction-norm", type=float, default=1.0, help="Outgoing normalized rw command clamp.")
    parser.add_argument("--max-drive-norm", type=float, default=1.0, help="Outgoing normalized drive command clamp.")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live matplotlib plot.",
    )
    parser.add_argument(
        "--tuning-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show slider panel for live controller/estimator tuning.",
    )
    parser.add_argument("--plot-window-s", type=float, default=10.0, help="Live plot history window (seconds).")
    parser.add_argument("--csv-log", type=str, default=None, help="Optional CSV log output path.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    bridge = HILBridge(args)
    if args.plot:
        run_with_plot(bridge)
        return

    bridge.start()
    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
