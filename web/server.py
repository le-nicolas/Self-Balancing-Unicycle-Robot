from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from control_core import (  # noqa: E402
    apply_control_delay,
    apply_upright_postprocess,
    base_commands_with_limits,
    compute_control_command,
    reset_controller_buffers,
    wheel_command_with_limits,
)
from residual_model import ResidualPolicy  # noqa: E402
from runtime_config import build_config, parse_args as parse_runtime_args  # noqa: E402
from runtime_model import (  # noqa: E402
    build_kalman_gain,
    build_measurement_noise_cov,
    build_partial_measurement_matrix,
    compute_robot_com_distance_xy,
    enforce_wheel_only_constraints,
    estimator_measurement_update,
    get_true_state,
    lookup_model_ids,
    reset_state,
    set_payload_mass,
)


STABLE_CONFIRM_S = 4.0


class FullFinalRuntime:
    """Headless runtime that mirrors final/final.py control logic for web streaming."""

    def __init__(self, mode: str = "smooth", payload_mass_kg: float = 0.4):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.mode = mode
        self.requested_mass_kg = max(float(payload_mass_kg), 0.0)
        self.max_stable_mass_kg = 0.0
        self.stable_recorded = False
        self.status_text = "Booting..."
        self.failure_reason = ""
        self.failed = False
        self.paused = False
        self._build_runtime_locked(self.mode, self.requested_mass_kg)
        self._thread = threading.Thread(target=self._run_loop, name="full-final-runtime", daemon=True)
        self._thread.start()

    def _build_runtime_locked(self, mode: str, payload_mass_kg: float):
        args = parse_runtime_args(
            [
                "--mode",
                mode,
                "--payload-mass",
                str(max(payload_mass_kg, 0.0)),
                "--stop-on-crash",
            ]
        )
        cfg = build_config(args)
        residual_policy = ResidualPolicy(cfg)
        initial_roll_rad = float(np.radians(args.initial_y_tilt_deg))

        xml_path = FINAL_DIR / "final.xml"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        if cfg.smooth_viewer and cfg.easy_mode:
            model.opt.gravity[2] = -6.5

        ids = lookup_model_ids(model)
        payload_mass = set_payload_mass(model, data, ids, cfg.payload_mass_kg)
        reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)

        nx = model.nq + model.nv
        nu = model.nu
        a_full = np.zeros((nx, nx))
        b_full = np.zeros((nx, nu))
        mujoco.mjd_transitionFD(model, data, 1e-6, True, a_full, b_full, None, None)

        idx = [
            ids.q_pitch,
            ids.q_roll,
            model.nq + ids.v_pitch,
            model.nq + ids.v_roll,
            model.nq + ids.v_rw,
            ids.q_base_x,
            ids.q_base_y,
            model.nq + ids.v_base_x,
            model.nq + ids.v_base_y,
        ]
        a = a_full[np.ix_(idx, idx)]
        b = b_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
        nx_ctrl = a.shape[0]
        nu_ctrl = b.shape[1]

        a_aug = np.block([[a, b], [np.zeros((nu_ctrl, nx_ctrl)), np.eye(nu_ctrl)]])
        b_aug = np.vstack([b, np.eye(nu_ctrl)])
        q_aug = np.block([[cfg.qx, np.zeros((nx_ctrl, nu_ctrl))], [np.zeros((nu_ctrl, nx_ctrl)), cfg.qu]])
        p_aug = solve_discrete_are(a_aug, b_aug, q_aug, cfg.r_du)
        k_du = np.linalg.inv(b_aug.T @ p_aug @ b_aug + cfg.r_du) @ (b_aug.T @ p_aug @ a_aug)

        a_w = a[np.ix_([0, 2, 4], [0, 2, 4])]
        b_w = b[np.ix_([0, 2, 4], [0])]
        q_w = np.diag([260.0, 35.0, 0.6])
        r_w = np.array([[0.08]])
        p_w = solve_discrete_are(a_w, b_w, q_w, r_w)
        k_paper_pitch = np.linalg.inv(b_w.T @ p_w @ b_w + r_w) @ (b_w.T @ p_w @ a_w)
        k_wheel_only = k_paper_pitch.copy() if cfg.wheel_only else None

        control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
        control_dt = control_steps * model.opt.timestep
        wheel_lsb = (2.0 * np.pi) / (cfg.wheel_encoder_ticks_per_rev * control_dt)
        wheel_budget_speed = min(cfg.wheel_spin_budget_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_budget_abs_rad_s)
        wheel_hard_speed = min(cfg.wheel_spin_hard_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_hard_abs_rad_s)
        if not cfg.allow_base_motion:
            wheel_hard_speed *= 1.10

        c = build_partial_measurement_matrix(cfg)
        qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        rn = build_measurement_noise_cov(cfg, wheel_lsb)
        l_gain = build_kalman_gain(a, qn, c, rn)

        act_ids = np.array([ids.aid_rw, ids.aid_base_x, ids.aid_base_y], dtype=int)
        xml_ctrl_low = model.actuator_ctrlrange[act_ids, 0]
        xml_ctrl_high = model.actuator_ctrlrange[act_ids, 1]
        if np.any(xml_ctrl_low > -cfg.max_u) or np.any(xml_ctrl_high < cfg.max_u):
            raise ValueError("XML actuator ctrlrange does not satisfy runtime max_u limits.")

        queue_len = 1 if not cfg.hardware_realistic else cfg.control_delay_steps + 1
        (
            x_est,
            u_applied,
            u_eff_applied,
            base_int,
            wheel_pitch_int,
            base_ref,
            base_authority_state,
            u_base_smooth,
            balance_phase,
            recovery_time_s,
            high_spin_active,
            cmd_queue,
        ) = reset_controller_buffers(nx_ctrl, nu_ctrl, queue_len)

        self.mode = mode
        self.cfg = cfg
        self.runtime_args = args
        self.residual_policy = residual_policy
        self.model = model
        self.data = data
        self.ids = ids
        self.initial_roll_rad = initial_roll_rad
        self.payload_mass_kg = payload_mass
        self.requested_mass_kg = max(float(payload_mass_kg), 0.0)
        self.a = a
        self.b = b
        self.k_du = k_du
        self.k_wheel_only = k_wheel_only
        self.k_paper_pitch = k_paper_pitch
        self.control_steps = int(control_steps)
        self.control_dt = float(control_dt)
        self.wheel_lsb = float(wheel_lsb)
        self.wheel_budget_speed = float(wheel_budget_speed)
        self.wheel_hard_speed = float(wheel_hard_speed)
        self.c = c
        self.l_gain = l_gain
        self.xml_ctrl_low = xml_ctrl_low
        self.xml_ctrl_high = xml_ctrl_high
        self.rng = np.random.default_rng(cfg.seed)
        self.x_est = x_est
        self.u_applied = u_applied
        self.u_eff_applied = u_eff_applied
        self.base_int = base_int
        self.wheel_pitch_int = wheel_pitch_int
        self.base_ref = base_ref
        self.base_authority_state = base_authority_state
        self.u_base_smooth = u_base_smooth
        self.balance_phase = balance_phase
        self.recovery_time_s = recovery_time_s
        self.high_spin_active = high_spin_active
        self.cmd_queue = cmd_queue
        self.upright_blend = 0.0
        self.despin_gain = 0.25

        self.sat_hits = np.zeros(nu_ctrl, dtype=int)
        self.du_hits = np.zeros(nu_ctrl, dtype=int)
        self.xml_limit_margin_hits = np.zeros(nu_ctrl, dtype=int)
        self.speed_limit_hits = np.zeros(5, dtype=int)
        self.derate_hits = np.zeros(3, dtype=int)
        self.step_count = 0
        self.control_updates = 0
        self.crash_count = 0
        self.max_pitch = 0.0
        self.max_roll = 0.0
        self.phase_switch_count = 0
        self.hold_steps = 0
        self.wheel_over_budget_count = 0
        self.wheel_over_hard_count = 0
        self.high_spin_steps = 0
        self.wheel_speed_abs_sum = 0.0
        self.wheel_speed_abs_max = 0.0
        self.wheel_speed_samples = 0
        self.wheel_hard_same_dir_request_count = 0
        self.wheel_hard_safe_output_count = 0
        self.residual_applied_count = 0
        self.residual_clipped_count = 0
        self.residual_gate_blocked_count = 0
        self.residual_max_abs = np.zeros(nu_ctrl, dtype=float)
        self.com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
        self.max_com_planar_dist = self.com_planar_dist
        self.com_over_support_steps = 0
        self.com_fail_streak = 0
        self.com_overload_failures = 0
        self.failed = False
        self.failure_reason = ""
        self.paused = False
        self.stable_recorded = False
        self.status_text = "Running"

    def _reset_after_instability_locked(self):
        reset_state(
            self.model,
            self.data,
            self.ids.q_pitch,
            self.ids.q_roll,
            pitch_eq=0.0,
            roll_eq=self.initial_roll_rad,
        )
        self.com_fail_streak = 0
        queue_len = 1 if not self.cfg.hardware_realistic else self.cfg.control_delay_steps + 1
        (
            self.x_est,
            self.u_applied,
            self.u_eff_applied,
            self.base_int,
            self.wheel_pitch_int,
            self.base_ref,
            self.base_authority_state,
            self.u_base_smooth,
            self.balance_phase,
            self.recovery_time_s,
            self.high_spin_active,
            self.cmd_queue,
        ) = reset_controller_buffers(self.a.shape[0], self.b.shape[1], queue_len)
        self.upright_blend = 0.0

    def _step_once_locked(self):
        self.step_count += 1
        if self.cfg.wheel_only:
            enforce_wheel_only_constraints(self.model, self.data, self.ids)

        x_true = get_true_state(self.data, self.ids)
        self.com_planar_dist = compute_robot_com_distance_xy(self.model, self.data, self.ids.base_y_body_id)
        self.max_com_planar_dist = max(self.max_com_planar_dist, self.com_planar_dist)

        if not np.all(np.isfinite(x_true)):
            self._reset_after_instability_locked()
            self.status_text = "Reset after numerical instability"
            return

        x_pred = self.a @ self.x_est + self.b @ self.u_eff_applied
        self.x_est = x_pred

        if self.step_count % self.control_steps == 0:
            self.control_updates += 1
            self.x_est = estimator_measurement_update(
                self.cfg,
                x_true,
                x_pred,
                self.c,
                self.l_gain,
                self.rng,
                self.wheel_lsb,
            )

            angle_mag = max(abs(float(x_true[0])), abs(float(x_true[1])))
            rate_mag = max(abs(float(x_true[2])), abs(float(x_true[3])))
            prev_phase = self.balance_phase
            if self.balance_phase == "recovery":
                if angle_mag < self.cfg.hold_enter_angle_rad and rate_mag < self.cfg.hold_enter_rate_rad_s:
                    self.balance_phase = "hold"
                    self.recovery_time_s = 0.0
            else:
                if angle_mag > self.cfg.hold_exit_angle_rad or rate_mag > self.cfg.hold_exit_rate_rad_s:
                    self.balance_phase = "recovery"
                    self.recovery_time_s = 0.0
            if self.balance_phase != prev_phase:
                self.phase_switch_count += 1
            if self.balance_phase == "recovery":
                self.recovery_time_s += self.control_dt

            (
                u_cmd,
                self.base_int,
                self.base_ref,
                self.base_authority_state,
                self.u_base_smooth,
                self.wheel_pitch_int,
                rw_u_limit,
                wheel_over_budget,
                wheel_over_hard,
                self.high_spin_active,
                _control_terms,
            ) = compute_control_command(
                cfg=self.cfg,
                x_est=self.x_est,
                x_true=x_true,
                u_eff_applied=self.u_eff_applied,
                base_int=self.base_int,
                base_ref=self.base_ref,
                base_authority_state=self.base_authority_state,
                u_base_smooth=self.u_base_smooth,
                wheel_pitch_int=self.wheel_pitch_int,
                balance_phase=self.balance_phase,
                recovery_time_s=self.recovery_time_s,
                high_spin_active=self.high_spin_active,
                control_dt=self.control_dt,
                K_du=self.k_du,
                K_wheel_only=self.k_wheel_only,
                K_paper_pitch=self.k_paper_pitch,
                du_hits=self.du_hits,
                sat_hits=self.sat_hits,
            )

            residual_step = self.residual_policy.step(
                x_est=self.x_est,
                x_true=x_true,
                u_nominal=u_cmd,
                u_eff_applied=self.u_eff_applied,
            )
            residual_delta = residual_step.delta_u
            self.residual_gate_blocked_count += int(residual_step.gate_blocked)
            if residual_step.applied:
                self.residual_applied_count += 1
                self.residual_clipped_count += int(residual_step.clipped)
                self.residual_max_abs = np.maximum(self.residual_max_abs, np.abs(residual_delta))
                u_cmd = u_cmd + residual_delta

            self.wheel_over_budget_count += int(wheel_over_budget)
            self.wheel_over_hard_count += int(wheel_over_hard)

            u_cmd, self.upright_blend = apply_upright_postprocess(
                cfg=self.cfg,
                u_cmd=u_cmd,
                x_est=self.x_est,
                x_true=x_true,
                upright_blend=self.upright_blend,
                balance_phase=self.balance_phase,
                high_spin_active=self.high_spin_active,
                despin_gain=self.despin_gain,
                rw_u_limit=rw_u_limit,
            )

            self.xml_limit_margin_hits += ((u_cmd < self.xml_ctrl_low) | (u_cmd > self.xml_ctrl_high)).astype(int)
            self.u_applied = apply_control_delay(self.cfg, self.cmd_queue, u_cmd)

        if self.balance_phase == "hold":
            self.hold_steps += 1
        if self.high_spin_active:
            self.high_spin_steps += 1

        self.data.ctrl[:] = 0.0
        wheel_speed = float(self.data.qvel[self.ids.v_rw])
        if abs(wheel_speed) >= self.wheel_hard_speed and abs(float(self.u_applied[0])) > 1e-9:
            if np.sign(float(self.u_applied[0])) == np.sign(wheel_speed):
                self.wheel_hard_same_dir_request_count += 1

        wheel_cmd = wheel_command_with_limits(self.cfg, wheel_speed, float(self.u_applied[0]))
        if abs(wheel_speed) >= self.wheel_hard_speed:
            if abs(wheel_cmd) <= 1e-9 or (abs(wheel_speed) > 1e-9 and np.sign(wheel_cmd) != np.sign(wheel_speed)):
                self.wheel_hard_safe_output_count += 1
        self.derate_hits[0] += int(abs(wheel_cmd - self.u_applied[0]) > 1e-9)
        self.data.ctrl[self.ids.aid_rw] = wheel_cmd

        if self.cfg.allow_base_motion:
            base_x_cmd, base_y_cmd = base_commands_with_limits(
                cfg=self.cfg,
                base_x_speed=float(self.data.qvel[self.ids.v_base_x]),
                base_y_speed=float(self.data.qvel[self.ids.v_base_y]),
                base_x=float(self.data.qpos[self.ids.q_base_x]),
                base_y=float(self.data.qpos[self.ids.q_base_y]),
                base_x_request=float(self.u_applied[1]),
                base_y_request=float(self.u_applied[2]),
            )
        else:
            base_x_cmd = 0.0
            base_y_cmd = 0.0
        self.derate_hits[1] += int(abs(base_x_cmd - self.u_applied[1]) > 1e-9)
        self.derate_hits[2] += int(abs(base_y_cmd - self.u_applied[2]) > 1e-9)
        self.data.ctrl[self.ids.aid_base_x] = base_x_cmd
        self.data.ctrl[self.ids.aid_base_y] = base_y_cmd
        self.u_eff_applied[:] = [wheel_cmd, base_x_cmd, base_y_cmd]

        mujoco.mj_step(self.model, self.data)
        if self.cfg.wheel_only:
            enforce_wheel_only_constraints(self.model, self.data, self.ids)

        wheel_speed_after = float(self.data.qvel[self.ids.v_rw])
        wheel_speed_abs_after = abs(wheel_speed_after)
        self.wheel_speed_abs_sum += wheel_speed_abs_after
        self.wheel_speed_abs_max = max(self.wheel_speed_abs_max, wheel_speed_abs_after)
        self.wheel_speed_samples += 1

        self.speed_limit_hits[0] += int(abs(self.data.qvel[self.ids.v_rw]) > self.cfg.max_wheel_speed_rad_s)
        self.speed_limit_hits[1] += int(abs(self.data.qvel[self.ids.v_pitch]) > self.cfg.max_pitch_roll_rate_rad_s)
        self.speed_limit_hits[2] += int(abs(self.data.qvel[self.ids.v_roll]) > self.cfg.max_pitch_roll_rate_rad_s)
        self.speed_limit_hits[3] += int(abs(self.data.qvel[self.ids.v_base_x]) > self.cfg.max_base_speed_m_s)
        self.speed_limit_hits[4] += int(abs(self.data.qvel[self.ids.v_base_y]) > self.cfg.max_base_speed_m_s)

        self.com_planar_dist = compute_robot_com_distance_xy(self.model, self.data, self.ids.base_y_body_id)
        self.max_com_planar_dist = max(self.max_com_planar_dist, self.com_planar_dist)
        if self.com_planar_dist > self.cfg.payload_support_radius_m:
            self.com_over_support_steps += 1
            self.com_fail_streak += 1
        else:
            self.com_fail_streak = 0

        pitch = float(self.data.qpos[self.ids.q_pitch])
        roll = float(self.data.qpos[self.ids.q_roll])
        self.max_pitch = max(self.max_pitch, abs(pitch))
        self.max_roll = max(self.max_roll, abs(roll))

        tilt_failed = abs(pitch) >= self.cfg.crash_angle_rad or abs(roll) >= self.cfg.crash_angle_rad
        com_failed = self.com_fail_streak >= self.cfg.payload_com_fail_steps
        if tilt_failed or com_failed:
            self.crash_count += 1
            self.com_overload_failures += int(com_failed)
            self.failure_reason = "COM overload" if com_failed else "Tilt overload"
            self.failed = True
            self.status_text = f"Failed: {self.failure_reason}"
            if self.cfg.stop_on_crash:
                self.paused = True
            else:
                self._reset_after_instability_locked()
                self.failed = False
                self.failure_reason = ""
                self.status_text = "Running"
            return

        if (not self.stable_recorded) and float(self.data.time) >= STABLE_CONFIRM_S:
            self.stable_recorded = True
            self.max_stable_mass_kg = max(self.max_stable_mass_kg, self.requested_mass_kg)
            self.status_text = "Stable"
        elif not self.paused:
            self.status_text = "Running"

    def _run_loop(self):
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            with self._lock:
                timestep = float(self.model.opt.timestep) if self.model is not None else 0.001
                if not self.paused and not self.failed:
                    self._step_once_locked()
                elif self.failed:
                    self.status_text = f"Failed: {self.failure_reason}" if self.failure_reason else "Failed"
                else:
                    self.status_text = "Paused"
            next_tick += timestep
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.010))
            else:
                next_tick = time.perf_counter()

    def set_paused(self, paused: bool):
        with self._lock:
            self.paused = bool(paused)
            if self.failed:
                self.status_text = f"Failed: {self.failure_reason}" if self.failure_reason else "Failed"
            elif self.paused:
                self.status_text = "Paused"
            else:
                self.status_text = "Running"

    def reset(self, payload_mass_kg: float | None = None, mode: str | None = None):
        with self._lock:
            next_mode = self.mode if mode is None else str(mode)
            next_mass = self.requested_mass_kg if payload_mass_kg is None else max(float(payload_mass_kg), 0.0)
            self._build_runtime_locked(next_mode, next_mass)
            self.failed = False
            self.failure_reason = ""
            self.paused = False
            self.status_text = "Running"
            return self._state_locked()

    def _state_locked(self) -> dict:
        geom_xpos = np.asarray(self.data.geom_xpos, dtype=float).reshape(-1).tolist()
        geom_xmat = np.asarray(self.data.geom_xmat, dtype=float).reshape(-1).tolist()
        return {
            "ok": True,
            "status": self.status_text,
            "mode": self.mode,
            "paused": bool(self.paused),
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
            "elapsed_s": float(self.data.time),
            "requested_mass_kg": float(self.requested_mass_kg),
            "effective_mass_kg": float(self.payload_mass_kg),
            "max_stable_mass_kg": float(self.max_stable_mass_kg),
            "com_dist_m": float(self.com_planar_dist),
            "pitch_rad": float(self.data.qpos[self.ids.q_pitch]),
            "roll_rad": float(self.data.qpos[self.ids.q_roll]),
            "step_count": int(self.step_count),
            "control_updates": int(self.control_updates),
            "ngeom": int(self.model.ngeom),
            "geom_xpos": geom_xpos,
            "geom_xmat": geom_xmat,
        }

    def get_state(self) -> dict:
        with self._lock:
            return self._state_locked()

    def shutdown(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)


class AppServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, runtime: FullFinalRuntime):
        super().__init__(server_address, handler_cls)
        self.runtime = runtime


class ApiStaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        sys.stdout.write(f"{self.log_date_time_string()} - {fmt % args}\n")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._write_json({"ok": True, "service": "full-final-runtime"})
            return
        if path == "/api/state":
            self._write_json(self.server.runtime.get_state())
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/reset":
            payload = self._read_json()
            mass = payload.get("payload_mass_kg", None)
            mode = payload.get("mode", None)
            try:
                state = self.server.runtime.reset(payload_mass_kg=mass, mode=mode)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._write_json(state)
            return
        if path == "/api/pause":
            payload = self._read_json()
            paused_value = payload.get("paused", None)
            if paused_value is None:
                current = self.server.runtime.get_state().get("paused", False)
                paused_value = not bool(current)
            self.server.runtime.set_paused(bool(paused_value))
            self._write_json(self.server.runtime.get_state())
            return
        self._write_json({"ok": False, "error": "Unknown API endpoint"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="Full final.py runtime web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--mode", choices=["smooth", "robust"], default="robust")
    parser.add_argument("--payload-mass", type=float, default=0.4)
    args = parser.parse_args()

    runtime = FullFinalRuntime(mode=args.mode, payload_mass_kg=args.payload_mass)
    server = AppServer((args.host, args.port), ApiStaticHandler, runtime)
    print(f"Serving on http://{args.host}:{args.port}/web/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.shutdown()


if __name__ == "__main__":
    main()
