# Legacy Script Index

These scripts are archived iteration artifacts and are not part of the maintained control stack.

Canonical maintained path:
- Runtime and controller stack: `final/final.py`
- Benchmark/evaluation: `final/benchmark.py`, `final/controller_eval.py`

## Script map

| Script | Primary experiment | Model dependency | Status |
|---|---|---|---|
| `reactionwheel_invertedpendulum.py` | Earliest open-loop fall validation for reaction-wheel pendulum dynamics | `archive/legacy/models/reactionwheel_invertedpendulum.xml` | Archived |
| `wheel.py` | Early wheel-only PID stabilizer | `archive/legacy/models/wheel_on_stick.xml` | Archived |
| `wheel_lqr.py` | Wheel-only state-feedback (LQR-style) trial | `archive/legacy/models/wheel_on_stick.xml` | Archived |
| `wheel_and_base.py` | Wheel+base dual-actuator stabilization prototype with wheel-speed derating | `archive/legacy/models/reactionwheel_basemotor.xml` | Archived |
| `run.py` | Minimal wheel-only PD baseline with conservative limits | `archive/legacy/models/wheel_on_stick.xml` | Archived |
| `test.py` | Cart-pole style PD sanity test | `archive/legacy/models/test.xml` | Archived |
| `fina_cop.py` | Transitional script moving from prototype models toward `final/final.xml` + LQR/Kalman stack | `final/final.xml` | Archived |
| `furuta_controlv2.py` | Manual keyboard-input Furuta pendulum control experiment | `archive/legacy/models/furuta_pendulum.xml` | Archived |
| `furuta_control.py` | Furuta swing-up + PD stabilization experiment | `archive/legacy/models/furuta_pendulum2.xml` | Archived |
| `furuta_control_lqr.py` | Furuta hybrid swing-up + near-upright LQR capture trial | `archive/legacy/models/furuta_pendulum2.xml` | Archived |

Use these only for historical comparison or idea mining. New work should target `final/`.
