# WiFi HIL Bridge (ESP32 <-> PC)

This setup runs your controller on PC and keeps ESP32 as an I/O bridge:

`PC Python stack -> UDP/WiFi -> ESP32 -> motors/sensors -> UDP/WiFi -> PC`

## Files

- `final/hil_bridge.py`: PC-side bridge using your runtime stack (`runtime_model` + `control_core`) or stub mode.
- `final/esp32_hil_bridge.ino`: ESP32 firmware with BMI088 + AS5600 + SimpleFOC (reaction wheel) + BTS7960 (drive).

## 1) Flash ESP32

1. Open `final/esp32_hil_bridge.ino`.
2. Set:
   - `WIFI_SSID`, `WIFI_PASSWORD`
   - `PC_IP`
3. Verify motor constants:
   - `RW_POLE_PAIRS`
   - `RW_SUPPLY_VOLTAGE`
   - `RW_MAX_VOLTAGE`
   - `RW_CMD_TO_VOLT_GAIN`
   - `RW_INVERT`
4. Upload and confirm serial output shows connected IP.

## 2) PC bring-up in stub mode first

```powershell
python final/hil_bridge.py --esp32-ip 192.168.1.50 --stub-control --plot
```

Expected:
- Live plot updates.
- Console updates every second.
- `missed`/`timeout` counters stay low.

## 3) Switch to your real runtime stack

```powershell
python final/hil_bridge.py --esp32-ip 192.168.1.50 --plot
```

Default runtime args are conservative:

`--mode robust --hardware-safe --control-hz 250 --controller-family current`

Override when needed:

```powershell
python final/hil_bridge.py --esp32-ip 192.168.1.50 --runtime-args "--mode robust --controller-family current_dob --dob-cutoff-hz 5.0 --control-hz 250"
```

The live plot now includes a tuning panel (sliders) by default.
This is the key point to highlight:

- You are doing parameter identification on real hardware using your existing stack.
- Controller and estimator parameters are adjusted live while the same runtime pipeline is running.

## 4) Command scaling and sign alignment

If motion direction is inverted or weak/strong, tune these first:

- PC side:
  - `--reaction-sign`
  - `--drive-sign`
  - `--roll-rate-sign`
  - `--pitch-rate-sign`
  - `--rw-cmd-scale`
  - `--drive-cmd-scale`
- ESP side:
  - `RW_INVERT`
  - `RW_CMD_TO_VOLT_GAIN`

## 5) SimpleFOC torque calibration sequence

1. Keep robot physically restrained/off-ground.
2. Start with low `RW_MAX_VOLTAGE` (for example `2.0`) and low `RW_CMD_TO_VOLT_GAIN` (for example `1.5`).
3. Verify command polarity with tiny commands.
4. Increase `RW_CMD_TO_VOLT_GAIN` gradually until torque response is sufficient.
5. Only then raise `RW_MAX_VOLTAGE` and runtime command limits.

## Safety notes

- `hil_bridge.py` has software ESTOP on tilt (`--pitch-estop-deg`, `--roll-estop-deg`).
- ESP32 zeroes command on communication timeout (`CMD_TIMEOUT_US`).
- Start conservative, then expand limits in small steps.
