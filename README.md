# Mujoco-series

MuJoCo experiments for balancing and reaction-wheel control, including:
- simulation controllers and model variants
- benchmarking scripts
- firmware-oriented parameter export under `final/firmware/`

## Start Here

The main showcase is the final controller stack in `final/`.

If you want a plain-English walkthrough of what this robot is, why it balances, how the control law works, and how to run it on your own PC:
- read `final/README.md`

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main controller:

```bash
python final/final.py --mode smooth
```

## Useful Commands

Run exporter parity test:

```bash
python final/test_export_parity.py
```

Export firmware parameters:

```bash
python final/export_firmware_params.py --mode smooth
```

## Notes

- Benchmark result images (`final/results/*.png`) are tracked; raw CSV/TXT benchmark dumps are ignored.
