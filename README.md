# Mujoco-series

MuJoCo experiments for balancing and reaction-wheel control, including:
- simulation controllers and model variants
- benchmarking scripts
- firmware-oriented parameter export under `final/firmware/`

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

- Generated benchmark outputs in `final/results/` are ignored by Git.
- This repo currently does not include a software license file.
