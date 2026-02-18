# Web Side Quest: Self Balancing Bot

This side quest runs the **full Python controller stack from `final/final.py`** (headless)
and streams live state to a Three.js renderer in the browser.
You change payload mass, reset, and observe whether it stabilizes or fails from overload.

## Target Browser

- Brave (latest desktop)

## Run

From repo root:

```bash
python web/server.py --port 8090
```

Default backend mode is `robust` for stronger payload tolerance.
If you want the original smoother profile, run:

```bash
python web/server.py --port 8090 --mode smooth
```

Then open:

`http://localhost:8090/web/`

## Behavior

- `Payload Mass` slider sets a physical payload body on top of the stick (Python MuJoCo runtime).
- Failure triggers when either:
  - tilt angle exceeds threshold, or
  - COM stays outside support radius for consecutive steps.
- `Max Stable Mass` updates after a mass survives the stability window.
