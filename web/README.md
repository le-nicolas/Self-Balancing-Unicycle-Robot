# Web Side Quest: Self Balancing Bot

This side quest runs a balancing bot in **MuJoCo WebAssembly** with **Three.js** rendering.
You change payload mass, reset, and observe whether it stabilizes or fails from overload.

## Target Browser

- Brave (latest desktop)

## Run

From repo root:

```bash
python -m http.server 8080
```

Then open:

`http://localhost:8080/web/`

## Behavior

- `Payload Mass` slider sets a physical payload body on top of the stick.
- Failure triggers when either:
  - tilt angle exceeds threshold, or
  - COM stays outside support radius for consecutive steps.
- `Max Stable Mass` updates after a mass survives the stability window.
