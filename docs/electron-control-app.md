# Electron Control App Concept

An Electron app makes sense for Hornsby, especially for demos and hackathons.

The app should be a local control panel, not the core robotics runtime.

## Recommendation

Use Electron for:

- camera preview;
- start/stop perception;
- model setup status;
- live detection list;
- robot mode selection;
- logs;
- emergency stop;
- manual motor controls;
- calibration tools.

Keep Python/MLX as a separate worker process for inference.

This separation keeps the UI simple and lets the AI code run in the environment where MLX works best.

## Proposed Architecture

```text
Electron UI
  -> starts/stops Python perception worker
  -> receives JSON lines over stdout or WebSocket
  -> renders detections and robot state
  -> sends high-level commands to hardware bridge

Python MLX worker
  -> loads YOLO26 model
  -> reads camera/image/video
  -> emits perception events

Hardware bridge
  -> receives safe movement commands
  -> controls motors and sensors
```

## First App Screens

The first version should have only what is needed:

- `Perception`: camera preview, detections, FPS, confidence threshold.
- `Robot`: armed/disarmed state, manual movement buttons, emergency stop.
- `Setup`: model status, environment checks, camera selection.
- `Logs`: perception events and errors.

## Why Not Put Everything In Electron

Electron is good for UI, but not ideal as the only robotics runtime.

Real-time control, MLX inference, and hardware interfaces should stay in small backend processes. The Electron app should coordinate them and make the system easier to use.

## Minimal First Implementation

1. Create an Electron shell.
2. Add a start button that runs:

```bash
python hornsby_ai/physical_ai.py --source 0
```

3. Read JSON lines from stdout.
4. Show detections in the UI.
5. Add a stop button.
6. Add an emergency stop placeholder before motor control exists.

## Later

- WebSocket bridge instead of stdout;
- annotated video preview;
- recording mode;
- dataset capture;
- object-following mode;
- motor calibration;
- telemetry timeline;
- integration with the robot firmware.
