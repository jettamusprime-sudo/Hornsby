# Physical AI Pipeline

Hornsby is being built around a simple physical AI loop:

```text
camera or image
  -> YOLO26 MLX perception
  -> structured detection events
  -> robot decision logic
  -> motor and sensor actions
  -> new world state
```

## Why YOLO26 MLX

YOLO26 MLX is useful for this project because it runs locally on Apple Silicon through MLX and Metal acceleration. That makes it practical for a Mac-based hackathon demo without needing a cloud GPU.

The current default model is `yolo26n`, the small model, because startup speed and responsiveness matter more than maximum accuracy for the first prototype.

## Local Files

Generated local files are intentionally ignored by Git:

- `.venv/`
- `models/yolo26n.pt`
- `models/yolo26n.npz`
- `third_party/yolo-mlx/`
- `results/`

Only the integration code and setup script are versioned.

## Perception Event

The perception script emits JSON so the rest of the robot can consume it without depending on the YOLO internals.

Example event:

```json
{
  "source": "0",
  "detections": [
    {
      "label": "cup",
      "confidence": 0.76,
      "box": [318.0, 102.0, 410.0, 280.0]
    }
  ]
}
```

## Suggested Robot Behaviors

Early behaviors should stay simple:

- stop when a person is detected;
- turn toward the largest target;
- follow a selected class;
- log objects seen during a run;
- use confidence thresholds to avoid unstable behavior;
- display perception status in a control app.

## Safety Notes

Before connecting detections to motors:

- keep motor speed low;
- add an emergency stop;
- require a visible armed/disarmed state;
- avoid autonomous movement near people until tested;
- log decisions and detections for debugging.

## Next Milestones

1. Add a selected target class.
2. Add camera frame rate and inference latency reporting.
3. Add a robot state event: `idle`, `tracking`, `lost_target`, `stopped`.
4. Add a motor command interface.
5. Add a local dashboard.
