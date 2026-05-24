# Physical AI Pipeline

Hornsby is being built around a simple physical AI loop:

```text
camera or image
  -> YOLO26 MLX perception
  -> orange ping pong ball classifier
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

## Orange Ping Pong Ball Classification

Hornsby includes a lightweight classifier for the first physical target: an orange ping pong ball. It runs beside YOLO26 and emits detections labeled `orange_ping_pong_ball`. The classifier looks for orange HSV regions, validates roundness and fill ratio, then blends those features with an optional learned prototype model.

Camera mode enables it by default:

```bash
python hornsby_ai/physical_ai.py --source 0 --show
```

Useful tuning options:

```bash
python hornsby_ai/physical_ai.py --source 0 --orange-threshold 0.62
python hornsby_ai/physical_ai.py --source 0 --no-orange-ball
```

To improve validation accuracy, collect positive and hard-negative examples in a JSONL manifest:

```json
{"image":"data/orange-ball/frame-001.jpg","box":[210,140,268,198],"label":1}
{"image":"data/orange-ball/frame-002.jpg","box":[80,92,160,170],"label":0}
```

Then train the prototype model:

```bash
scripts/train_orange_ball_classifier.py data/orange-ball/manifest.jsonl
```

The generated `models/orange_ping_pong_classifier.json` is loaded automatically by the perception script when present.

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
