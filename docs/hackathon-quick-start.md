# Hackathon Quick Start

This guide is the fastest path to show Hornsby doing local physical AI on a Mac.

## Goal

Run a YOLO26 MLX perception loop locally on Apple Silicon and produce structured detections that can later drive robot behavior.

## Requirements

- Apple Silicon Mac;
- macOS;
- Python 3;
- camera access if using live video;
- internet connection for first setup.

## Setup

From the repository root:

```bash
scripts/setup_yolo26_mlx_macos.sh
source .venv/bin/activate
```

The script:

- creates `.venv`;
- installs MLX, OpenCV, Pillow, NumPy, and YOLO26 MLX;
- downloads `yolo26n.pt`;
- converts it to `models/yolo26n.npz`;
- keeps downloaded models out of Git.

## Run With Camera

```bash
python hornsby_ai/physical_ai.py --source 0 --show
```

Press `q` in the preview window to stop.

## Run With An Image

```bash
python hornsby_ai/physical_ai.py --source path/to/image.jpg --save
```

The output is JSON:

```json
{
  "source": "path/to/image.jpg",
  "detections": [
    {
      "label": "person",
      "confidence": 0.82,
      "box": [120.0, 44.0, 310.0, 420.0]
    }
  ]
}
```

## Hackathon Demo Flow

1. Show the physical robot concept and 3D files.
2. Run the camera perception loop.
3. Put objects in front of the camera.
4. Show detections becoming structured perception events.
5. Explain that the next step is motor behavior driven by those events.

## Current Scope

This is a perception starter, not a full autonomous robot yet.

The current stack can see. The next milestones are deciding and acting.
