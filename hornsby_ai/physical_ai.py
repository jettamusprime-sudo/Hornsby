#!/usr/bin/env python3
"""YOLO26 MLX perception entry point for Hornsby.

This module keeps the hackathon path short:

  python hornsby_ai/physical_ai.py --source image.jpg
  python hornsby_ai/physical_ai.py --source 0 --show

It uses YOLO26 through MLX on Apple Silicon and emits simple perception
events that can later be connected to motor control and robot behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[float]


@dataclass
class PerceptionFrame:
    source: str
    width: int
    height: int
    timestamp: float
    detections: list[Detection]

    @property
    def has_target(self) -> bool:
        return bool(self.detections)

    @property
    def best_target(self) -> Detection | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda item: item.confidence)


def load_model(model_path: Path) -> Any:
    try:
        from yolo26mlx import YOLO
    except ImportError as exc:
        raise SystemExit(
            "YOLO26 MLX is not installed. Run: scripts/setup_yolo26_mlx_macos.sh"
        ) from exc

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    return YOLO(str(model_path))


def normalize_result(result: Any) -> list[Detection]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    detections: list[Detection] = []

    if boxes is None:
        return detections

    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)

    if xyxy is None or conf is None or cls is None:
        return detections

    for box, score, class_id in zip(xyxy, conf, cls):
        class_index = int(class_id)
        label = names.get(class_index, str(class_index)) if isinstance(names, dict) else str(class_index)
        detections.append(
            Detection(
                label=label,
                confidence=float(score),
                box=[float(value) for value in box],
            )
        )

    return detections


def emit(perception: PerceptionFrame) -> None:
    print(json.dumps(asdict(perception), ensure_ascii=False), flush=True)


def predict_image(model: Any, source: str, confidence: float, save: bool) -> PerceptionFrame:
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size

    results = model.predict(source, conf=confidence, save=save)
    detections = normalize_result(results[0]) if results else []
    return PerceptionFrame(
        source=source,
        width=width,
        height=height,
        timestamp=time.time(),
        detections=detections,
    )


def predict_camera(
    model: Any,
    camera_index: int,
    confidence: float,
    show: bool,
    frame_interval: float,
) -> None:
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera: {camera_index}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(frame, conf=confidence)
            detections = normalize_result(results[0]) if results else []
            height, width = frame.shape[:2]
            perception = PerceptionFrame(
                source=str(camera_index),
                width=width,
                height=height,
                timestamp=time.time(),
                detections=detections,
            )
            emit(perception)

            if show:
                annotated = results[0].plot() if results else frame
                cv2.imshow("Hornsby Physical AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_interval > 0:
                time.sleep(frame_interval)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hornsby YOLO26 MLX physical AI")
    parser.add_argument("--model", default="models/yolo26n.npz", help="Path to converted YOLO26 MLX model")
    parser.add_argument("--source", required=True, help="Image path, directory path, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated image/video to results/")
    parser.add_argument("--show", action="store_true", help="Show live camera window")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.15,
        help="Seconds to wait between camera inference frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(Path(args.model))

    if args.source.isdigit():
        predict_camera(model, int(args.source), args.conf, args.show, args.frame_interval)
        return

    perception = predict_image(model, args.source, args.conf, args.save)
    emit(perception)

    if perception.best_target:
        target = perception.best_target
        print(f"Hornsby sees {target.label} with {target.confidence:.2f} confidence.", file=sys.stderr)
    else:
        print("Hornsby does not see a target yet.", file=sys.stderr)


if __name__ == "__main__":
    main()
