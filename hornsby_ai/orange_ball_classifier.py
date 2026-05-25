"""Orange ping pong ball classifier for Hornsby perception.

The classifier combines a strong color/shape prior with a tiny prototype model
that can be updated from validation samples later. It is intentionally small so
it can run beside the YOLO loop on a Mac camera feed.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OrangeBallCandidate:
    label: str
    confidence: float
    box: list[float]
    source_model: str
    orange_score: float
    roundness: float
    validation_score: float


class OrangeBallClassifier:
    """Classifies orange, round blobs as ping pong balls."""

    def __init__(self, model_path: str | Path | None = None, threshold: float = 0.58) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.positive_centroid: np.ndarray | None = None
        self.negative_centroid: np.ndarray | None = None
        self.feature_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        if self.model_path and self.model_path.exists():
            self._load(self.model_path)

    def _load(self, model_path: Path) -> None:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        positive = payload.get("positive_centroid")
        negative = payload.get("negative_centroid")
        scale = payload.get("feature_scale")
        if positive:
            self.positive_centroid = np.array(positive, dtype=np.float32)
        if negative:
            self.negative_centroid = np.array(negative, dtype=np.float32)
        if scale:
            self.feature_scale = np.array(scale, dtype=np.float32)
        print(f"[+] Loaded orange ball prototype ML model from {model_path.name}!", file=sys.stderr, flush=True)

    def detect(self, frame: np.ndarray) -> list[OrangeBallCandidate]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Optimized HSV bounds based on the actual color profile of your orange ball under your room lighting.
        # This prevents filtering out the ball under dimmer, warmer, or shadow conditions.
        lower = np.array([3, 75, 50], dtype=np.uint8)
        upper = np.array([24, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        image_area = max(1, width * height)
        candidates: list[OrangeBallCandidate] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            # Require a minimum size to avoid tiny noise spots
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            radius = max(w, h) / 2.0
            if radius <= 0:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0 else min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
            aspect = min(w, h) / max(w, h)
            
            # STRICT SHAPE FILTERS: A ping pong ball is extremely round and square-bounded!
            if circularity < 0.70 or aspect < 0.70:
                continue

            roi_mask = mask[y : y + h, x : x + w]
            orange_ratio = float(cv2.countNonZero(roi_mask)) / max(1, w * h)
            size_ratio = min(1.0, area / image_area * 30.0)
            mean_hsv = cv2.mean(hsv[y : y + h, x : x + w], mask=roi_mask)
            saturation = mean_hsv[1] / 255.0

            features = np.array([orange_ratio, circularity, aspect, saturation, size_ratio], dtype=np.float32)
            heuristic_score = self._heuristic_score(features)
            validation_score = self._prototype_score(features)
            confidence = 0.70 * heuristic_score + 0.30 * validation_score

            # Enforce a slightly higher threshold for high confidence tracking
            if confidence < 0.60:
                continue

            candidates.append(
                OrangeBallCandidate(
                    label="orange_ping_pong_ball",
                    confidence=round(float(confidence), 4),
                    box=[float(x), float(y), float(x + w), float(y + h)],
                    source_model="orange_ball_classifier",
                    orange_score=round(float(orange_ratio), 4),
                    roundness=round(float((circularity + aspect) / 2.0), 4),
                    validation_score=round(float(validation_score), 4),
                )
            )

        # ISOLATION: Sort by confidence and return ONLY the single best target candidate
        # to prevent the robot from getting confused by multiple background noise reflections!
        if not candidates:
            return []
        
        best_candidate = max(candidates, key=lambda item: item.confidence)
        return [best_candidate]

    def _heuristic_score(self, features: np.ndarray) -> float:
        orange_ratio, circularity, aspect, saturation, size_ratio = features
        raw = (
            2.4 * orange_ratio
            + 1.6 * circularity
            + 1.3 * aspect
            + 0.9 * saturation
            + 0.5 * size_ratio
            - 3.1
        )
        return 1.0 / (1.0 + math.exp(-float(raw)))

    def _prototype_score(self, features: np.ndarray) -> float:
        if self.positive_centroid is None or self.negative_centroid is None:
            return self._heuristic_score(features)

        scaled = features / np.maximum(self.feature_scale, 1e-6)
        positive = self.positive_centroid / np.maximum(self.feature_scale, 1e-6)
        negative = self.negative_centroid / np.maximum(self.feature_scale, 1e-6)
        positive_distance = float(np.linalg.norm(scaled - positive))
        negative_distance = float(np.linalg.norm(scaled - negative))
        return 1.0 / (1.0 + math.exp(positive_distance - negative_distance))

    def _dedupe(self, candidates: list[OrangeBallCandidate]) -> list[OrangeBallCandidate]:
        kept: list[OrangeBallCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if all(_iou(candidate.box, existing.box) < 0.35 for existing in kept):
                kept.append(candidate)
        return kept[:5]


def train_from_manifest(manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Train prototype centroids from a JSONL validation manifest.

    Each line should contain image, box, and label where label is 1 for an
    orange ping pong ball and 0 for a hard negative.
    """

    manifest = Path(manifest_path)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    helper = OrangeBallClassifier()

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = cv2.imread(str(row["image"]))
        if image is None:
            raise ValueError(f"Could not read image: {row['image']}")
        x1, y1, x2, y2 = [int(value) for value in row["box"]]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        features = _features_for_crop(crop, helper)
        if int(row["label"]):
            positives.append(features)
        else:
            negatives.append(features)

    if not positives or not negatives:
        raise ValueError("Training needs at least one positive and one negative sample.")

    all_features = np.vstack([positives, negatives])
    payload = {
        "positive_centroid": np.mean(np.vstack(positives), axis=0).round(6).tolist(),
        "negative_centroid": np.mean(np.vstack(negatives), axis=0).round(6).tolist(),
        "feature_scale": np.maximum(np.std(all_features, axis=0), 0.05).round(6).tolist(),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _features_for_crop(crop: np.ndarray, helper: OrangeBallClassifier) -> np.ndarray:
    detections = helper.detect(crop)
    if detections:
        best = max(detections, key=lambda item: item.confidence)
        return np.array([best.orange_score, best.roundness, best.roundness, 1.0, best.confidence], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union
