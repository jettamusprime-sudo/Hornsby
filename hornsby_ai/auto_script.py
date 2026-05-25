#!/usr/bin/env python3
"""Autonomous 3rd-Person tracking script for Hornsby.

This script runs a complete self-contained physical AI loop:
1. Streams frames continuously from the camera.
2. Performs YOLO26 MLX classification on the scene.
3. Performs Orange Ball classification to track the target ping pong ball.
4. Emits structured JSON events on stdout for Electron to draw bounding boxes.
5. Prints autonomous driving steering decisions to stderr for the Electron logs panel.
"""

from __future__ import annotations

import sys
import time
import os
import uuid
import json
import signal
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np

# Ensure root folder is in path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hornsby_ai.orange_ball_classifier import OrangeBallClassifier
from hornsby_ai.physical_ai import load_model, normalize_result, PerceptionFrame, Detection, emit


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hornsby Autonomous 3rd-Person tracking script")
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
    parser.add_argument(
        "--orange-ball",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable orange ping pong ball classifier",
    )
    parser.add_argument(
        "--orange-threshold",
        type=float,
        default=0.58,
        help="Minimum orange ping pong ball classifier confidence",
    )
    parser.add_argument(
        "--orange-model",
        default="models/orange_ping_pong_classifier.json",
        help="Optional learned prototype model for orange ball validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Register gracefull exit for SIGTERM signal (used by Electron to terminate child)
    def sigterm_handler(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    # Telemetry accumulator variables
    session_start = time.time()
    total_frames = 0
    ball_detections = 0
    yolo_hits = 0
    custom_hits = 0
    
    actions_count = {
        "SEARCHING": 0,
        "STEER LEFT": 0,
        "STEER RIGHT": 0,
        "DRIVE FORWARD": 0,
        "ESTOP (ARRIVED)": 0
    }
    
    confidences = []
    orange_scores = []
    roundnesses = []
    validation_scores = []
    
    # 1. Initialize models
    model_path = Path(args.model)
    orange_model = Path(args.orange_model)
    
    print(f"[*] Loading YOLO26 MLX model: {model_path}...", file=sys.stderr, flush=True)
    try:
        model = load_model(model_path)
        print("[+] YOLO26 MLX model loaded successfully!", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[-] Failed to load YOLO model: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    print("[*] Initializing Orange Ball Classifier...", file=sys.stderr, flush=True)
    classifier = load_orange_classifier(args.orange_ball, str(orange_model), args.orange_threshold)
    print("[+] Orange Ball Classifier initialized!", file=sys.stderr, flush=True)

    # 2. Open Camera
    source_val = args.source
    if source_val.isdigit():
        camera_index = int(source_val)
        print(f"[*] Accessing Camera (index {camera_index})...", file=sys.stderr, flush=True)
        capture = cv2.VideoCapture(camera_index)
    else:
        print(f"[*] Loading video/image source: {source_val}...", file=sys.stderr, flush=True)
        capture = cv2.VideoCapture(source_val)
        
    if not capture.isOpened():
        print(f"[-] Error: Could not open source {source_val}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    print("[+] Camera stream opened successfully!", file=sys.stderr, flush=True)
    print("--------------------------------------------------", file=sys.stderr, flush=True)

    try:
        while True:
            loop_start = time.time()
            ok, frame = capture.read()
            if not ok:
                break

            total_frames += 1
            height, width = frame.shape[:2]
            
            # 3. Run YOLO26 Inference (detect robot/environment objects)
            results = model.predict(frame, conf=args.conf)
            yolo_dets = normalize_result(results[0]) if results else []
            
            # 4. Run Orange Ball Classifier (detect ping pong ball)
            balls = classifier.detect(frame) if classifier else []
            
            # 4b. Integrate YOLO's native class 49 ("orange") detections into ball candidates
            for det in yolo_dets:
                if det.label == "orange_ping_pong_ball":
                    from hornsby_ai.orange_ball_classifier import OrangeBallCandidate, _iou
                    if all(_iou(det.box, b.box) < 0.40 for b in balls):
                        balls.append(
                            OrangeBallCandidate(
                                label="orange_ping_pong_ball",
                                confidence=det.confidence,
                                box=det.box,
                                source_model="yolo26",
                                orange_score=1.0,
                                roundness=1.0,
                                validation_score=1.0
                            )
                        )
            
            # 5. Extract Best Target Ball
            target_ball = None
            if balls:
                target_ball = max(balls, key=lambda b: b.confidence)
                
            # 6. Autonomous Steering & Action Logic
            action = "SEARCHING"
            action_color = (0, 165, 255) # Orange
            detail_msg = "Scanning environment for orange ping pong ball..."
            target_center_x = None
            target_center_y = None
            relative_width = 0.0

            if target_ball:
                ball_detections += 1
                if target_ball.source_model == "yolo26":
                    yolo_hits += 1
                else:
                    custom_hits += 1
                
                confidences.append(target_ball.confidence)
                orange_scores.append(target_ball.orange_score)
                roundnesses.append(target_ball.roundness)
                validation_scores.append(target_ball.validation_score)

                x1, y1, x2, y2 = target_ball.box
                target_center_x = int((x1 + x2) / 2)
                target_center_y = int((y1 + y2) / 2)
                relative_x = target_center_x / width
                relative_width = (x2 - x1) / width

                # Auto-isolate crop for validation dataset collection
                if not hasattr(main, "last_save_time"):
                    main.last_save_time = 0
                current_time = time.time()
                if current_time - main.last_save_time > 2.0:  # Save at most 1 crop per 2 seconds
                    main.last_save_time = current_time
                    bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        crop_dir = ROOT / "results" / "isolated_detections"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_id = str(uuid.uuid4())[:8]
                        crop_filename = f"results/isolated_detections/crop_{crop_id}.jpg"
                        crop_full_path = crop_dir / f"crop_{crop_id}.jpg"
                        cv2.imwrite(str(crop_full_path), crop)
                        
                        # Generate JSONL manifest entry
                        manifest_entry = {
                            "image": crop_filename,
                            "box": [0, 0, crop.shape[1], crop.shape[0]],
                            "label": 1  # 1 represents positive target ball sample
                        }
                        manifest_path = crop_dir / "manifest.jsonl"
                        with open(manifest_path, "a") as mf:
                            mf.write(json.dumps(manifest_entry) + "\n")
                        
                        print(f"[AUTO] Crop isolated & saved: {crop_filename}", file=sys.stderr, flush=True)
                
                # Decision guidelines
                if relative_width > 0.35:
                    action = "ESTOP (ARRIVED)"
                    action_color = (0, 0, 255) # Red
                    detail_msg = f"Target reached! Ball is close ({relative_width:.1%}). Halting motors."
                elif relative_x < 0.40:
                    action = "STEER LEFT"
                    action_color = (255, 191, 0)
                    detail_msg = f"Ball at {relative_x:.1%} X (LEFT corridor). Steering left."
                elif relative_x > 0.60:
                    action = "STEER RIGHT"
                    action_color = (255, 191, 0)
                    detail_msg = f"Ball at {relative_x:.1%} X (RIGHT corridor). Steering right."
                else:
                    action = "DRIVE FORWARD"
                    action_color = (0, 255, 0)
                    detail_msg = f"Ball centered at {relative_x:.1%} X. Driving forward to target."
            
            # Increment action statistics
            actions_count[action] = actions_count.get(action, 0) + 1
            
            # Print decision to stderr so it shows up in Electron logs panel
            print(f"[AUTO] Status: {action.ljust(15)} | {detail_msg}", file=sys.stderr, flush=True)

            # 7. Construct PerceptionFrame JSON
            detections = []
            for det in yolo_dets:
                detections.append(det)
            for ball in balls:
                detections.append(
                    Detection(
                        label=ball.label,
                        confidence=ball.confidence,
                        box=ball.box,
                        source_model=ball.source_model,
                        orange_score=ball.orange_score,
                        roundness=ball.roundness,
                        validation_score=ball.validation_score
                    )
                )

            # Emit structured JSON on stdout (for Electron renderer drawing bounding boxes)
            frame_data = PerceptionFrame(
                source=str(args.source),
                width=width,
                height=height,
                timestamp=time.time(),
                detections=detections
            )
            emit(frame_data)

            # 8. Render local CV2 HUD window if requested
            if args.show:
                # Draw tracking corridor lines
                cv2.line(frame, (int(width * 0.40), 0), (int(width * 0.40), height), (80, 80, 80), 2, cv2.LINE_AA)
                cv2.line(frame, (int(width * 0.60), 0), (int(width * 0.60), height), (80, 80, 80), 2, cv2.LINE_AA)
                
                # Draw YOLO detections
                for det in yolo_dets:
                    bx1, by1, bx2, by2 = [int(v) for v in det.box]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (235, 206, 135), 2)
                    cv2.putText(frame, f"{det.label} {det.confidence:.1%}", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 206, 135), 1, cv2.LINE_AA)

                # Draw Orange Ball detections
                for ball in balls:
                    bx1, by1, bx2, by2 = [int(v) for v in ball.box]
                    is_target = (target_ball and ball.box == target_ball.box)
                    box_color = (0, 102, 255) if is_target else (0, 191, 255)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 3 if is_target else 2)
                    cv2.putText(frame, f"Ball {ball.confidence:.0%}", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2 if is_target else 1, cv2.LINE_AA)
                    
                if target_center_x is not None:
                    cv2.drawMarker(frame, (target_center_x, target_center_y), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
                    cv2.circle(frame, (target_center_x, target_center_y), 12, (0, 0, 255), 2)
                    
                    screen_center_x = int(width / 2)
                    cv2.arrowedLine(frame, (screen_center_x, height - 20), (target_center_x, target_center_y), action_color, 3, tipLength=0.05)

                cv2.rectangle(frame, (0, 0), (width, 60), (15, 15, 15), -1)
                cv2.putText(frame, "HORNSBY 3RD-PERSON AUTO CONTROL ACTIVE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(frame, f"ACTION: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2, cv2.LINE_AA)
                
                loop_duration = time.time() - loop_start
                stat_text = f"FPS: {1.0 / max(0.001, loop_duration):.1f} | Targets: {len(balls)}"
                cv2.putText(frame, stat_text, (width - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

                cv2.imshow("Hornsby Autonomous Tracking (3rd Person)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.frame_interval > 0:
                # Add frame throttling
                time.sleep(args.frame_interval)
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
            
        # Compile and write session report
        duration = time.time() - session_start
        report_path = ROOT / "results" / "session_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        avg_conf = np.mean(confidences) if confidences else 0.0
        avg_orange = np.mean(orange_scores) if orange_scores else 0.0
        avg_round = np.mean(roundnesses) if roundnesses else 0.0
        avg_val = np.mean(validation_scores) if validation_scores else 0.0
        
        report_content = f"""==================================================
              HORNSBY CALIBRATION REPORT                 
==================================================
Session Duration: {duration:.1f} seconds
Total Frames Processed: {total_frames}
Ball Detections: {ball_detections} (Detection Rate: {ball_detections/max(1, total_frames):.1%})

Tracking Source Breakdown:
- Custom HSV/Shape Detector Hits: {custom_hits}
- Deep Learning YOLO Class 49 Hits: {yolo_hits}

Average Target Quality Metrics:
- Average Target Confidence: {avg_conf:.1%}
- Average Orange Color Density: {avg_orange:.1%}
- Average Shape Roundness: {avg_round:.1%}
- Average ML Validation Score: {avg_val:.1%}

Autonomous Action Statistics:
- Searching: {actions_count.get('SEARCHING', 0)}
- Steer Left: {actions_count.get('STEER LEFT', 0)}
- Steer Right: {actions_count.get('STEER RIGHT', 0)}
- Drive Forward: {actions_count.get('DRIVE FORWARD', 0)}
- Arrived (ESTOP): {actions_count.get('ESTOP (ARRIVED)', 0)}

Optimal Calibration Variables:
- GUI Confidence Threshold: {args.conf}
- GUI Orange Threshold: {args.orange_threshold}
- GUI Frame Interval: {args.frame_interval}s
==================================================
"""
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(report_content)
            
        # Print a short summary to stderr as well so they see it in the Electron logs panel
        print(f"\n[AUTO] Calibration report saved to results/session_report.txt!", file=sys.stderr, flush=True)
        print(f"[AUTO] Total Frames: {total_frames} | Ball Detections: {ball_detections}", file=sys.stderr, flush=True)


def load_orange_classifier(enabled: bool, model_path: str, threshold: float) -> Any | None:
    if not enabled:
        return None
    try:
        from hornsby_ai.orange_ball_classifier import OrangeBallClassifier
    except ImportError:
        from orange_ball_classifier import OrangeBallClassifier
    return OrangeBallClassifier(model_path=model_path, threshold=threshold)


if __name__ == "__main__":
    main()
