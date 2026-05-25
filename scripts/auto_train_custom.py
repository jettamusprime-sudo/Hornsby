#!/usr/bin/env python3
"""Automatically compiles uploaded ball photos, generates a manifest with negative controls, and trains the model."""

from __future__ import annotations

import sys
import shutil
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hornsby_ai.orange_ball_classifier import OrangeBallClassifier, train_from_manifest

def main():
    print("[*] Starting custom training pipeline...")
    
    # 1. Source image paths (system saved uploads)
    media_dir = Path("/Users/adeptjettamusprime/.gemini/antigravity/brain/4e1f969a-35ab-47e1-907e-6d2e8e5d43e0")
    uploaded_files = sorted(list(media_dir.glob("media__*.jpg")))
    
    if not uploaded_files:
        # Fallback to tempmediaStorage if glob is empty
        uploaded_files = sorted(list((media_dir / ".tempmediaStorage").glob("*.jpg")))
        
    if not uploaded_files:
        print("[-] Error: No uploaded images found in system app data!")
        sys.exit(1)
        
    print(f"[+] Found {len(uploaded_files)} uploaded images.")
    
    # 2. Setup directories
    dest_dir = ROOT / "results" / "custom_training"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_entries = []
    
    # Initialize our optimized classifier to find the ball bounding boxes automatically
    classifier = OrangeBallClassifier(threshold=0.55)
    
    # 3. Process each uploaded image
    for i, file_path in enumerate(uploaded_files, 1):
        print(f"[*] Processing image {i}: {file_path.name}")
        
        # Load image
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"  [-] Failed to read image: {file_path.name}")
            continue
            
        # Save a copy in our workspace
        workspace_img_name = f"ball_{i}.jpg"
        workspace_img_path = dest_dir / workspace_img_name
        shutil.copy(str(file_path), str(workspace_img_path))
        
        # Detect orange ball automatically
        balls = classifier.detect(img)
        # Ensure we don't pick up massive background contours (like the whole image) as the "ball"
        h, w = img.shape[:2]
        valid_balls = [b for b in balls if (b.box[2] - b.box[0]) < w * 0.50]
        
        if valid_balls:
            best_ball = max(valid_balls, key=lambda b: b.confidence)
            x1, y1, x2, y2 = best_ball.box
            
            # Save positive entry (1)
            manifest_entries.append({
                "image": f"results/custom_training/{workspace_img_name}",
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "label": 1
            })
            print(f"  [+] Located orange ball at box: {[int(v) for v in best_ball.box]}")
            
            # Automatically extract a 100x100 background patch as a Negative Control (0)
            # Take it from the top-left or bottom-right corner where the ball is NOT located
            h, w = img.shape[:2]
            neg_x, neg_y = 10, 10
            # If the ball is close to top-left, take bottom-right
            if x1 < w * 0.3 and y1 < h * 0.3:
                neg_x, neg_y = w - 110, h - 110
                
            neg_crop = img[neg_y:neg_y+100, neg_x:neg_x+100]
            neg_crop_name = f"neg_{i}.jpg"
            neg_crop_path = dest_dir / neg_crop_name
            cv2.imwrite(str(neg_crop_path), neg_crop)
            
            # Save negative entry (0)
            manifest_entries.append({
                "image": f"results/custom_training/{neg_crop_name}",
                "box": [0.0, 0.0, 100.0, 100.0],
                "label": 0
            })
            print(f"  [+] Saved negative control crop from background: {neg_crop_name}")
        else:
            # Fallback color thresholding if strict classifier missed it due to new environment
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([3, 75, 50], dtype=np.uint8)
            upper = np.array([24, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest)
                manifest_entries.append({
                    "image": f"results/custom_training/{workspace_img_name}",
                    "box": [float(x), float(y), float(x + w_box), float(y + h_box)],
                    "label": 1
                })
                print(f"  [+] Located orange ball via fallback thresholding at: {[x, y, x+w_box, y+h_box]}")
                
                # Extract negative control
                neg_crop = img[10:110, 10:110]
                neg_crop_name = f"neg_{i}.jpg"
                neg_crop_path = dest_dir / neg_crop_name
                cv2.imwrite(str(neg_crop_path), neg_crop)
                
                manifest_entries.append({
                    "image": f"results/custom_training/{neg_crop_name}",
                    "box": [0.0, 0.0, 100.0, 100.0],
                    "label": 0
                })
            else:
                print("  [-] Warning: No orange contours found in this image, skipping.")
                
    # 4. Write manifest file
    manifest_path = dest_dir / "custom_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"[+] Custom manifest compiled successfully at: {manifest_path.name}")
    print("[*] Training prototype centroids model from manifest...")
    
    # 5. Execute Centroid Training
    output_model_path = ROOT / "models" / "orange_ping_pong_classifier.json"
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = train_from_manifest(manifest_path, output_model_path)
    
    print("==================================================")
    print("      TRAINING COMPLETED SUCCESSFULLY             ")
    print("==================================================")
    print(f"Positive Samples trained: {payload.get('positive_samples')}")
    print(f"Negative Control samples: {payload.get('negative_samples')}")
    print(f"Positive Centroid vector: {payload.get('positive_centroid')}")
    print(f"Negative Centroid vector: {payload.get('negative_centroid')}")
    print(f"Model saved to: models/orange_ping_pong_classifier.json")
    print("==================================================")

if __name__ == "__main__":
    main()
