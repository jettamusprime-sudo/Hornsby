#!/usr/bin/env python3
"""Train Hornsby's orange ping pong ball prototype classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hornsby_ai.orange_ball_classifier import train_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train orange ping pong ball validation classifier")
    parser.add_argument("manifest", help="JSONL file with image, box, and label fields")
    parser.add_argument(
        "--output",
        default="models/orange_ping_pong_classifier.json",
        help="Output classifier JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = train_from_manifest(args.manifest, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
