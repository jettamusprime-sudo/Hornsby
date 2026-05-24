#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
YOLO_MLX="$ROOT/third_party/yolo-mlx"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup is intended for macOS on Apple Silicon."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "MLX requires Apple Silicon. Current architecture: $(uname -m)"
  exit 1
fi

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements-macos-mlx.txt"

mkdir -p "$ROOT/third_party" "$ROOT/models" "$ROOT/results"

if [[ ! -d "$YOLO_MLX/.git" ]]; then
  git clone --depth 1 https://github.com/thewebAI/yolo-mlx.git "$YOLO_MLX"
fi

python -m pip install -e "$YOLO_MLX"
python -m pip install -e "$YOLO_MLX[convert]"

if [[ ! -f "$ROOT/models/yolo26n.pt" ]]; then
  (cd "$YOLO_MLX" && bash scripts/download_yolo26_models.sh --model n)
  cp "$YOLO_MLX/models/yolo26n.pt" "$ROOT/models/yolo26n.pt"
fi

if [[ ! -f "$ROOT/models/yolo26n.npz" ]]; then
  yolo-mlx converters convert "$ROOT/models/yolo26n.pt" -o "$ROOT/models/yolo26n.npz" --verify
fi

echo "YOLO26 MLX is ready."
echo "Activate with: source .venv/bin/activate"
echo "Test with: python hornsby_ai/physical_ai.py --source examples/sample.jpg"
