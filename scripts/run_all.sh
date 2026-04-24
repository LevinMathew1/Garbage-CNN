#!/usr/bin/env bash
# Full training + evaluation pipeline.
# Run from project root: bash scripts/run_all.sh

set -e

DATA_DIR="Garbage classification"

echo "=== Training CustomCNN ==="
python src/train.py --model custom_cnn --data-dir "$DATA_DIR" --epochs 30 --batch-size 64 --lr 1e-3

echo "=== Evaluating CustomCNN ==="
python src/evaluate.py --model custom_cnn --data-dir "$DATA_DIR"

echo "=== Training MobileNetV2 ==="
python src/train.py --model mobilenet_v2 --data-dir "$DATA_DIR" --epochs 25 --batch-size 64 --lr 1e-3

echo "=== Evaluating MobileNetV2 ==="
python src/evaluate.py --model mobilenet_v2 --data-dir "$DATA_DIR"

echo "=== Comparing Models ==="
python scripts/compare_models.py

echo "=== Pipeline complete. Check outputs/ for results. ==="
